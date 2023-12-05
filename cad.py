from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch

class CAD:
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,):
        self.model = model
        self.tokenizer = tokenizer

    def _temperature_warp(self, 
                          logits: torch.FloatTensor,
                          temperature: float) -> torch.FloatTensor:
        return logits / temperature
    
    def _top_p_warp(self,
                    logits: torch.FloatTensor,
                    top_p: float) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_k_warp(self,
                    logits: torch.FloatTensor,
                    top_k: int) -> torch.FloatTensor:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _repetition_penalty_warp(self,
                                 input_ids: torch.LongTensor,
                                 logits: torch.FloatTensor,
                                 repetition_penalty: float) -> torch.FloatTensor:
        score = torch.gather(logits, 1, input_ids)
        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(1, input_ids, score)
        return logits
    
    def generate(self,
                 texts: Union[List[str], str], # shape: (batch_size, ...)
                 texts_with_context: Union[List[str], str] = None,
                 alpha: float = 0.5, # cad weight
                 max_length: int = 8096,
                 max_new_tokens: int = 4096,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 temperature: float = 0.2,
                 do_sample: bool = True,
                 repetition_penalty: float = 1.2,
                 pad_token_id: int = None,
                 eos_token_id: int = None,
                 ):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=max_length,
                                add_special_tokens=False)
        input_ids = inputs['input_ids'].to(self.model.device) # shape: (batch_size, seq_len)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        # context aware decoding
        use_cad = texts_with_context is not None
        if use_cad:
            inputs_with_context = self.tokenizer(texts_with_context, return_tensors='pt', 
                                                 padding=True, truncation=True,
                                                 max_length=max_length, add_special_tokens=False)
            input_ids_with_context = inputs_with_context['input_ids'].to(self.model.device)
            attention_mask_with_context = inputs_with_context['attention_mask'].to(self.model.device)
        input_token_len = input_ids_with_context.shape[1] if use_cad else \
                            input_ids.shape[1]
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.model.device) if eos_token_id is not None else None
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(inputs['input_ids'].shape[0], dtype=torch.long, 
                                          device=self.model.device) # shape: (batch_size, )
        this_peer_finished = False
        new_token_count = 0
        with torch.no_grad():
            while not this_peer_finished:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                next_token_logits = outputs.logits[:, -1, :] # shape: (batch_size, vocab_size)
                if use_cad:
                    outputs_with_context = self.model(
                        input_ids=input_ids_with_context,
                        attention_mask=attention_mask_with_context,
                    )
                    next_token_logits_with_context = outputs_with_context.logits[:, -1, :]
                    next_token_logits = (1 + alpha) * next_token_logits - alpha * next_token_logits_with_context

                # repetition penalty
                if repetition_penalty != 1.0:
                    generated_ids = input_ids_with_context[:, input_token_len:] if use_cad else \
                                        input_ids[:, input_token_len:]
                    next_token_logits = self._repetition_penalty_warp(
                        input_ids=generated_ids,
                        logits=next_token_logits,
                        repetition_penalty=repetition_penalty,
                    )
                
                # top-k sampling
                if top_k is not None and top_k > 0:
                    next_token_logits = self._top_k_warp(
                        logits=next_token_logits,
                        top_k=top_k,
                    )
                
                # top-p sampling
                if top_p is not None and top_p < 1.0:
                    next_token_logits = self._top_p_warp(
                        logits=next_token_logits,
                        top_p=top_p,
                    )
                
                # temperature sampling
                if temperature is not None and temperature != 0:
                    next_token_logits = self._temperature_warp(
                        logits=next_token_logits,
                        temperature=temperature,
                    )
                
                # sample
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # update input_ids, attention_mask
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)
                if use_cad:
                    input_ids_with_context = torch.cat([input_ids_with_context, next_tokens.unsqueeze(-1)], dim=-1)
                    attention_mask_with_context = torch.cat([attention_mask_with_context, unfinished_sequences.unsqueeze(-1)], dim=-1)

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )

                    # stop when each sentence is finished
                    if unfinished_sequences.max() == 0:
                        this_peer_finished = True

                # check max length
                new_token_count += 1
                if new_token_count >= max_new_tokens:
                    this_peer_finished = True

        generated_ids = input_ids_with_context[:, input_token_len:] if use_cad else \
                            input_ids[:, input_token_len:]
        return self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
