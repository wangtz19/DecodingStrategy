from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch

torch.manual_seed(42)

# Correction with Backtracking (CoBa)
class CoBa:
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
                 text: Union[List[str], str], # shape: (batch_size, ...)
                 delta: float = 0.3, # probability threshold
                 phi: float = 0.9, # max cosine similarity threshold
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
        inputs = self.tokenizer(text, return_tensors='pt', padding=True,
                                truncation=True, max_length=max_length,
                                add_special_tokens=False)
        # sanity check
        assert delta < 1.0, "probability threshold must be less than 1.0"
        assert 0 <= phi <= 1.0, "cosine similarity threshold must be between 0 and 1.0"
        assert max_new_tokens > 0, "max_new_tokens must be greater than 0"

        input_ids = inputs['input_ids'].to(self.model.device) # shape: (batch_size, seq_len)
        assert input_ids.shape[0] == 1, "batch_size must be 1 for CoBa, since we need to backtrack"
        attention_mask = inputs['attention_mask'].to(self.model.device)

        input_token_len = input_ids.shape[1]
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
        total_searched_tokens = 0
        last_token = None
        with torch.no_grad():
            while not this_peer_finished:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                next_token_logits = outputs.logits[:, -1, :] # shape: (1, vocab_size)

                # repetition penalty
                if repetition_penalty != 1.0:
                    generated_ids = input_ids[:, input_token_len:]
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
                
                # get the most probable next token
                probs = F.softmax(next_token_logits, dim=-1) # shape: (1, vocab_size)
                if last_token is not None and total_searched_tokens <= 10 * max_new_tokens:
                    # print(f"Last token: {self.tokenizer.decode(last_token)}")
                    # print(f"Probability of last token: {probs[:, last_token].item()} set to 0.0")
                    probs[:, last_token] = 0.0 # set the probability of last hallucinated token to 0
                try:
                    if do_sample:
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # shape: (1, )
                    else:
                        next_token = torch.argmax(probs, dim=-1)
                    next_token_prob = probs.gather(1, next_token.unsqueeze(-1)).squeeze(-1) # shape: (1, )

                    # check hallucination
                    # 1. probability threshold
                    is_hallucination = next_token_prob.item() < delta
                    # 2. cosine similarity threshold
                    # TODO: implement cosine similarity threshold

                except: # if all remaining tokens have probability 0
                    if new_token_count > 0 and total_searched_tokens <= 10 * max_new_tokens:
                        is_hallucination = True
                    else:
                        is_hallucination = False
                        probs = F.softmax(next_token_logits, dim=-1) # shape: (1, vocab_size)
                        if do_sample:
                            next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # shape: (1, )
                        else:
                            next_token = torch.argmax(probs, dim=-1)

                total_searched_tokens += 1
                if is_hallucination and new_token_count > 0 and total_searched_tokens <= 10 * max_new_tokens:
                    # print("Hallucination detected! Backtracking...")
                    # print(f"Remove the {new_token_count}-th token: {self.tokenizer.decode(next_token.item())}")
                    # remove the last token
                    last_token = input_ids[:, -1].item()
                    input_ids = input_ids[:, :-1]
                    attention_mask = attention_mask[:, :-1]
                    new_token_count -= 1
                    continue
                else:
                    last_token = None # reset last_token_idx
                
                # update input_ids, attention_mask
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)

                # check if the next token is eos_token_id
                if eos_token_id is not None:
                    for eos_token in eos_token_id_tensor:
                        if next_token.item() == eos_token.item():
                            unfinished_sequences.fill_(0)
                            this_peer_finished = True

                # check max length
                new_token_count += 1
                if new_token_count >= max_new_tokens:
                    this_peer_finished = True

        generated_ids = input_ids[:, input_token_len:]
        return self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
