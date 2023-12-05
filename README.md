# Optimized Decoding Strategies for Large Language Models
We provide unofficial implementations for decoding strategies that are recently proposed to mitigate LLM hallucinations but without open codes.

## Supported strategies
* CAD, [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739)
* TODO

## Examples
```python
from cad import CAD
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", torch_dtype=torch.float16, )
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False, padding_side="left", )
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
cad = CAD(model=model, tokenizer=tokenizer)

x = "Better late than"
c = 'Write a quote that ends in word "early": '

raw_output = cad.generate(texts=x,)
print(raw_output)

conditioned_output = cad.generate(texts=x, texts_with_context=c+x,)
print(conditioned_output)
```