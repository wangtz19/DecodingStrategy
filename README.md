# Optimized Decoding Strategies for Large Language Models
We provide unofficial implementations for decoding strategies that are recently proposed to mitigate LLM hallucinations but without open codes.

## Supported strategies
* CAD, [Trusting Your Evidence: Hallucinate Less with Context-aware Decoding](https://arxiv.org/abs/2305.14739)
* CoBa, [Correction with Backtracking Reduces Hallucination in Summarization](https://arxiv.org/pdf/2310.16176.pdf)
* TODO

## Examples
* CAD
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

* CoBa
```python
from coba import CoBa
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", torch_dtype=torch.float16, )
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False, padding_side="left", )
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1
coba = CoBa(model=model, tokenizer=tokenizer)

text = """
Scunthorpe midfielder Neal Bishop has signed a one-year contract extension. The 35-year-old 
joined the Iron from Blackpool in 2013 and has made 119 league appearances for the League One side. He 
helped them to a third-placed finish this season, before they were beaten by Millwall in the play-off semi-finals. 
Bishop told the club website: "With the way the season finished, it's a sense of unfinished business and it was 
disappointing for all of us."
Summarize above sentences.
"""

# without coba
normal_output = coba.generate(
    text=text,
    delta=-1, # delta <= 0 means no coba
)
print(normal_output)

# with coba
coba_output = coba.generate(
    text=text,
)
print(coba_output)
```