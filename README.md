# Mixnu
Mixnu is a Korean Large Language Model with the Mixture of Experts architecture.  

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("seemdog/Mixnu")
model = AutoModelForCausalLM.from_pretrained("seemdog/Mixnu")
```

## Tokenizer
Mixnu uses an extended version of Llama-2 tokenizer.  
The table below compares the vocabulary size, percentage of unkown tokens(tokens segmeted in byte-level), and the average number of tokens per test sentence. The test sentences consist of carefully curated Korean text encompassing various domains(e.g. law, medicine, finance, social studies etc.)  
|       Model      | Vocab size | % of UNK token | Avg. num of tokens |
|:----------------:|:----------:|:--------------:|:------------------:|
|  Llama-2 |    32,000   |      48.61     |       325.27       |
|       Mixnu      |   46,262   |      0.96      |       125.91       |

## Model Architecture

## Qualitative Evaluation

## Citation

## Acknowledgement
Mixnu is developed by [SNU CL_NLP Lab](http://knlp.snu.ac.kr/) with the support of [BIGDATABUB UNIVERSITY](https://bigdatahub.ac.kr).
