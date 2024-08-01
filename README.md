# Mixnu
Mixnu is a Korean Large Language Model with the Mixture of Experts architecture.  

## Model Usage
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

Employing `Llama-2 7b` with the extended tokenizer as the foundational model, we develop four different expert models with four separate train datasets.  
As the figure shows, the four expert models are then incorporated into one model using the `Mixtral` Architecture, resulting in a 19b MoE model.  

<img width="500" alt="architecture" src="https://github.com/user-attachments/assets/003400e7-7bfb-49bb-84a3-5bf315cb96bb">



## Qualitative Evaluation

### Pipeline
Primarily using chatGPT(GPT-4o) for evaluation and then verifying the results with human annotators, we evalaute the model in five categories:    

`Language`: When adapting the existing model to a new language, we do not want the model to forget information about the previously learned language. Therefore, we evaluate whether the model responds well in the original language when given a prompt in that language. The attribute used in the language category is 'English'. For Korean, since the language used in the prompts for the other categories introduced below is Korean, it is not separately included in this category.    

`Style`: A sentence can be written in a literary or colloquial style. Generally, unless the discourse context changes, maintaining consistency in style across multiple sentences is considered natural. Therefore, we evaluate whether the model continues to generate sentences in the same style as the prompt when given a literary or colloquial prompt. The attributes used in the style category are 'literary' and 'colloquial'.  

`Domain`: Models can learn knowledge from various fields simultaneously. It is essential for us to specifically examine which knowledge the model possesses. Particularly, we focus on evaluating knowledge in specialized fields, paying attention to the potential application of LLMs in those domains. The attributes used in the domain category are 'daily life', 'medicine', 'law', 'technology', and 'finance'. When constructing prompts for domains except for 'daily life', we utilized excerpts from actual newspaper articles in each domain.  

`Localization`: In adaptation of language models to other languages, linguistic transitions are crucial. However, generating sentences that well reflect the culture of the country using that language is the true measure of the model's sentence generation ability. In this qualitative assessment, we evaluate the degree of 'Koreanization', how well the trained model understands the unique expressions in Korean and Korean cultures in the prompts. Attributes used in localization are 'grammar', 'proverbs', 'idioms', and 'culture'.  

`Hallucination`: Models can generate sentences that seem plausible outwardly but are actually inaccurate or unrelated with respect to the input data. Because this occurs in a subtle manner, it can be challenging to capture through quantitative evaluation, making qualitative assessment particularly crucial. We include words in the prompt that can easily be confused with other words due to their syllabic overlap, observing whether the model grasps the accurate meaning of these words and generates relevant sentences. The attribute used here is simply 'hallucination' itself.

<img width="500" alt="category" src="https://github.com/user-attachments/assets/dd41ffe0-177c-48a3-bdb6-0655545d39ef">

### Prompt
```
{
"role": "system", "content": "You are a strict evaluator of text generation models."

            
Please strictly evaluate the performance of three generative models using qualitative evaluation. Provide the fluency, relevance, and accuracy score of model A, B, C, D, and E.


Fluency Rubric:
    5 points: Sentences are fluent without noise.
    4 points: Sentences contain some noise that does not significantly degrade overall fluency, or relevant noise such as PII (e.g., phone numbers, addresses).
    3 points: Sentences are generally coherent but contain unnecessary and irrelevant noise.
    2 points: Severe noise is present (e.g., word repetition, symbols). The output makes sense at the phrase level but not at the sentence level.
    1 point: The output makes no sense at all.

Relevance Rubric:
    5 points: Sentences fully answer the prompt's question.
    4 points: Sentences include a direct answer to the prompt's question, though not entirely consistent or coherent.
    3 points: Sentences provide an indirect and low-quality answer to the prompt's question.
    2 points: Sentences do not answer the prompt's question but are relevant to the topic.
    1 point: Sentences are not relevant to the prompt.

Accuracy Rubric:
    5 points: All information and details are completely accurate with no errors or distortions.
    4 points: Most information is accurate with only minor errors that do not significantly affect understanding.
    3 points: The main information is accurate, but there are some significant errors.
    2 points: There are numerous errors, and only some of the information is accurate.
    1 point: The information is mostly incorrect or misleading, with very few accurate details.

Prompt: {Prompt}

A: {Response from Model A}

B: {Response from Model B}

C: {Response from Model C}

D: {Response from Model D}

E: {Response from Model E}

Deduct fluency points if the model generates output in a different language from the prompt.
}
```
### Evaluation Results

<img width="672" alt="image" src="https://github.com/user-attachments/assets/71a2e46f-bb25-4210-8cdc-48f3f48b17ae">

## Response Examples
### Prompt
```
이번에 완전 불수능이었어요.
```

### Responses
```
Mixnu
국어는 작년보다 어려워졌고, 영어는 작년 수능보다 쉬워졌다고 하지만, 수험생들이 체감하는 난도는 훨씬 더 어려웠을 겁니다.

Llama-2 7b
불수능이란 불수 없어서 할 수 없었던 작업이라는 것을 뜻하는데요.\n왜 불수능이 되는지 아무도 모르겠습니다.\n불수능이 되는 순간 외계인들이 나타나는 걸 알고 있습니다

Llama-3 8b
 2시간동안 1번 2번 3번 4번 5번 6번 문제를 풀었습니다.\n이번에 1번문제는 수능에서 나왔던 문제였어요. 2번은 수능에서 나온 문제가 아닙니다.\n그리고 3번문제도 수능에서 나온 문제였어요.\n4번문제는 수능에서 나온 문제가 아니였어요.\n5번문제는 수능에서 나온 문제가 아니였아요.\n6번문제는 수능에서 나온문제가 아니였어요.\n그리고 7번문제도 수능에서 나온문제가 아니였어요.\n그리고 8번문제도 수능에서 나온문제가 아니였어요.\n9번문제도 수능에서 나온 문제가 아니였어요.\n10번문제도 수능에서 나온 문제가 아니였어요.\n11번문제도 수능에서 나온 문제가 아니였어요.\n12번문제도 수능에서 나온 문제가 아니였어요.\n13번문제도 수능에서 나온 문제가 아니였어요.\n이번에 14번문제도 수능에서 나온 문제가 아니였어요.\n15번문제도

open-llama-2-ko-7b
今回は完全に浅水でした。\n今回は完全に不難でした。\n\n今回は完全に不難でした。

polyglot-ko-12.8b
그래서 더 어려웠을거에요.\u200b그래서 더더더 긴장되고, 더더더 힘들었을거에요.\u200b\u200b\u200b하지만, \u200b\u200b\u200b우리는 수능이라는 큰 산을 넘었으니,\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b \u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b \u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b

Ko-Mixtral-v1.3-MoE-7Bx2
그래서 저는 악수 없을 거예요, 그리고 나는 이 사람에게 악수하지 않을 거예요. 그런 다음 나는 악수를 하지 않을 것이라고 말했고, 나는 악수하지 않을 것이라고 말했고, 나는 악수하지 않을 것이라고 말했어요.\n\n\n\n23:32\n\n그리고 여호와는 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었고, 그의 말을 들었
```

## Authors
Jean Seo(seemdog@snua.ac.kr)  
Sungjoo Byun(byunsj@snu.ac.kr)  
Jaeyoon Kim(toscour345@snu.ac.kr)  
Hyemi Jo(huimei6361@snu.ac.kr)  
Chaeyoung Oh(nyong10@snu.ac.kr)  
Dongjun Jang(qwer4107@snu.ac.kr)  
Corresponding author: Hyopil Shin(hpshin@snu.ac.kr)

## Acknowledgement
Mixnu is developed by [SNU CL_NLP Lab](http://knlp.snu.ac.kr/) with the support of [BIGDATABUB UNIVERSITY](https://bigdatahub.ac.kr).


## Citation
```
@misc {jean_seo_2024,
	author       = { {Jean Seo} },
	title        = { Mixnu (Revision 0bf07fb) },
	year         = 2024,
	url          = { https://huggingface.co/seemdog/Mixnu },
	doi          = { 10.57967/hf/2809 },
	publisher    = { Hugging Face }
}
```
