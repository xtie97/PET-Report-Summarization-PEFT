# Automatic Impression Generation for Positron Emission Tomography Reports using Lightweight Adaptation of Pretrained Large Language Models:bookmark_tabs:

This repository contains the code for our team project (Nuohao Liu, Xin Tie, Xiaogeng Liu) for the course CS776 Advanced NLP. 

## Overview :mag_right:
**Background**: 
Adapting LLMs for PET report summarization can be quite expensive in terms of computational time and memory useage. Parameter Efficient Fine-tuning (PEFT) can potentially maintain the performance with much fewer training recourses. In this project, we aim to evaluate the effectiveness of PEFT in fine-tuning LLMs for summarizing PET findings. Our end goal is to solve the problem of summarizing multiple radiology reports.  

Fine-tuning Large Language Models (LLMs) for the purpose of summarizing Positron Emission Tomography (PET) reports often requires significant computational resources and time. However, Parameter Efficient Fine-tuning (PEFT) presents a promising alternative that could retain high performance while utilizing considerably fewer training resources. This project is focused on assessing the efficacy of PEFT in the context of fine-tuning LLMs specifically for PET report summarization. Our ultimate objective is to address the challenge of efficiently summarizing multiple radiology reports.


## Usage 🚀
We investigated three PEFT techniques: 
- LoRA
- (IA)3
- Prompt tuning 

The training was powered by [**deepspeed**]([https://github.com/microsoft/DeepSpeed])
```bash
finetuned_model = "xtie/PEGASUS-PET-impression"
tokenizer = AutoTokenizer.from_pretrained(finetuned_model) 
model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model, ignore_mismatched_sizes=True).eval()

findings_info =
"""
Description: PET CT WHOLE BODY
Radiologist: James
Findings:
Head/Neck: xxx Chest: xxx Abdomen/Pelvis: xxx Extremities/Musculoskeletal: xxx
Indication:
The patient is a 60-year old male with a history of xxx
"""

inputs = tokenizer(findings_info.replace('\n', ' '),
                  padding="max_length",
                  truncation=True,
                  max_length=1024,
                  return_tensors="pt")
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")
outputs = model.generate(input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512, 
                        num_beam_groups=1,
                        num_beams=4, 
                        do_sample=False,
                        diversity_penalty=0.0,
                        num_return_sequences=1, 
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                        )
# get the generated impressions
output_str = tokenizer.decode(outputs,
                              skip_special_tokens=True)

```

## Human Evaluation :busts_in_silhouette:
We released the webpage designed for expert review. Feel free to check it out. :point_right: [PET-Report-Expert-Evaluation](https://github.com/xtie97/PET-Report-Expert-Evaluation)

## Citation 📚

