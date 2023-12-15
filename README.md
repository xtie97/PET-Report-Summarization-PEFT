# Automatic Impression Generation for Positron Emission Tomography Reports using Lightweight Adaptation of Pretrained Large Language Models :bookmark_tabs:

This repository contains the code for our team project (member: Nuohao Liu, Xin Tie, Xiaogeng Liu) as part of the **CS776** Advanced Natural Language Processing course. Check our [**presentation**](https://github.com/xtie97/PET-Report-Summarization-PEFT/blob/main/presentation.pdf) 📜.

## Overview :mag_right:
**Background**: 
Adapting LLMs for PET report summarization can be quite expensive in terms of computational time and memory useage. Parameter Efficient Fine-Tuning (PEFT) presents a promising alternative that could retain high performance while utilizing fewer training resources. In this project, we aim to evaluate the effectiveness of PEFT in fine-tuning LLMs for summarizing PET findings. Our ultimate goal is to address the challenge of increasing memory when training a LLM for summarizing multiple radiology reports. 

## Usage 🚀
We investigated three PEFT techniques: 
- LoRA
- (IA)3
- Prompt tuning 

The training was powered by [**deepspeed**](https://github.com/microsoft/DeepSpeed)

To run the training
```bash
python finetune_FlanT5.py
```

To run the prediction 
```bash
python predict_FlanT5.py
```

To test the output impressions using ROUGE
```bash
python compute_rouge.py
```
