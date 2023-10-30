import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install deepspeed xformers==0.0.16 peft") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ninja accelerate --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install datasets evaluate dill==0.3.4")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers huggingface-hub tokenizers triton")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1 numpy==1.20.3")

import pandas as pd
from tqdm import tqdm 
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import datasets
import numpy as np
import pandas as pd
from peft import PeftModel, PeftConfig
# Command: deepspeed --num_gpus=2 predict_XL.py
# https://github.com/tloen/alpaca-lora/blob/main/generate.py


def predict_text(model_id, reports, lora_weights, savename, test_epoch):
    #model.tie_weights() # Now you can use the model for inference without encountering the error
    config = PeftConfig.from_pretrained(lora_weights)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=False, device_map="auto")  
    #model.config.n_positions = 1024 
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path) # add special tokens
    # Load the Lora model
    model = PeftModel.from_pretrained(model, lora_weights, device_map="auto")  
    model = model.merge_and_unload() # merge the weights and unload the Lora model
    model.eval()
    
    encoder_max_length = 1024
    decoder_max_length = 512
    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        inputs = tokenizer(batch["findings_info"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        # Add model = model.merge_and_unload()
        # If you treat the input as a pointer such as, in this case: .generate(**inputs) that works as well

        outputs = model.generate(input_ids, max_new_tokens=decoder_max_length, \
                                            num_beam_groups=1,\
                                            num_beams=4, \
                                            do_sample=False,\
                                            diversity_penalty=0.0,\
                                            num_return_sequences=1, \
                                            length_penalty=2.0,\
                                            no_repeat_ngram_size=3,\
                                            early_stopping=True) 
    
        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["pred"] = output_str
        return batch

    test_data = datasets.Dataset.from_pandas(reports)
    results = test_data.map(generate_summary, batched=True, batch_size=4)
    pred_str = results["pred"]
    reports['AI_impression'] = pred_str
    
    save_pred_dir = f'{savename}'
    os.makedirs(save_pred_dir, exist_ok=True)
    reports.to_excel(f'{save_pred_dir}/test_{test_epoch}.xlsx', index=False)


def filter_test_data(df):
    # select the recent 5 years data
    exam_date = df['Exam Date Time']
    exam_date_new = []
    for ii in tqdm(range(len(exam_date))):
        if '2018' in exam_date[ii] or '2019' in exam_date[ii] or '2020' in exam_date[ii] \
        or '2021' in exam_date[ii] or '2022' in exam_date[ii] or '2023' in exam_date[ii]:
            exam_date_new.append(exam_date[ii])
        else:
            exam_date_new.append('Remove')
    df['Exam Date Time'] = exam_date_new
    # drop the data with no exam date
    df = df[df['Exam Date Time'] != 'Remove'].reset_index(drop=True)
    df = df[df['Study Description'] == 'PET CT WHOLE BODY'].reset_index(drop=True)
    df = df.sample(n=1000, random_state=716).reset_index(drop=True)
    return df 

if __name__ == '__main__':
    # Testing    
    df = pd.read_excel('./archive/test.xlsx')
    #df = filter_test_data(df)
    
    # clean data
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n', ' '))
    df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n', ' '))  

    text_column = "findings_info" # column of input text is
    summary_column = "impressions" # column of the output text 
    model_id = "google/flan-t5-large" # Hugging Face Model Id
    savename = 'flan_t5_large_lora'

    steps = [525, 1050, 1575, 2100, 2625, 3150, 3675, 4200, 4725, 5250, 5775, 6300, 6825, 7350, 7875, 8400, 8925, 9450, 9975, 10500]
    for test_epoch in [8]:
        lora_weights = f"my-flan-t5-large/checkpoint-{steps[test_epoch]}" 
        predict_text(model_id, df, lora_weights, savename, test_epoch)
    