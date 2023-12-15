import os 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install rouge-score==0.1.2") 
import pandas as pd
from tqdm import tqdm 
import torch
from rouge_score import rouge_scorer
import numpy as np 
import nltk
nltk.download('punkt')

def predict_text(df):
    #Import the pretrained model
    findings = df['findings_info'].tolist()
    findings = [i.replace('\n',' ') for i in findings]
    AI_impression = df['AI_impression'].tolist()
    impression = df['impressions'].tolist() 
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    
    results_rouge1 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge2 = {'precision': [], 'recall': [], 'f': []} 
    results_rouge3 = {'precision': [], 'recall': [], 'f': []} 
    results_rougeL = {'precision': [], 'recall': [], 'f': []} 
    results_rougeLsum = {'precision': [], 'recall': [], 'f': []} 
    for i in tqdm(np.arange(len(impression))):
        gen_text = AI_impression[i]
        gt_text = impression[i]
        
        scores = scorer.score(gen_text, gt_text)
        results_rouge1['precision'].append(list(scores['rouge1'])[0]) 
        results_rouge1['recall'].append(list(scores['rouge1'])[1]) 
        results_rouge1['f'].append(list(scores['rouge1'])[2]) 

        results_rouge2['precision'].append(list(scores['rouge2'])[0]) 
        results_rouge2['recall'].append(list(scores['rouge2'])[1]) 
        results_rouge2['f'].append(list(scores['rouge2'])[2]) 

        results_rouge3['precision'].append(list(scores['rouge3'])[0]) 
        results_rouge3['recall'].append(list(scores['rouge3'])[1]) 
        results_rouge3['f'].append(list(scores['rouge3'])[2]) 

        results_rougeL['precision'].append(list(scores['rougeL'])[0]) 
        results_rougeL['recall'].append(list(scores['rougeL'])[1]) 
        results_rougeL['f'].append(list(scores['rougeL'])[2]) 

        results_rougeLsum['precision'].append(list(scores['rougeLsum'])[0])
        results_rougeLsum['recall'].append(list(scores['rougeLsum'])[1])
        results_rougeLsum['f'].append(list(scores['rougeLsum'])[2])
    
    df['ROUGE1'] = results_rouge1['f']
    df['ROUGE2'] = results_rouge2['f']
    df['ROUGE3'] = results_rouge3['f']
    df['ROUGEL'] = results_rougeL['f']
    df['ROUGELsum'] = results_rougeLsum['f']
    df.to_excel('./PEGASUS_ROUGE_score.xlsx')
    
    results_rouge1['precision'] = np.around(np.mean(results_rouge1['precision']), 3)
    results_rouge1['recall'] = np.around(np.mean(results_rouge1['recall']), 3)
    results_rouge1['f'] = np.around(np.mean(results_rouge1['f']), 3)

    results_rouge2['precision'] = np.around(np.mean(results_rouge2['precision']), 3)
    results_rouge2['recall'] = np.around(np.mean(results_rouge2['recall']), 3)
    results_rouge2['f'] = np.around(np.mean(results_rouge2['f']), 3)

    results_rouge3['precision'] = np.around(np.mean(results_rouge3['precision']), 3)
    results_rouge3['recall'] = np.around(np.mean(results_rouge3['recall']), 3)
    results_rouge3['f'] = np.around(np.mean(results_rouge3['f']), 3)

    results_rougeL['precision'] = np.around(np.mean(results_rougeL['precision']), 3)
    results_rougeL['recall'] = np.around(np.mean(results_rougeL['recall']), 3)
    results_rougeL['f'] = np.around(np.mean(results_rougeL['f']), 3)

    results_rougeLsum['precision'] = np.around(np.mean(results_rougeLsum['precision']), 3)
    results_rougeLsum['recall'] = np.around(np.mean(results_rougeLsum['recall']), 3)
    results_rougeLsum['f'] = np.around(np.mean(results_rougeLsum['f']), 3)

    return results_rouge1['f'], results_rouge2['f'] , results_rouge3['f'], results_rougeL['f'], results_rougeLsum['f']

if __name__ == '__main__':
    # Testing 
    r1_list = []
    r2_list = []
    r3_list = []
    rl_list = []
    rls_list = []
    for test_epoch in [6,7,8,9,10,11]:
        #Get data
        df = pd.read_excel(f'./flan_t5_xl_lora/test_{test_epoch}.xlsx') 
                                                        
        r1, r2, r3, rl, rlsum = predict_text(df)
        r1_list.append(r1)
        r2_list.append(r2)
        r3_list.append(r3)
        rl_list.append(rl)
        rls_list.append(rlsum)

    print(r1_list)
    print(r2_list)
    print(r3_list)
    print(rl_list)
    print(rls_list)
