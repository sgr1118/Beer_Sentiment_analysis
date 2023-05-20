import re
import torch
from glob import glob

import random
import pandas as pd
from tqdm import tqdm
from simplet5 import SimpleT5
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_review_dataset(random_seed = 1, file_path='beer_preprocess.csv'):
    df = pd.read_csv(file_path)

    # train, test 분류
    X_train, X_valid, y_train, y_valid = \
        train_test_split(df['Review'].tolist(), df['label'].tolist(),
                         shuffle=True, test_size=0.2, random_state=random_seed, stratify=df['label'])

        
    X_val, X_test, y_val, y_test = \
        train_test_split(X_valid, y_valid,
                         shuffle=True, test_size=0.5, random_state=random_seed, stratify=y_valid)
        
    # transform to pandas dataframe
    train_data = pd.DataFrame({'source_text': X_train, 'target_text': y_train})    
    val_data = pd.DataFrame({'source_text': X_val, 'target_text': y_val})  
    test_data = pd.DataFrame({'source_text': X_test, 'target_text': y_test})  

    return train_data, val_data, test_data


from glob import glob

# create data
train_df, val_df, test_df = load_review_dataset(1)    
# load model
model = SimpleT5()
model_type = 't5'

model.from_pretrained(model_type=model_type, model_name="t5-base")

# train model
model.train(train_df=train_df,
            eval_df=val_df, 
            source_max_token_len=300, 
            target_max_token_len=200, 
            batch_size=32, 
            max_epochs=2, 
            outputdir = "outputs",
            use_gpu=True
            )
# fetch the path to last model
last_epoch_model = None 
for file in glob("./outputs/*"):
    if 'epoch-1' in file:
        last_epoch_model = file

model = SimpleT5()
model_type = 't5'
print('loading model')
# load the last model
model.load_model(model_type, last_epoch_model, use_gpu=True)
# test and save
# for each test data perform prediction
print('making prediction')
predictions = []
for index, row in tqdm(test_df.iterrows()):
    prediction = model.predict(row['source_text'])[0]
    predictions.append(prediction)
df = test_df.copy()
df['predicted'] = predictions
df['original'] = df['target_text']

df.to_csv(f"result_run_Binary_class.csv", index=False)