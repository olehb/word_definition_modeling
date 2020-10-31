#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


# !pip install transformers


# In[3]:


from transformers import (EncoderDecoderModel,
                          PreTrainedModel,
                          BertTokenizer,
                          BertGenerationEncoder,
                          BertGenerationDecoder)


# In[4]:


import os
import torch
from torch import nn
from tqdm import tqdm


# ### Determining device to run on

# In[5]:


if torch.cuda.is_available():
    main_device = torch.device("cuda:0")
    device_count = torch.cuda.device_count()
    if device_count > 1:
        src_device = torch.device("cuda:1")
    else:
        src_device = main_device
        
    print("Running on the GPU", main_device)
else:
    main_device = torch.device("cpu")
    src_device = torch.device("cpu")
    device_count = 1
    print("Running on the CPU", main_device)


# ### Reading hyperparameters from SageMaker env variables

# In[6]:


model_type = os.environ.get('SM_HP_MODEL_TYPE', 'bert-base-uncased')
data_loc = os.environ.get('SM_HP_DATA_LOC', '../data')
epochs = int(os.environ.get('SM_HP_EPOCHS', 2))
batch = int(os.environ.get('SM_HP_BATCH', 32)) * device_count
lr = float(os.environ.get('SM_HP_LR', 1e-5))
train_remotely = bool(int(os.environ.get('SM_HP_TRAIN_REMOTELY', 1)))
is_sagemaker_estimator = 'TRAINING_JOB_NAME' in os.environ  # This code is running on the remote SageMaker estimator machine


# In[7]:


BOS_TOKEN_ID = 101
EOS_TOKEN_ID = 102


# ### Initializing data loaders for Oxford2019 dataset

# In[8]:


from dataset import Oxford2019Dataset
from torch.utils.data import DataLoader

def make_data_loader(filename: str, file_loc: str = os.path.join(data_loc, 'Oxford-2019')) -> DataLoader:
    dataset = Oxford2019Dataset(data_loc=os.path.join(file_loc, filename))
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True)
    return data_loader

train_set = make_data_loader('train.txt')
test_set = make_data_loader('test.txt')
valid_set = make_data_loader('valid.txt')


# ### Function to run through one epoch
# This function is used in training, validation, and testing phases.

# In[9]:


from typing import Callable

def run(model: nn.Module,
        data_loader: DataLoader,
        tokenizer: BertTokenizer,
        post_hook: Callable = lambda i, b: ''):

    loss = 0
    for i, (words, examples, defs, _) in enumerate(tqdm(data_loader, disable=is_sagemaker_estimator)):
        input_ids = tokenizer(examples,
                              add_special_tokens=False,
                              padding=True,
                              truncation=True,
                              return_tensors="pt").input_ids
        output_ids = tokenizer(defs,
                               padding=True,
                               truncation=True,
                               return_tensors="pt").input_ids
        
        input_ids = input_ids.to(src_device)
        output_ids = output_ids.to(src_device)
        
        outputs = model(input_ids=input_ids,
                        decoder_input_ids=output_ids,
                        labels=output_ids,
                        return_dict=True)
        batch_loss = outputs.loss.sum()
        loss += batch_loss.item()

        post_hook(i, batch_loss)
    return loss


# ### Training loop function

# In[10]:


from transformers import AdamW
from torch import nn


def train(epochs: int, train_data_loader: DataLoader, valid_data_loader: DataLoader = None, model: nn.Module = None):
    if model is None:
        encoder = BertGenerationEncoder.from_pretrained(model_type,
                                                        bos_token_id=BOS_TOKEN_ID,
                                                        eos_token_id=EOS_TOKEN_ID) # add cross attention layers and use BERT’s cls token as BOS token and sep token as EOS token

        decoder = BertGenerationDecoder.from_pretrained(model_type,
                                                        add_cross_attention=True,
                                                        is_decoder=True,
                                                        bos_token_id=BOS_TOKEN_ID,
                                                        eos_token_id=EOS_TOKEN_ID)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(src_device)
        model = nn.DataParallel(model, 
                                device_ids=list(range(1, torch.cuda.device_count())), 
                                output_device=0)


    optimizer = AdamW(model.parameters(), lr=lr)

    tokenizer = BertTokenizer.from_pretrained(model_type)
    

    def update_weights(i, batch_loss):
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 10 == 0:
            print(f'batch_error={batch_loss.item()};')

    for i in range(epochs):
        model.train()
        train_loss = run(model, train_data_loader, tokenizer, update_weights)

        if valid_data_loader is not None:
            with torch.no_grad():
                model.eval()
                val_loss = run(model, valid_data_loader, tokenizer)
        else:
            val_loss = 'N/A'
        
        msg = f'train_error={train_loss};  valid_error={val_loss};'
        print(msg)
        get_ipython().system("echo '{msg}' >> log.txt")
    return model


# ### Quick sanity check for the training loop

# In[ ]:


if not is_sagemaker_estimator:
#     encoder = BertGenerationEncoder.from_pretrained(model_type,
#                                                     bos_token_id=BOS_TOKEN_ID,
#                                                     eos_token_id=EOS_TOKEN_ID) # add cross attention layers and use BERT’s cls token as BOS token and sep token as EOS token

#     decoder = BertGenerationDecoder.from_pretrained(model_type,
#                                                     add_cross_attention=True,
#                                                     is_decoder=True,
#                                                     bos_token_id=BOS_TOKEN_ID,
#                                                     eos_token_id=EOS_TOKEN_ID)
#     model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)    
    
    torch.cuda.empty_cache()
    train_file = os.path.join(data_loc, 'Oxford-2019', 'train.txt')
    tiny_size = batch * 5
    tiny_file = os.path.join(data_loc, 'Oxford-2019', 'tiny.txt')
    get_ipython().system('head -n {tiny_size} {train_file} > {tiny_file}')
    tiny_set = make_data_loader('tiny.txt')
    model = train(epochs=1, train_data_loader=tiny_set, valid_data_loader=tiny_set)


# In[ ]:


if not is_sagemaker_estimator:
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = torch.tensor(tokenizer.encode("Basketball Basketball's early adherents were dispatched to YMCAs throughout the United States, and it quickly spread through the United States and Canada", add_special_tokens=True)).unsqueeze(0).to(src_device)
    edm = model.module
    generated = edm.generate(input_ids, decoder_start_token_id=edm.config.decoder.pad_token_id)
    print(tokenizer.decode(generated.squeeze(), skip_special_tokens=True))


# ### Function for saving the model

# In[11]:


def save_model(model: PreTrainedModel):
    out_loc = '/opt/ml/model' if is_sagemaker_estimator else '.'
    get_ipython().system('mkdir -p {out_loc}')

    model.save_pretrained(out_loc)

    get_ipython().system('cp main_bert.py {out_loc}')
    get_ipython().system('cp main_bert.ipynb {out_loc}')
    get_ipython().system('cp log.txt {out_loc}')


# ### Training
# Training can be done either on the same machine where notebook is running or remotely on SageMaker estimator

# In[12]:


import sagemaker
from sagemaker.pytorch import PyTorch

if is_sagemaker_estimator:
    model = train(epochs=epochs, train_data_loader=valid_set, valid_data_loader=test_set)
    save_model(model)
elif train_remotely:
    role = sagemaker.get_execution_role()
    output_path = f's3://chegg-ds-data/oboiko/wdm-output'

    pytorch_estimator = PyTorch(entry_point='train.sh',
                                base_job_name='wdm-1',
                                role=role,
                                train_instance_count=1,
                                train_instance_type='ml.p2.8xlarge',  # GPU instance
                                train_volume_size=50,
                                train_max_run=86400,  # 24 hours
                                hyperparameters={
                                  'model_type': 'bert-base-uncased',
                                  'data_loc': '/opt/data',
                                  'batch': 32,
                                  'epochs': 1,
                                  'lr': 1e-5,
                                  'train_remotely': 0,
                                  'notebook_name': 'main_bert_distributed'
                                },
                                framework_version='1.6.0',
                                py_version='py3',
                                source_dir='.',  # This entire folder will be transferred to training instance
                                debugger_hook_config=False,
                                output_path=output_path,  # Model files will be uploaded here
                                image_name='954558792927.dkr.ecr.us-west-2.amazonaws.com/sagemaker/wdm:latest',
                                metric_definitions=[
                                    {'Name': 'train:error', 'Regex': 'train_error=(.*?);'},
                                    {'Name': 'validation:error', 'Regex': 'valid_error=(.*?);'},
                                    {'Name': 'batch:error', 'Regex': 'batch_error=(.*?);'}
                                ]
                     )

    pytorch_estimator.fit('s3://chegg-ds-data/oboiko/wdm/dummy.txt', wait=False)


# TODO: For loss function... instead of doing log_softmax, do MSE with actual GloVe vector and minimize this loss function.
# Then for BLEU evaluation, you'll need a function to find the closest vector to the one produced by the model.
# 
# Interesting to compare these results to log_softmax
