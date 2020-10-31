import os
import torch
from torch import nn
import shutil
from tqdm import tqdm
from dataset import Oxford2019Dataset
from torch.utils.data import DataLoader
from typing import Callable
from transformers import AdamW
from torch import nn
from loguru import logger
from transformers import (EncoderDecoderModel,
                          PreTrainedModel,
                          BertTokenizer,
                          BertGenerationEncoder,
                          BertGenerationDecoder)

logger.add("log.txt")

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

model_type = os.environ.get('SM_HP_MODEL_TYPE', 'bert-base-uncased')
data_loc = os.environ.get('SM_HP_DATA_LOC', '../data')
epochs = int(os.environ.get('SM_HP_EPOCHS', 2))
batch = int(os.environ.get('SM_HP_BATCH', 32)) * device_count
lr = float(os.environ.get('SM_HP_LR', 1e-5))
train_remotely = bool(int(os.environ.get('SM_HP_TRAIN_REMOTELY', 1)))
is_sagemaker_estimator = 'TRAINING_JOB_NAME' in os.environ  # This code is running on the remote SageMaker estimator machine

BOS_TOKEN_ID = 101
EOS_TOKEN_ID = 102


def make_data_loader(filename: str, file_loc: str = os.path.join(data_loc, 'Oxford-2019')) -> DataLoader:
    dataset = Oxford2019Dataset(data_loc=os.path.join(file_loc, filename))
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True)
    return data_loader


train_set = make_data_loader('train.txt')
test_set = make_data_loader('test.txt')
valid_set = make_data_loader('valid.txt')


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


def train(epochs: int, train_data_loader: DataLoader, valid_data_loader: DataLoader = None, model: nn.Module = None):
    if model is None:
        encoder = BertGenerationEncoder.from_pretrained(model_type,
                                                        bos_token_id=BOS_TOKEN_ID,
                                                        eos_token_id=EOS_TOKEN_ID)  # add cross attention layers and use BERT’s cls token as BOS token and sep token as EOS token

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
            logger.info(f'batch_error={batch_loss.item()};')

    for i in range(epochs):
        model.train()
        train_loss = run(model, train_data_loader, tokenizer, update_weights)

        if valid_data_loader is not None:
            with torch.no_grad():
                model.eval()
                val_loss = run(model, valid_data_loader, tokenizer)
        else:
            val_loss = 'N/A'

        logger.info(f'train_error={train_loss};  valid_error={val_loss};')
    return model


def save_model(model: PreTrainedModel):
    out_loc = '/opt/ml/model'
    os.makedirs(out_loc, exist_ok=True)

    model.save_pretrained(out_loc)

    shutil.copyfile(__file__, os.path.join(out_loc, __file__))
    shutil.copyfile('log.txt', os.path.join(out_loc, 'log.txt'))


model = train(epochs=epochs, train_data_loader=test_set, valid_data_loader=valid_set)
save_model(model)
