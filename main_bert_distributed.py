import argparse
import os
import shutil
import torch
from loguru import logger
from torch import nn
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import (EncoderDecoderModel,
                          PreTrainedModel,
                          BertTokenizer,
                          BertGenerationEncoder,
                          BertGenerationDecoder)
from typing import Callable

from dataset import Oxford2019Dataset

model_type = os.environ.get('SM_HP_MODEL_TYPE', 'bert-base-uncased')
data_loc = os.environ.get('SM_HP_DATA_LOC', '../data')
epochs = int(os.environ.get('SM_HP_EPOCHS', 1))
batch = int(os.environ.get('SM_HP_BATCH', 24))
lr = float(os.environ.get('SM_HP_LR', 1e-5))
is_sagemaker_estimator = 'TRAINING_JOB_NAME' in os.environ  # This code is running on the remote SageMaker estimator machine

BOS_TOKEN_ID = 101
EOS_TOKEN_ID = 102


def make_data_loader(filename: str, file_loc: str = os.path.join(data_loc, 'Oxford-2019')) -> DataLoader:
    dataset = Oxford2019Dataset(data_loc=os.path.join(file_loc, filename))
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True)
    return data_loader


def run(model: nn.Module,
        data_loader: DataLoader,
        tokenizer: BertTokenizer,
        device,
        post_hook: Callable = None):
    loss = 0
    num_batches = len(data_loader)
    for i, (words, examples, defs, _) in enumerate(tqdm(data_loader, disable=True)):
        input_ids = tokenizer(examples,
                              add_special_tokens=False,
                              padding=True,
                              truncation=True,
                              return_tensors="pt").input_ids
        output_ids = tokenizer(defs,
                               padding=True,
                               truncation=True,
                               return_tensors="pt").input_ids

        input_ids = input_ids.to(device)
        output_ids = output_ids.to(device)

        outputs = model(input_ids=input_ids,
                        decoder_input_ids=output_ids,
                        labels=output_ids,
                        return_dict=True)
        batch_loss = outputs.loss.sum()
        loss += batch_loss.item()

        if post_hook is not None:
            post_hook(i, device.index, num_batches, batch_loss)
    return loss


def train(epochs: int,
          train_data_loader: DataLoader,
          valid_data_loader: DataLoader = None,
          model: nn.Module = None,
          rank = None):
    device = torch.device(f'cuda:{rank}')
    if model is None:
        encoder = BertGenerationEncoder.from_pretrained(model_type,
                                                        bos_token_id=BOS_TOKEN_ID,
                                                        eos_token_id=EOS_TOKEN_ID)  # add cross attention layers and use BERT’s cls token as BOS token and sep token as EOS token

        decoder = BertGenerationDecoder.from_pretrained(model_type,
                                                        add_cross_attention=True,
                                                        is_decoder=True,
                                                        bos_token_id=BOS_TOKEN_ID,
                                                        eos_token_id=EOS_TOKEN_ID)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    optimizer = AdamW(model.parameters(), lr=lr)

    tokenizer = BertTokenizer.from_pretrained(model_type)

    def update_weights(bi, di, num_batches, batch_loss):
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if bi % 100 == 0:
            logger.info(f'training: device={di}; batch={bi+1}/{num_batches}; batch_error={batch_loss.item()};')
            
    def valid_loss_progress_log(bi, di, num_batches, batch_loss):
        if bi % 100 == 0:
            logger.info(f'validation: device={di}; batch={bi+1}/{num_batches}; val_batch_error={batch_loss.item()};')
        

    for i in range(epochs):
        model.train()
        train_loss = run(model, train_data_loader, tokenizer, device, update_weights)

        if valid_data_loader is not None:
            with torch.no_grad():
                model.eval()
                val_loss = run(model, valid_data_loader, tokenizer, device)
        else:
            val_loss = 'N/A'

        logger.info(f'epoch={i}; device={rank}; train_error={train_loss};  valid_error={val_loss};')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    rank = args.local_rank

    # TODO: Copy all log files to the output
    log_name = f'log_{rank}.txt'
    logger.add(log_name)
    
    try:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend=Backend.NCCL, init_method='env://')

        train_set = make_data_loader('train.txt')
        test_set = make_data_loader('test.txt')
        valid_set = make_data_loader('valid.txt')

        model = train(epochs=epochs, train_data_loader=test_set, valid_data_loader=valid_set, rank=rank)

        out_loc = '/opt/ml/model'
        os.makedirs(out_loc, exist_ok=True)
        if rank == 0:
            model.save_pretrained(out_loc)
            shutil.copyfile(__file__, os.path.join(out_loc, __file__))
        shutil.copyfile(f'log.txt', os.path.join(out_loc, 'log.txt'))
    except:
        logger.exception("training failed")
        raise
