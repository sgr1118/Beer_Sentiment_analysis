# -*- coding: utf-8 -*-
"""simpleT5_clone_coding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14Zg5JZwqEAUmWzfhHcgXYCVkQmHNIyVs

# 목표

### simpleT5 클론 코딩을 통하여 구조를 이해한다.
- 각 코드를 이해하기 위하여 주석으로 설명하기로한다.

### 사용하는 프레임워크
- Pytorch Lightning == 1.5.10
- numpy
- pandas
- sentencepiece
- torch>=1.7.0,!=1.8.0
- transformers==4.16.2
"""

# 라이브러리 불러오기

import torch
import numpy as np
import pandas as pd
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

torch.cuda.empty_cache()
pl.seed_everything(42)

# 데이터셋 모듈

class PytorchDataModule(Dataset):
    """ Pytorch Dataset class """

    def __init__(
        self,
        data : pd.DataFrame,
        tokenizer : PreTrainedTokenizer,
        source_max_token_len: int = 512, # 최대 토큰 길이
        target_max_token_len: int = 512, # 최대 토큰 길이
    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ 데이터 길이 반환 """
        return len(self.data)

    def __getitem__(self, index: int):
        """ 이 코드는 T5/MT5 모델에 입력으로 사용될 수 있는 텐서 딕셔너리를 반환하는 함수 """
        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_legnth = self.source_max_token_len,
            padding = "max_legnth",
            truncation = True, # max_legnth 보다 긴 문장은 자른다.
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = "pt",
        )

        target_text_encoding  = self.tokenizer(
            data_row["target_text"],
            max_legnth = self.target_max_token_len,
            padding = "max_legnth",
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = "pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[labels == 0] = -100 # 레이블 텐서에서 모든 패딩 토큰을 -100으로 대체하여 
        # 모델이 패딩 토큰을 예측하지 않도록 한다.

        return dict(
            source_text_input_ids = source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask = source_text_encoding["attention_mask"].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = target_text_encoding["attention_mask"].flatten()
        ) # dict에 각 요소를 1차원으로 평탄화하여 저장
          # model 입력은 1차원 텐서로 들어가야하기 때문에 flatten()을 사용해준다.

class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512, # 최대 토큰 길이
        target_max_token_len: int = 512, # 최대 토큰 길이
        num_workers: int = 2,
    ):

        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage = None):
        self.train_dataset = PytorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

        self.test_dataset = PytorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer,
        model,
        outputdir: str = "outputs", # outputs dir 이름
        save_only_last_epoch: bool = False
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch

    def forward(self, input_ids, attention_mask,decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids, # 토큰 ID 시퀀스
            attention_mask = attention_mask, # 각 토큰의 유효성 여부를 나타내는 마스킹 시퀀스
            labels = labels,
            decoder_attention_mask = decoder_attention_mask,
        )
        return output.loss, output.logits
        #logit은 모델이 예측한 값으로, 확률이 아닌 숫자값으로 출력됩니다. 이 값을 확률값으로 변환하여 예측값을 계산
        
    def training_step(self, batch, batch_size): # LightningDataModule에서 생성한 데이터 로더(dataloader)에서 반환된 batch
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch['source_text_attention_mask']
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels,
        )

        self.log(
            "train_loss", loss, prog_ar = True, logger = True, op_epoch = True, on_step = True
        )
        return loss # 손실 저장

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "val_loss", loss, prog_bar = True, logger = True, on_epoch = True, on_step = True
        )
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)
        # AdamW : Adam optimizer의 변형으로, 가중치 감쇠(weight decay)를 적용하여 모델 파라미터의 크기를 줄이는 효과

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.outputdir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"

        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )

class SimpleT5:
    """ Custom SimpleT5 class """

    def __init__(self) -> None:
        """ initiates SimpleT5 class """
        pass

    def from_pretrained(self, model_type = "t5", model_name = "t5-base") -> None:
        """
        loads T5/MT5 Model model for training/finetuning
        Args:
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        """

        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        outputdir: str = "outputs",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32, # 모델 훈련에 사용할 데이터 타입의 정밀도
        logger="default",
        dataloader_num_workers: int = 2,
        save_only_last_epoch: bool = False,
    ):
        """
        trains T5/MT5 model on custom dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
            dataloader_num_workers (int, optional): number of workers in train/test/val dataloader
            save_only_last_epoch (bool, optional): If True, saves only the last epoch else models are saved at every epoch
        """

        self.data_module = LightningDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )
        
        self.T5Model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            outputdir=outputdir,
            save_only_last_epoch=save_only_last_epoch,
        )

        # add callbacks
        callbacks = [TQDMProgressBar(refresh_rate = 5)]

        if early_stopping_patience_epochs > 0: