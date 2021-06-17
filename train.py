# Copyright 2021 hyeontae seo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import pandas as pd
import random
import argparse
import math
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainerforBase(object):
    def __init__(self,
            model_name='KETI-AIR/ke-t5-base',
            model_fname=None,
            target_vocab='vocab/target.model',
            batch_size=16,
            num_epochs=500):

        self.batch_size = batch_size
        self.num_epochs =num_epochs
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.model.to(device)

        self.optimizer = Adafactor(self.model.parameters(),lr=1e-3,
                              eps=(1e-30, 1e-3),
                              clip_threshold=1.0,
                              decay_rate=-0.8,
                              beta1=None,
                              weight_decay=0.0,
                              relative_step=False,
                              scale_parameter=False,
                              warmup_init=False)

    def data_load(self, train_path, val_path):
        train_df = pd.read_csv(train_path, index_col=[0])
        train_df = train_df.sample(frac = 1)

        val_df = pd.read_csv(val_path, index_col=[0])
        val_df = val_df.sample(frac = 1)

        return train_df, val_df

    def evaluator(self, val_df):
        self.model.eval()
        losses = 0
        num_of_batches = int(len(val_df)/self.batch_size)
        num_of_batches = 1 if num_of_batches == 0 else num_of_batches
        f1score = list()
        accuracy = list()

        for i in tqdm(range(num_of_batches)):
            input_list=list()
            target_list=list()
            data = val_df[i*self.batch_size:i*self.batch_size+self.batch_size]
            for indx, row in data.iterrows():
                _input = row['problem']
                _target = row['expression']
                input_list.append(_input)
                target_list.append(_target)

            inputs = self.tokenizer.batch_encode_plus(input_list, return_tensors='pt',
                    padding='max_length', truncation=True, max_length=100)
            targets = self.tokenizer.batch_encode_plus(target_list, return_tensors='pt',
                    padding='max_length', truncation=True, max_length=30)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs= self.model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"], labels=targets["input_ids"])
            loss = outputs.loss

            #########
            # F1Score & Acc
            #########

            losses += loss.item()
        
        val_loss = losses/num_of_batches
        #return val_loss
        return val_loss, np.mean(np.array(f1score)), np.mean(np.array(accuracy))

    def train(self, train_df, val_df):
        best_loss = math.inf
        best_epoch = 0
        num_of_batches=int(len(train_df)/self.batch_size)

        for epoch in range(1,self.num_epochs+1):
            # Training
            self.model.train()
            losses = 0

            for i in tqdm(range(num_of_batches)):
                input_list=list()
                target_list=list()
                data = train_df[i*self.batch_size:i*self.batch_size+self.batch_size]
                for indx, row in data.iterrows():
                    _input = row['problem']
                    _target = row['expression']
                    input_list.append(_input)
                    target_list.append(_target)

                inputs = self.tokenizer.batch_encode_plus(input_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=100)
                targets = self.tokenizer.batch_encode_plus(target_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=30)
                inputs = inputs.to(device)
                targets = targets.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"], labels=targets["input_ids"])
                loss = outputs.loss

                losses += loss.item()
                loss.backward()
                self.optimizer.step()

            train_loss = losses/num_of_batches

            # Evaluation
            val_loss, f1score, accuracy = self.evaluator(val_df)
            if best_loss > val_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), 'result/pytorch_model.bin')

            print("epoch [{}/{}], train loss:{:.4f}, val loss:{:.4f},\
                  f1 score:{:.4f}, accuracy:{:.4f}, best loss:{:.4f}[{}/{}]"
                  .format(epoch, self.num_epochs, train_loss, val_loss,
                  f1score, accuracy, best_loss, best_epoch, self.num_epochs))

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("-e", "--num_epochs", default=500,
                     type=int, help="number of epochs")
    par.add_argument("-b", "--batch_size", default=32,
                     type=int, help="number of batch size")
    par.add_argument("-t", "--train_path", default="data/train.csv",
                     type=str, help="Train data path")
    par.add_argument("-v", "--val_path", default="data/val.csv",
                     type=str, help="Validation data path")
    args = par.parse_args()

    base_trainer = TrainerforBase(num_epochs=args.num_epochs,
                                  batch_size=args.batch_size)
    train_df, val_df = base_trainer.data_load(args.train_path, args.val_path)
    base_trainer.train(train_df, val_df)
