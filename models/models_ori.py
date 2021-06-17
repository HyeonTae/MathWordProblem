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

from transformers import T5ForConditionalGeneration

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

class T5GenerationBase(T5ForConditionalGeneration):
    def __init__(self, config='models/config.json',
            model_name=None,
            model_fname=None,
            return_dict=True):
        super(T5GenerationBase, self).__init__(config)

        if model_name:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_fname:
            print(model_fname)
            model = T5ForConditionalGeneration.from_pretrained(model_fname, return_dict=return_dict)
        else:
            raise ValueError('You have to specify either model path or name')

        self.model = model

    def forward(self, inputs, targets=None):
        loss = None
        if targets is not None:
            outputs = self.model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"], labels=targets["input_ids"])
        else:
            outputs = self.model.generate(input_ids)
        loss = outputs.loss

        return outputs, loss
