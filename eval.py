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


import pandas as pd
import random
import argparse
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

class PredictorforBase(object):
    def __init__(self,
            model_name='KETI-AIR/ke-t5-base',
            log_path='result/'):

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(log_path, return_dict=True)

    def data_load(self, test_path):
        test_df = pd.read_csv(test_path, index_col=[0])
        test_df = test_df.sample(frac = 1)

        return test_df

    def predict(self, test_df):
        self.model.eval()
        point = 0
        large = list()
        _none = list()
        num_list = list(range(1, 1001))
        for num in tqdm(num_list):
            data = test_df.loc[[num]].values
            inputs = self.tokenizer.encode(data[0][0], return_tensors="pt")
            outputs = self.model.generate(inputs)
            result = self.tokenizer.decode(outputs[0])
            result = result.replace("</s>", "").replace("<pad>", "").replace("<unk>", "")

            if len(data[0][1].split()) > 30:
                large.append(num)
                continue

            try:
                answer = round(float(data[0][2]), 2)
            except:
                _none.append(num)
                continue

            try:
                pred_answer = round(eval(result), 2)
            except:
                continue

            if pred_answer == answer:
                point += 1

        print("Large: {}".format(large))
        print("None: {}".format(_none))

        return point/(len(num_list)-len(large)-len(_none))

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("-t", "--test_path", default="data/val.csv",
                     type=str, help="Train data path")
    args = par.parse_args()

    base_predictor = PredictorforBase()
    test_df = base_predictor.data_load(args.test_path)
    accuracy = base_predictor.predict(test_df)

    print("Accuracy: {}%".format(accuracy*100))
