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

import json
import pandas as pd

import argparse

par = argparse.ArgumentParser()
par.add_argument("-a", "--answer_path", default="data/answer_mathQA.json",
                 type=str, help="answer sheet path")
par.add_argument("-p", "--problem_path", default="data/question_mathQA.json",
                 type=str, help="problem sheet path")
args = par.parse_args()

with open(args.answer_path, 'r') as f:
    answersheet = json.load(f)

with open(args.problem_path, 'r') as f:
    problemsheet = json.load(f)

if len(answersheet) != len(problemsheet):
    raise ValueError('The correct answer and problem pairs do not match.')

train = {"problem":[], "expression":[], "solution":[]}
val = {"problem":[], "expression":[], "solution":[]}
#test = {"problem":[], "expression":[], "solution":[]}

split_num = len(answersheet)*0.9

for i in range(len(answersheet)):
    if i < split_num:
        train['problem'].append(problemsheet[str(i+1)]['question'])
        train['expression'].append(answersheet[str(i+1)]['equation'])
        train['solution'].append(answersheet[str(i+1)]['answer'])
    #elif i % 2 == 0:
    else:
        val['problem'].append(problemsheet[str(i+1)]['question'])
        val['expression'].append(answersheet[str(i+1)]['equation'])
        val['solution'].append(answersheet[str(i+1)]['answer'])
    #else:
    #    test['problem'].append(problemsheet[str(i+1)]['question'])
    #    test['expression'].append(answersheet[str(i+1)]['equation'])
    #    test['solution'].append(answersheet[str(i+1)]['answer'])

train_df=pd.DataFrame(train)
train_df.to_csv('data/train.csv')
val_df=pd.DataFrame(val)
val_df.to_csv('data/val.csv')
#test_df=pd.DataFrame(test)
#test_df.to_csv('data/test.csv')
