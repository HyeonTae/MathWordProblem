import json
import argparse

par = argparse.ArgumentParser()
par.add_argument("-t", "--data_type", default="q",
        type=str, help="origin data path",
        choices=['q', 'a', 'question', 'answer'])
args = par.parse_args()

if 'q' in args.data_type:
    ori_path = 'question_mathQA_ori.json'
    refined_path = 'question_mathQA_refined.json'
    save_path = 'question_mathQA.json'
else:
    ori_path = 'answer_mathQA_ori.json'
    refined_path = 'answer_mathQA_refined.json'
    save_path = 'answer_mathQA.json'

with open(ori_path, 'r', encoding="utf-8") as inputfile:
    ori_data = json.load(inputfile)

with open(refined_path, 'r', encoding="utf-8") as inputfile:
    refined_data = json.load(inputfile)

count = 0
for k in refined_data.keys():
    ori_data[k] = refined_data[k]
    count += 1

with open(save_path, 'w', encoding="utf-8") as outfile:
    json.dump(ori_data, outfile, indent=4, ensure_ascii=False)

print("{} data refine".format(count))
# python3 data_merge.py -t q
