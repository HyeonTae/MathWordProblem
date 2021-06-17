import json
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_path", default="questionsheet_MATHQA.json",
                     type=str, help="data path")
args = par.parse_args()

with open(args.data_path, 'r', encoding="utf-8") as inputfile:
    data = json.load(inputfile)

with open(args.data_path, 'w', encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=4, ensure_ascii=False)

# python3 indentation.py -d questionsheet_MATHQA.json
