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

import sentencepiece as spm

corpus = 'target.txt'
prefix = 'target'
#vocab_size = 28
#model_type = 'word'
vocab_size = 37
model_type = 'bpe'
character_coverage = 1.0

input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s'

cmd = input_argument%(corpus,prefix,vocab_size,model_type,character_coverage)

spm.SentencePieceTrainer.train(cmd)
