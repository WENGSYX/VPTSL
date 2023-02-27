from transformers import DebertaV2Model,DebertaV2Config,DebertaV2Tokenizer
from collections import OrderedDict
import os
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--shape", default='large',type=str)
args = parser.parse_args()

MODEL_TYPE = {'large':'microsoft/deberta-v3-large',
              'base':'microsoft/deberta-v3-base',
              'small':'microsoft/deberta-v3-small',
              'xsmall':'microsoft/deberta-v3-xsmall'}

SHAPE = args.shape

if __name__ == '__main__':
    model = DebertaV2Model.from_pretrained(MODEL_TYPE[SHAPE])
    model_config = DebertaV2Config.from_pretrained(MODEL_TYPE[SHAPE])
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_TYPE[SHAPE])
    VPTSL = model.state_dict()
    VPTSL2 = model.state_dict()
    for m in VPTSL2:
        VPTSL[m.replace('deberta','deberta2')] = VPTSL2[m]
    try:
        os.mkdir('VPTSL_{}'.format(SHAPE))
    except:
        pass
    torch.save(VPTSL,'VPTSL_{}/pytorch_model.bin'.format(SHAPE))
    model_config.save_pretrained('VPTSL_{}'.format(SHAPE))
    tokenizer.save_pretrained('VPTSL_{}'.format(SHAPE))