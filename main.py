import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
from accelerate import Accelerator
import glob


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data(name):
    with open('data/text/{}'.format(name), encoding='utf-8') as f:
        data = json.load(f)
    return data

def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature
def load_video_features(root, max_position_length):
    video_features = dict()
    filenames = glob.glob(os.path.join(root, "*.npy"))
    for filename in tqdm(filenames, total=len(filenames), desc="loading video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_position_length is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_position_length)
            video_features[video_id] = new_feature
    return video_features


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]

        return data

def pad_video_seq(sequences, max_length=1024):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length


def collate_fn(data):
    input_ids, attention_mask, token_type_ids,start_labels,end_labels,video_features,audios,h_labels = [], [], [],[],[],[],[],[]
    for x in data:
        input_id,attention,token_type = [],[],[]
        sub = x['video_sub_title']
        min_start = 10000
        min_end = 10000
        start_text = x['video_sub_title'][0]['text']
        end_text = x['video_sub_title'][-1]['text']
        for s in range(len(sub)):
            if abs(sub[s]['start']-x['answer_start_second']) < min_start:
                start_text = sub[s]['text']
                start_id = s
                min_start = abs(sub[s]['start']-x['answer_start_second'])
            if abs(sub[s]['start']+sub[s]['duration']-x['answer_end_second']) <= min_end:
                end_text = sub[s]['text']
                end_id = s
                min_end = abs(sub[s]['start']+sub[s]['duration']-x['answer_end_second'])

        text = x['question']
        text = tokenizer(text)
        input_id.extend(text.input_ids)
        token_type.extend([0]*len(text.input_ids))

        for s in range(len(sub)):
            if s == start_id:
                start_label = len(input_id)+1
            input_id.extend(tokenizer(sub[s]['text']).input_ids[1:])
            if s == end_id:
                end_label = len(input_id)-1
        vi = visual[x['video_id']]
        video_features.append(vi)

        h_label = np.zeros(shape=[1024], dtype=np.int32)
        st, et = x['answer_start_second'],x['answer_end_second']
        cur_max_len = vi.shape[0]
        extend_len = round(args.highlight_hyperparameter * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_label[st_:(et_ + 1)] = 1
        else:
            h_label[st:(et + 1)] = 1
        h_labels.append(h_label)

        token_type.extend([1] * (len(input_id)-len(token_type)))
        attention = [1] * len(input_id)
        input_ids.append(input_id)
        attention_mask.append(attention)
        token_type_ids.append(token_type)
        start_labels.append(start_label)
        end_labels.append(end_label)

    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats_mask = torch.stack([torch.cat([torch.ones([vi.shape[0]]),torch.zeros([1024-vi.shape[0]])])])
    h_labels=torch.tensor(h_labels)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    vfeats = torch.tensor(vfeats)
    start_labels = torch.tensor(start_labels)
    end_labels = torch.tensor(end_labels)
    return input_ids, attention_mask, token_type_ids,vfeats,start_labels,end_labels,h_labels,vfeats_mask

def get_args(args):
    l = []
    for k in list(vars(args).keys()):
        l.append(('%s: %s' % (k, vars(args)[k])))
    return l
class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(model,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs_start = AverageMeter()
    accs_end = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_type_ids,vfeats,start_labels,end_labels,h_labels,vfeats_mask) in enumerate(tk):
        with autocast():
            output = model(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids, video_features=vfeats,start_positions=start_labels,end_positions=end_labels,vfeats_mask=vfeats_mask)
            h_labels = h_labels.type(torch.float32)
            weights = torch.where(h_labels == 0.0, h_labels + 1.0, 2.0 * h_labels)
        loss_per_location = nn.BCELoss(reduction='none')(output.h_score, h_labels)
        loss_per_location = loss_per_location * weights
        loss2 = torch.sum(loss_per_location * vfeats_mask) / (torch.sum(vfeats_mask) + 1e-12)
        loss = output.loss + args.loss_hyperparameter*loss2
        scaler.scale(loss).backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        losses.update(loss.item(), input_ids.size(0))
        acc_start = (output.start_logits.argmax(1) == start_labels).sum().item() / end_labels.size(0)
        acc_end = (output.end_logits.argmax(1) == end_labels).sum().item() / end_labels.size(0)
        accs_start.update(acc_start)
        accs_end.update(acc_end)
        tk.set_postfix(loss=losses.avg,start=accs_start.avg,end=accs_end.avg)
        if step == 0:
            log(['Start Train:','Now epoch:{}'.format(epoch),'Now Loss：{}'.format(str(loss.item())),'all of the step:{}'.format(len(tk))],path)

    log(['Now Loss：{}'.format(str(loss.item())),'Avg Loss：{}'.format(losses.avg),'Avg Start Acc:{}'.format(accs_start.avg),'Avg End Acc:{}'.format(accs_end.avg),'End this round of training'],path)
    return losses.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs_start = AverageMeter()
    accs_end = AverageMeter()
    f1s  = AverageMeter()
    optimizer.zero_grad()
    pred_start = []
    pred_end = []
    token_id = []
    tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_type_ids,vfeats,start_labels,end_labels,h_labels,vfeats_mask) in enumerate(tk):

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids, video_features=vfeats,start_positions=start_labels,end_positions=end_labels,vfeats_mask=vfeats_mask)
            loss = output.loss

        losses.update(loss.item(), input_ids.size(0))
        pred_start.extend(output.start_logits)
        pred_end.extend(output.end_logits)
        token_id.extend(input_ids)
        acc_start = (output.start_logits.argmax(1) == start_labels).sum().item() / end_labels.size(0)
        acc_end = (output.end_logits.argmax(1) == end_labels).sum().item() / end_labels.size(0)
        accs_start.update(acc_start)
        accs_end.update(acc_end)
        tk.set_postfix(loss=losses.avg,start=accs_start.avg,end=accs_end.avg)


    return losses.avg,pred_start,pred_end,token_id

def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    try:
        os.mkdir('paperlog/' + log_name)
    except:
        log_name = log_name+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.mkdir('paperlog/' + log_name)

    with open('paperlog/' + log_name + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)
    with open('paperlog/' + log_name + '.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__),'r',encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    path = 'paperlog/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path

def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default='large', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--maxlen", default=1800, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--lr", default=8e-6, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--device", default=2, type=float)
    parser.add_argument("--highlight_hyperparameter", default=0.25, type=float)
    parser.add_argument("--loss_hyperparameter", default=0.1, type=float)
    args = parser.parse_args()
    CFG = {
        'seed': args.seed,
        'model': 'microsoft/deberta-v3-large',
        'max_len': args.maxlen,
        'epochs': args.epochs,
        'train_bs': 1,
        'valid_bs': 1,
        'lr': args.lr,
        'num_workers': args.num_workers,
        'accum_iter': args.batchsize,
        'weight_decay': args.weight_decay,
        'device': args.device,
    }

    accelerator = Accelerator()
    seed_everything(CFG['seed'])
    torch.cuda.set_device(CFG['device'])
    device = accelerator.device
    visual = load_video_features(os.path.join('data', 'features', 'I3D'), 1024)

    train = get_data(r'train.json')
    valid = get_data(r'val.json')
    test = get_data(r'test.json')

    tokenizer = AutoTokenizer.from_pretrained(CFG['model'])
    tokenizer.add_tokens('[Music]')

    train_set = MyDataset(train)
    valid_set = MyDataset(valid)
    test_set = MyDataset(test)

    train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
    valid_loader = DataLoader(valid_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                              num_workers=CFG['num_workers'])
    test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                              num_workers=CFG['num_workers'])
    best_acc = 0

    from VPTSL_model import VPTSLModel
    model = VPTSLModel.from_pretrained(CFG['model'])


    model = model.to(device)
    optimizer = AdamW(model.parameters(),lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                                CFG['epochs'] * len(train_loader) // CFG['accum_iter'])

    train_loader,val_loader,test_loader = accelerator.prepare(train_loader,valid_loader,test_loader)
    scaler = GradScaler()
    log_name = 'VPTSL_1'
    path = log_start(log_name)
    log(get_args(args),path)


    for epoch in range(CFG['epochs']):
        train_loss = train_model(model,train_loader)
        val_loss, pred_start,pred_end,token_id = test_model(model, val_loader)
        ious = []
        for i in range(len(token_id)):
            start = pred_start[i].argmax().item()
            end = pred_end[i].argmax().item()
            token = token_id[i].tolist()
            tokens = []
            ts = []


            for t_num in range(len(token)):
                ts.append(token[t_num])
                if t_num == start:
                    start_time = valid[i]['video_sub_title'][len(tokens)-1]['start']
                if t_num == end:
                    end_time = valid[i]['video_sub_title'][len(tokens)-1]['start']+valid[i]['video_sub_title'][len(tokens)-1]['duration']
                if token[t_num] == 2:
                    tokens.append(ts)
                    ts = []
            if start_time>=end_time:
                end_time = valid[i]['video_sub_title'][-1]['start']+valid[i]['video_sub_title'][-1]['duration']

            iou = calculate_iou(i0=[start_time, end_time], i1=[valid[i]["answer_start_second"], valid[i]["answer_end_second"]])
            ious.append(iou)
        r1i3_1 = calculate_iou_accuracy(ious, threshold=0.3)
        r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        mi1 = np.mean(ious) * 100.0
        # write the scores
        score_str = ["Epoch {}".format(epoch)]
        score_str += ["Rank@1, IoU=0.3: {:.2f}".format(r1i3_1)]
        score_str += ["Rank@1, IoU=0.5: {:.2f}".format(r1i5)]
        score_str += ["Rank@1, IoU=0.7: {:.2f}".format(r1i7)]
        score_str += ["mean IoU: {:.2f}".format(mi1)]
        log(score_str,path)
        val_loss, pred_start,pred_end,token_id = test_model(model, test_loader)
        ious = []
        for i in range(len(token_id)):
            start = pred_start[i].argmax().item()
            end = pred_end[i].argmax().item()
            token = token_id[i].tolist()
            tokens = []
            ts = []

            for t_num in range(len(token)):
                ts.append(token[t_num])
                if t_num == start:
                    start_time = test[i]['video_sub_title'][len(tokens)-1]['start']
                if t_num == end:
                    end_time = test[i]['video_sub_title'][len(tokens)-1]['start']+test[i]['video_sub_title'][len(tokens)-1]['duration']
                if token[t_num] == 2:
                    tokens.append(ts)
                    ts = []
            if start_time>=end_time:
                end_time = test[i]['video_sub_title'][-1]['start']+test[i]['video_sub_title'][-1]['duration']
            iou = calculate_iou(i0=[start_time, end_time], i1=[test[i]["answer_start_second"], test[i]["answer_end_second"]])
            ious.append(iou)
        r1i3_2 = calculate_iou_accuracy(ious, threshold=0.3)
        r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        mi2 = np.mean(ious) * 100.0
        # write the scores
        score_str = ["Epoch {}".format(epoch)]
        score_str += ["Rank@1, IoU=0.3: {:.2f}".format(r1i3_2)]
        score_str += ["Rank@1, IoU=0.5: {:.2f}".format(r1i5)]
        score_str += ["Rank@1, IoU=0.7: {:.2f}".format(r1i7)]
        score_str += ["mean IoU: {:.2f}".format(mi2)]

        model_name = path+'/{}_{}_{}_model'.format(epoch,round(mi1,2),round(mi2,2))
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)
        log(score_str+['SAVE MODEL:{}'.format(model_name)],path)

