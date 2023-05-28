# dataset = 'IEMOCAP'  # 50
dataset = 'MELD'   # 10
# dataset = 'DailyDialog'   # 8
# dataset = 'EmoryNLP'   # 14
max_speaker = {
    'IEMOCAP' : 2,
    'MELD': 9,
    'DailyDialog':2,
    'DailyDialog2':2,
    'EmoryNLP':9
}  # 8 for MELD
all_speaker = {
    'IEMOCAP' : 303,
    'MELD': 305,
    'DailyDialog':2,
    'DailyDialog2':2,
    'EmoryNLP':286
}  # 8 for MELD
speaker_vocab = all_speaker[dataset]
max_speaker_num = max_speaker[dataset]
batch_size = 32
if dataset == 'IEMOCAP':
    nclass = 6
    label_names = ['exc', 'neu', 'fru', 'sad', 'hap', 'ang']
else:
    nclass = 7
    label_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
input_dim = 1024
hidden_dim = 300
lr = 1e-4
epoch = 100
cuda = True
w1 = 2   # past window
w2 = 0
layers = 2
dropout = 0.1
att_agg = False
alpha = 1e-10
init_way = 'global'   # local random embed global
fix_user = ['Chandler', 'Ross', 'Phoebe', 'Joey', 'Monica', 'Rachel']


# lr0.0001  batch32
# 39.64(w=4, l=2, drop0, alpha1e-4)
# 39.0(w=4, l=2, drop0, alpha1e-4 drop0.2)
# 39.95(w=4, l=2, drop0, alpha1e-3 drop0.)  BEST
# 39.78(w=4, l=2, drop0, alpha2e-3 drop0.)
# 39.49(w=6, l=2, drop0, alpha1e-3 drop0.)
# 39.60(w=4, l=2, drop0, alpha1e-3 drop0.01)
# 39.66(w=4, l=2, drop0, alpha1e-2 drop0.)
# 39.67(w=4, l=2, drop0, alpha5e-3 drop0.)

# 去除对角线的值之后 40.13 alpha-1e-2 BEST


# BEST   39.73(w=2, l=2, drop0, alpha1e-3) pool_agg drop0.
# BEST   39.4(w=2, l=2, drop0, alpha1e-2) pool_agg drop0.1
# BEST   39.76(w=2, l=2, drop0, alpha1e-2) pool_agg drop0.
# BEST   39.08(w=4, l=2, drop0, alpha1e-3) pool_agg drop0.
# BEST   39.38(w=6, l=2, drop0, alpha1e-3) pool_agg drop0.

# MELD 全局特征，RGAT以及拼接线性变换 Speaker GRU聚合 63.87(w=2, batch32) 64.03(drop0.1)
# 1e-10 64.09

# DailyDialog 59.3(w=4) 59.42(w=2)
# lr=1e-4 w=4 apha=1e-3 59.32
# lr=1e-4 w=4 apha=1e-3 drop0.2 59.46
# lr=1e-4 w=4 apha=1e-3 drop0.1 95.46
# lr=1e-4 w=4 apha=1e-4 drop0.2 59.3
# lr=1e-4 w=4 apha=1e-2 drop0.2 59.44
# lr=1e-4 w=4 apha=1e-2 drop0.  59.42
# lr=1e-4 w=4 apha=1e-2 drop0.1 59.57
# lr=1e-4 w=6 apha=1e-2 drop0.1 59.36
# lr=1e-4 w=5 apha=1e-2 drop0.1 59.56
# lr=1e-4 w=3 apha=1e-2 drop0.1 59.55
# lr=1e-4 layer=4 w=4 apha=1e-2 drop0.1 59.61 BEST
# lr=1e-4 layer=4 w=4 apha=1e-3 drop0.1 59.37
# lr=1e-4 layer=6 w=4 apha=1e-3 drop0.1 59.21

# 去除对角线
# lr=1e-4 layer=2 w=4 apha=1e-3 drop0.1 59.66 BEST
# lr=1e-4 layer=2 w=2 apha=1e-3 drop0.1 X
# lr=1e-4 layer=2 w=4 apha=1e-2 drop0.1 X
# lr=1e-4 layer=2 w=6 apha=1e-3 drop0.1 59.01


# lr=0.001
# IEMOCAP 68.09(lr=0.001, w=4)  68.56(w=2 drop0.1)  68.53(w=1, batchsize=8 drop0.1)
# BEST 68.99 (lr0.001, w=2, layer=2, drop0, alpha1e-3, batch8)

# 去除对角线 lambda 1e-4 69.0