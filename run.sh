shape=base \
seed=42 \
epochs=32 \
max_seq_length=1800 \
batchsize=4 \
lr=1e-5 \
num_workers=0 \
weight_decay=1e-6 \
highlight_hyperparameter=0.25 \
loss_hyperparameter=0.1 \

python set_model.py --shape $shape

python main.py --shape $shape \
	--seed $seed \
	--maxlen $max_seq_length \
	--epochs $epochs \
	--batchsize $batchsize \
	--lr $lr \
	--num_workers $num_workers \
	--weight_decay $weight_decay \
	--highlight_hyperparameter $highlight_hyperparameter \
	--loss_hyperparameter $loss_hyperparameter

