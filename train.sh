cd /sde1/qinyang/projects/Cross-modal/RCSL

init_txt=glove
bs=256
gpu=0
data_path=/sde1/hupeng/ImageText/SCAN/data/data
vocab_path=/sde1/qinyang/projects/EvidenceTextImage/ETI/data/vocab

tau=0.03

data=f30k #coco 
gpu=0
data_name=${data}_precomp
for paired_length in 500 1000 2000 5000
do
tag=${init_txt}_pl${paired_length}_bs${bs}_tau${tau}   
logger_path=./runsx/${data}_${tag}/log
model_path=./runsx/${data}_${tag}/checkpoint

CUDA_VISIBLE_DEVICES=$gpu python3 train.py --gpu $gpu  --paired_length $paired_length\
  --data_name ${data_name} --logger_path $logger_path --model_path $model_path   --init_txt $init_txt \
  --log_step 200 --embed_size 1024 --tau $tau --batch_size $bs --data_path $data_path --vocab_path $vocab_path
done