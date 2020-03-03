train_path='/home/chendixi/Datasets/VisDA2017/train/image_list.txt'
validation_path='/home/chendixi/Datasets/VisDA2017/validation/image_list.txt'
dataset_prefix='/home/chendixi/Datasets/VisDA2017'
dataset='visda'
num_k=(3 4 5)
epochs=2
test_interval=1

for((index=0; index<3; index++))
do
    echo "training task : num_k = ${num_k[index]}"

    CUDA_LAUNCH_BLOCKING=1 python main.py \
        --train-path ${train_path} \
        --validation-path ${validation_path} \
        --dataset-prefix ${dataset_prefix}\
        --dataset ${dataset} \
        --num-k ${num_k[index]} \
        --epochs ${epochs} \
        --test-interval ${test_interval}

done
