checkpoint_path='/home/chendixi/Research/Experiments/Domain_Adaptation/MCD_with_DEV/TrainedModel/resnet50/num_k=3/best_model_checkpoint.pth.tar'
netD='TrainedModel/netD/netD.pth'
python DEV.py \
    --netD ${netD} \
    --checkpoint-path ${checkpoint_path}


