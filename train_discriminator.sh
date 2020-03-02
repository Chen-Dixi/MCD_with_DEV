lr=0.001
epochs=6
netD='TrainedModel/netD/netD.pth'
python train_discriminator.py \
    --lr ${lr} \
    --epochs ${epochs}
    #--netD ${netD}
