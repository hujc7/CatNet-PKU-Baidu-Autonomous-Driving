# Training note

resetn50 

resnet18 Trained with 20 epochs, final training loss ~13, model does not converge. Testing again with 40 epoches

resnet18 with lr=0.01, 30 epochs and multistep scheduler [0.6, 0.9]. Did not overfit

resnet18 with multistep scheduler, either [0.5, 0.8] or [0.6, 0.9] yield similar results overfit to around 100 dev loss

resnet50 train with 60 epochs, 1 raw, 2 cut aug, 3 color aug
     private public
raw: 0.017 0.018
cug: 0.015 0.017
color: 0.012 0.01
raw has the bets performance.

resnet 50 train with fpn, 60 epochs.
* same performance with raw resnet50 for private score, but lower public score
  * Probably means less overfitting

According to https://www.kaggle.com/code/phoenix9032/center-resnet-starter/notebook, the network is trained from scratch and can reach 0.05 public score. Tested it out but it does not work. Need to use pretrained network.
* Running the original script yields the same performane as stated
* Looking into the difference
  * Model used resnet18 with groupnrom
  