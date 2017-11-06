# kaggle-Iceberg

# Update:
10.26 (v0.1):

- simple network
```
Epoch 56/100
0s - loss: 0.1940 - acc: 0.9192 - val_loss: 0.2143 - val_acc: 0.8910
```
- public LB is 0.19822


10.27 (v0.2):

- simple network + more channels
```
Epoch 50/500
0s - loss: 0.1531 - acc: 0.9352 - val_loss: 0.2329 - val_acc: 0.9003
```
- public LB is 0.1901

10.28 (v0.2)

- add shear augmentation
```
Epoch 50/500
0s - loss: 0.1669 - acc: 0.9390 - val_loss: 0.2052 - val_acc: 0.9159
```
- public LB is 0.1855

10.29 (v0.3)
- ensemble with cross validation
- public LB is 0.1739

11.06 (v0.4)
- add random crop data augmentation and the 3rd dummy channel
- public LB is 0.1787 for single model and 0.1726 for ensemble model