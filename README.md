The following repository contains the source code to help investigate the 
effects of knowledge distillation for neural network compression.

## MNIST
contains initial tests/results with MNIST
### InitialImpl.py
Source code that was run on Google Colab
https://colab.research.google.com/drive/1UlbQGrmDsMsgcKANhjgyUBExSHfYiZRl

## CIFAR100
contains secondary tests with cifar100 mobileNet and Resnet18
### 0. Tensorboard (via AWS SSH or GCP SSH)
```
(ssh into remote)
$ ssh -i <pemfile>.pem -L 16006:127.0.0.1:6006 ubuntu@<awsPublicDNS>
or
$ <view gcloud command> -- -L 16006:127.0.0.1:6006
(start new screen)
$ screen -S tensorboard
(in a new screen)
$ pip install tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
$ ctrl-a ctrl-d
(back in original terminal)
```
Now you can access the tensorboard visuals by going to http://127.0.0.1:16006

### 1. Train without KD
```
$ python train.py -model <modelname>
```

### 2. Train with KD
```
$ python trainDistill.py -model <studentModelName> -tSaved <savedTeacher> -tModel <teacherModelName>
(for example)
$ python trainDistill.py -model mobilenet -tSaved SavedModels/resnet18/lr03/resnet18-199.pth -tModel resnet18
````

### 3. Train with EOS
```
$ python trainEOS.py -model <studentModelName> -tSaved <savedTeacher> -tModel <teacherModelName>
$ python trainEOS.py -model mobilenet -tSaved SavedModels/resnet18/lr03/resnet18-199.pth -tModel resnet18
````

### 4. Test
```
$ python test.py -model <modelname> -state <savedModel>
```
