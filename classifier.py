#!/usr/bin/python3

# https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
# 1 - Carregar bibliotecas
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn.metrics import accuracy_score

# 2 -Variáveis globais

classes = ['Alien','Predator']
train_path = './data/train/'
test_path  = './data/test/'
best_epoch = 0
batch_size = 10   #número de amostras a passar antes de atualizar o modelo (01)
#Ref: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
valid_size = 0.3 #porcentagem do dataset de treino usado para validação 
n_epochs   = 10  #quantas vezes o dataset sera passado pela rede	  (02)

# Data Augmentation
transforms_train = transforms.Compose([
	transforms.ToPILImage(), 
	transforms.RandomHorizontalFlip(), # Gira pil horizontalmente
	transforms.RandomRotation(10), # Rotaciona a imagem em um certo angulo
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transforms_test = transforms.Compose([
	transforms.ToPILImage(),# Converte ndarray/tensor para imagem (PIL)
	transforms.ToTensor(), # Tensor é uma matrix multidimensional mesmo tipo
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normaliza tensor (?)
])

# 3 - Definição de classes e funções

#Carrega o dataset, as imagens
class CreateDataset(Dataset):
	def __init__(self, df_data, data_dir = './', transform=None):
		super().__init__()
		self.df = df_data.values
		self.data_dir  = data_dir
		self.transform = transform

	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, index):
		img_name,label = self.df[index]
		img_path = os.path.join(self.data_dir, img_name)
		image = cv2.imread(img_path) # retorna np.ndarray
		image = image_pad(image,False) # faz padding para todas imagens terem a mesma dimensão

		if self.transform is not None:
			image = self.transform(image)
		return image, label

# Define arquitetura da cnn
# conv : camada de convolução, extração de caracteristicas (03)
# pool : maxpooling layer, basicamente faz uma redução de ordem da imagem (04)
# fc   : fullyconnect layer, mlp full connected (05)
#          Ref : https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
# dropout : tecnica de regularização para evitar overfitting, ignora um certa quantidade de neurônios no processo de treinamento, os neurônios são escolhidos aleatoriamente
#          Ref : https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5
# relu : È a função de ativação
# https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU
# mostra o valor se for positivo e zero se ele for negativo
# https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
class CNN(nn.Module):
		def __init__(self):
				super(CNN, self).__init__()
				self.conv1 = nn.Conv2d( 3, 16,3,padding=1)
				self.conv2 = nn.Conv2d(16, 32,3,padding=1)
				self.conv3 = nn.Conv2d(32, 64,3,padding=1)
				self.conv4 = nn.Conv2d(64,128,3,padding=1)
				self.pool = nn.MaxPool2d(2, 2)
				self.fc1 = nn.Linear(128*25*25, 512)
				self.fc2 = nn.Linear(512, 100)
				self.fc3 = nn.Linear(100, 2)

				self.dropout = nn.Dropout(0.2)
				
		def forward(self, x):
				#400x400x3
				x = self.pool(F.relu(self.conv1(x)))
				#200x200x16
				x = self.pool(F.relu(self.conv2(x)))
				#100x100x32
				x = self.pool(F.relu(self.conv3(x)))
				#050x050x64
				x = self.pool(F.relu(self.conv4(x)))
				#025x025x128

				x = x.view(-1, 128 * 25 * 25)
				x = self.dropout(x)
				x = F.relu(self.fc1(x))
				x = self.dropout(x)
				x = F.relu(self.fc2(x))
				x = self.dropout(x)
				x = F.relu(self.fc3(x))

				return x

def image_pad(image,resize):
	c_max = 400
	if resize:
		old_size = image.shape[:2]
		ratio    = float(c_max)/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])
		image    = cv2.resize(image,(new_size[1],new_size[0]))

		delta_w = c_max - old_size[1]
		delta_h = c_max - old_size[0]
		top , bottom = delta_h//2, delta_h-(delta_h//2)
		left, right  = delta_w//2, delta_w-(delta_w//2)

		color  = [0,0,0]
		new_im = cv2.copyMakeBorder(image,
					    top,
					    bottom,
					    left,
					    right,
					    cv2.BORDER_CONSTANT,
					    value=color)
	else:
		old_size = image.shape[:2]
		delta_w = c_max - old_size[1]
		delta_h = c_max - old_size[0]
		top , bottom = delta_h//2, delta_h-(delta_h//2)
		left, right  = delta_w//2, delta_w-(delta_w//2)

		color  = [0,0,0]
		new_im = cv2.copyMakeBorder(image,
					    top,
					    bottom,
					    left,
					    right,
					    cv2.BORDER_CONSTANT,
					    value=color)
	
	return new_im
	
def imshow(img):
    '''Helper function to un-normalize and display an image'''
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

# Função que ira treinar a cnn
def cnn_train(model,train_loader,valid_loader):
	global best_epoch

	#loss function : avalia a predição, seus valores ficam altos caso a predição desvie muito do esperado
	# Ref : https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
	#       https://en.wikipedia.org/wiki/Loss_function
	# crossentropyloss , seu valor aumenta quando as probabilidades preditas divergem da classe esperada
	# Ref : https://pytorch.org/docs/stable/nn.html
	#	https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
	criterion = nn.CrossEntropyLoss()
	#otimizador
	# Ref : https://pytorch.org/docs/stable/_modules/torch/optim/adamax.html
	optimizer = optim.Adamax(model.parameters(), lr=0.001)

	valid_loss_min = np.Inf
	train_losses = []
	valid_losses = []

	for epoch in range(1, n_epochs+1):
		train_loss = 0.0
		valid_loss = 0.0

		model.train() # https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/
		for data, target in train_loader:
			optimizer.zero_grad() # # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
			output = model(data)
			loss   = criterion(output, target) # calculo do erro
			loss.backward()  # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
			optimizer.step() # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
			train_loss += loss.item()*data.size(0)

		model.eval() # https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/
		for data, target in valid_loader:
			output = model(data)
			loss = criterion(output, target)
			valid_loss += loss.item()*data.size(0)

		train_loss = train_loss/len(train_loader.sampler)
		valid_loss = valid_loss/len(valid_loader.sampler)
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
		if valid_loss <= valid_loss_min:
			best_epoch = epoch
			torch.save(model.state_dict(), 'best_model.pt')
			valid_loss_min = valid_loss

# 4 - Main
#Carrega dados 

train_df = pd.read_csv("./train.csv")

train_data = CreateDataset(df_data=train_df,data_dir=train_path, transform=transforms_train)

# 1/4 of the dataset consiste of 1's if the result is bad maybe this could be something to change (skewed classes)
#print(len(train_df[train_df['pneumonia'] == 0])/len(train_df))
#print(len(train_df[train_df['pneumonia'] == 1])/len(train_df))

num_train = len(train_data)
indices   = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

sample_sub  = pd.read_csv("./test.csv")
test_data   = CreateDataset(df_data=sample_sub,data_dir=test_path,transform=transforms_test)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

#Mostra 20 imagens do dataset

dataiter = iter(train_loader)
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
	images, labels = dataiter.next()
	images = images.numpy()

	ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
	imshow(images[0])
	ax.set_title(classes[labels[0]])

#plt.show()

#Mostra arquitetura da rede e mais alguns parametros 

model = CNN()
print("batch_size = " + str(batch_size))
print("n_epochs   = " + str(n_epochs))
print(model)

#Treina a rede e mostra em qual epoch obtivemos o menor erro

cnn_train(model,train_loader,valid_loader)
print("best_epoch = " + str(best_epoch))

#Carrega o melhor modele e avalia sua acuracia no dataset de teste

model.load_state_dict(torch.load('./best_model.pt'))
preds = []

for batch_i, (data,target) in enumerate(test_loader):
	output = model(data)
	_, predicted = torch.max(output,1) # retorna o index do neurônio cuja saida teve o maior
	predicted = predicted.numpy()
	for i in predicted:
		preds.append(i)

acc = accuracy_score(sample_sub['class'].values,preds)
print("Accuracy   = " + str(acc))
