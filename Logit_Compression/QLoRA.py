import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device='cuda' if torch.cuda.is_available() else 'cpu'

L_QLoRA=torch.FloatTensor([-1.0, -0.6961928009986877, -0.5250730514526367,-0.39491748809814453, -0.28444138169288635, -0.18477343022823334,-0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,0.24611230194568634, 0.33791524171829224, 0.44070982933044434,0.5626170039176941, 0.7229568362236023, 1.0]).to(device)

def Decoupage64(Tensor1):
  return(torch.split(Tensor1, 64))

def Decoupage256(Tensor2):
  return(torch.split(Tensor2, 256))

# Quantification avec la liste Normal Float


def NF4(L_Weights, L_Approx):
  L_Approx.to(device)
  L_expand=L_Weights.expand(16,64).to(device)
  bloc0_transp=torch.transpose(L_expand, 0, 1)
  L_Weights=torch.argmin(torch.abs(torch.sub(bloc0_transp,L_QLoRA)),dim=1)
  return(L_Weights)


#Compression Poids
def CompressionPoids(L):

  List_rescale_values=list()
  Total_List_W=list()

  # Quantification pour chaque bloc de chaque poids

  for i in tqdm(range(len(L))):

    #Divise en bloc de 64 valeurs
    Image_bloc=Decoupage64(L[i])

    New_list_bloc=[]

    #Pour chaque bloc
    for bloc in (Image_bloc):
      #Stocke valeur de normalisation
      max=abs(torch.max(bloc))
      List_rescale_values.append(max.item())

      #On approxime avec liste QLoRA
      bloc_norm=F.normalize(bloc,dim=0).to(device)
      New_list_bloc.append(NF4(bloc_norm,L_QLoRA))

    Total_List_W.append(New_list_bloc) #ensemble des poids 64 normalisés

  return(List_rescale_values,Total_List_W)

# Quantification pour chaque valeurs rescale

def CompressionRescale(List_rescale_values):
  Tensor_rescale_values=torch.FloatTensor(List_rescale_values)
  Bloc_rescale=Decoupage256(Tensor_rescale_values)
  values_mean=[]
  new_bloc_rescale=[]

  for bloc in Bloc_rescale:
    mean=torch.mean(bloc) #valeur moyenne bloc
    new_bloc_rescale.append(NF4([value-mean for value in bloc],L_QLoRA))
    values_mean.append(mean)
  return(values_mean,new_bloc_rescale)

## Decompression Valeurs Rescale

def DecompressionRescale(values_mean,new_bloc_rescale):
    Rescale_decompresse=[]
    for i,rescale in enumerate(new_bloc_rescale):
      Rescale_decompresse.append([L_QLoRA[value]+values_mean[i] for value in rescale]) #new_bloc

    new_rescale_bien_mise=[]
    for sub_list in Rescale_decompresse:
      new_rescale_bien_mise.extend(sub_list)
    return(new_rescale_bien_mise)

## Decompression Poids

def DecompressionPoids(new_rescale_bien_mise,Total_List_W):

    W_total=[]
    k=0
    for W_quantif in Total_List_W: #pour chaque poids
      W_decompresse=[]

      for bloc in W_quantif: #pour chaque bloc -> 4bit à valeur et après valeur *normalisation
        valeur_rescale=new_rescale_bien_mise[k]
        k+=1
        W_decompresse.append([L_QLoRA[value]*valeur_rescale for value in bloc]) #new_bloc
      W_total.append(W_decompresse)

    new_QLORA=[]
    for weight in W_total:
      L_Qlora=[]
      for bloc in weight:
        L_Qlora.extend(bloc)
      new_QLORA.append(L_Qlora)

    return(new_QLORA)