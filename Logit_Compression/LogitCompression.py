import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

device='cuda' if torch.cuda.is_available() else 'cpu'

I_base=torch.FloatTensor(np.linspace(0.001,0.999,16)).to(device)

#Decoupage 192

def Decoupage192(Tensor1):
  return(torch.split(Tensor1, 192))

# Quantification avec la liste Normal Float

def NF4(L_Weights, L_Approx):
  L_expand=L_Weights.expand(16,192).to(device)
  bloc0_transp=torch.transpose(L_expand, 0, 1)
  L_Weights=torch.argmin(torch.abs(torch.sub(bloc0_transp,I_base)),dim=1)
  return(L_Weights)

# Fonction logit

def funct(p , a: float, b:float, c:float):
  return (c*np.log(a*p/(1-b*p)))
  
# Compression

def Compression(L):
    Bloc_params=[]
    Weight_compressed=[]
    for i in tqdm(range(len(L))):
      W=L[i]
      Bloc_i=Decoupage192(W)
      New_bloc=[]
      for j,bloc in enumerate(Bloc_i): 

      # On trie par ordre croissant
        bloc_sort= bloc.sort().values.tolist().cpu()
        x_data=np.linspace(0.001,0.999,len(bloc_sort))
        popt, pcov = curve_fit(funct, x_data, bloc_sort)

        Bloc_params.append([*popt])

      #   # On approxime chaque valeur du bloc de poids a la plus proche de logit(NF4)

        Logit_NF4=funct(I_base, *popt).to(device)
        bloc.to(device)

        New_bloc.append(NF4(bloc, Logit_NF4))

        Weight_compressed.append(New_bloc)

    return(Bloc_params,Weight_compressed)

## Decompression

def Decompression(Bloc_params,Weight_compressed):

    bloc_param=0
    W_decompressed=[]
    for i,Weight in enumerate(Weight_compressed):
      Bloc_decompressed=[]
      for bloc_compressed in Weight: #pour chaque bloc
          Bloc_decompressed.append(funct(I_base[bloc_compressed], *Bloc_params[bloc_param]))
          bloc_param+=1
      W_decompressed.append(Bloc_decompressed)

    new_weight_total=[]
    for weight in W_decompressed:
      new_list=[]
      for bloci in weight:
        new_list.extend(bloci)
      new_weight_total.append(new_list)
    return new_weight_total
