import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

device='cuda' if torch.cuda.is_available() else 'cpu'

#Decoupage 192

def Decoupage192(Im1):
    List_Image_Blocs = list()
    H=Im1.shape[0]
    if (H<=192):
      List_Image_Blocs.append(Im1)
    else:
      for i in range(H//192+1):
        if (i==H//192+1): #si dernier
          Im_Bloc1 = Im1[i*192:]
          if (len(Im_Bloc2!=0)):
            List_Image_Blocs.append(Im_Bloc1)
        else:
          Im_Bloc2 = Im1[i*192:(i+1)*192]
          if (len(Im_Bloc2!=0)):
            List_Image_Blocs.append(Im_Bloc2)
    return (List_Image_Blocs)

# Quantification avec la liste Normal Float

def NF4(L_Weights, L_Approx):
  
  new_L=[]
  for i in (L_Weights):
    L_provisoire=[]
    for j in L_Approx:
      L_provisoire.append(abs(i-j))
    new_L.append(L_provisoire.index(min(L_provisoire)))
  return(new_L)

# Fonction logit

def funct(p , a: float, b:float, c:float):
  return (c*np.log(a*p/(1-b*p)))
  
# Compression

def Compression(L):
    Bloc_params=[]
    Weight_compressed=[]
    I_base=np.linspace(0.001,0.999,16)


    for i in tqdm(range(len(L))):
      W=L[i]
      Bloc_i=Decoupage192(W)
      New_bloc=[]
      for j,bloc in tqdm(enumerate(Bloc_i)): 

      # On trie par ordre croissant
        bloc_sort= bloc.sort().values.tolist()
        x_data=np.linspace(0.001,0.999,len(bloc_sort))
        popt, pcov = curve_fit(funct, x_data, bloc_sort)

        Bloc_params.append([*popt])

      #   # On approxime chaque valeur du bloc de poids a la plus proche de logit(NF4)

        Logit_NF4=funct(torch.FloatTensor(np.array(I_base)), *popt)

        New_bloc.append(NF4(torch.FloatTensor([value for value in bloc]), torch.FloatTensor(Logit_NF4)))

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
