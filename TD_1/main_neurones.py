import numpy as np
import skimage
from main_kppv import unpickle,lecture_cifar
np.random.seed(1) # pour que l'exécution soit déterministe

path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/TD_1/cifar-10-batches-py/data_batch_1'

X,Y=lecture_cifar(path)

##########################
# Génération des données #
##########################

# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)
N, D_in, D_h, D_out = 30, 2, 10, 3

# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires

X = np.random.random((N, D_in))
Y = np.random.random((N, D_out))

# Initialisation aléatoire des poids du réseau
W1 = 2 * np.random.random((D_in, D_h)) - 1
b1 = np.zeros((1,D_h))
W2 = 2 * np.random.random((D_h, D_out)) - 1

b2 = np.zeros((1,D_out))

####################################################
# Passe avant : calcul de la sortie prédite Y_pred #
####################################################

I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
O2 = 1/(1+np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie

########################################################
# Calcul et affichage de la fonction perte de type MSE #
########################################################
loss = np.square(Y_pred - Y).sum() / 2
print(loss)