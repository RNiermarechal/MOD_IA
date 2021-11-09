import numpy as np
import skimage
from main_kppv import unpickle,lecture_cifar,evaluation_classifieur
import matplotlib.pyplot as plt

np.random.seed(1) # pour que l'exécution soit déterministe



def forward_2layers(X,W1,b1,W2,b2,Y):
    ###################################################
    # Passe avant : calcul de la sortie prédite Y_pred pour réseau à 2 couches#
    ###################################################
    I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
    O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
    O2 = 1/(1+np.exp(-I2)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
    Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie
    #######################################################
    # Calcul de la fonction perte de type MSE #
    #######################################################
    loss = np.square(Y_pred - Y).sum() / 2
    return O1,Y_pred,loss

def back_propagate_2layers(X,Y,O1,O2,W1,W2,B1,B2,lr):
    ## Gradient pour W2
    dL_dI2=(O2-Y)*((1-O2)*O2)
    dL_dw2=np.dot(np.transpose(O1),dL_dI2)

    ## Gradient pour W1
    dL_dO1=np.dot(dL_dI2,np.transpose(W2))
    dL_dI1=np.multiply(dL_dO1,(O1*(1-O1)))
    dL_dw1=np.dot(np.transpose(X),dL_dI1)

    ## Gradient pour B2
    dL_db2=dL_dI2

    ## Gradient pour B1
    dL_db1=dL_dI1
    
    # Update des parametres du modele
    w1=W1-dL_dw1*lr # pour w1
    b1=B1-dL_db1*lr# pour b1
    w2=W2-dL_dw2*lr # pour w2
    b2=B2-dL_db2*lr # pour b2
    
    return w1,b1,w2,b2

def train_nn_2layers(X,Y,D_h,lr,n_iter):
    # Géometrie du problème
    X=X[:,:]/255 # rescale pour faciliter la convergence
    Y=Y[:,:]/9
    N,D_in=X.shape
    D_out=1
    
    # Initialisation aléatoire des poids du réseau
    W1 = 2 * np.random.random((D_in, D_h)) - 1
    B1 = np.zeros((1,D_h))
    W2 = 2 * np.random.random((D_h, D_out)) - 1
    B2 = np.zeros((1,D_out))

    ## Back propagation
    loss_list=[]
    accuracy_list=[]
    for n in range(n_iter):
        O1,O2,loss=forward_2layers(X, W1, B1, W2, B2, Y)
        W1,B1,W2,B2=back_propagate_2layers(X, Y, O1, O2, W1, W2, B1, B2, lr)
        loss_list.append(loss)
        accuracy_list.append(evaluation_classifieur(Y, np.round(O2)))

    return loss_list,accuracy_list
    
def train_nn_3layers(X,Y,D_h,D_h2,lr,n_iter):
    X=X[:,:]/255 # rescale pour faciliter la convergence
    Y=Y[:,:]/9
    N,D_in=X.shape
    D_out=1

    def forward_3layers(X,W1,b1,W2,b2,W3,b3,Y):
        I1 = X.dot(W1) + b1 # Potentiel d'entrée de la 1e couche cachée
        O1 = 1/(1+np.exp(-I1)) # Sortie de la 1e couche cachée (fonction d'activation de type sigmoïde)
        I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la 2e couche cachée
        O2 = 1/(1+np.exp(-I2)) # Sortie de la seconde couche cachée (fonction d'activation de type sigmoïde)
        I3 = O2.dot(W3) + b3 # Potentiel d'entrée de la couche de sortie
        O3 = 1/(1+np.exp(-I3)) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)

        Y_pred = O3 # Les valeurs prédites sont les sorties de la couche de sortie
        loss = np.square(Y_pred - Y).sum() / 2
        return O1,O2,Y_pred,loss
    
    def back_propagate_3layers(X,Y,O1,O2,O3,W1,W2,W3,B1,B2,B3,lr):
        #Gradient pour couche de sortie
        dL_dI3=(O3-Y)*((1-O3)*O3)
        dL_dw3=np.dot(np.transpose(O2),dL_dI3)

        #Gradient pour 2e couche cachée
        dL_dO2=np.dot(dL_dI3,np.transpose(W3))
        dL_dI2=np.multiply(dL_dO2,(O2*(1-O2)))
        dL_dw2=np.dot(np.transpose(O1),dL_dI2)

        #Gradient pour 1e couche cachée
        dL_dO1=np.dot(dL_dI2,np.transpose(W2))
        dL_dI1=np.multiply(dL_dO1,(O1*(1-O1)))
        dL_dw1=np.dot(np.transpose(X),dL_dI1)

        dL_db3=dL_dI3 #Gradient biais 3
        dL_db2=dL_dI2 #Gradient biais 2
        dL_db1=dL_dI1 #Gradient biais 1

        # Update des parametres du modele
        w1=W1-dL_dw1*lr 
        b1=B1-dL_db1*lr
        w2=W2-dL_dw2*lr 
        b2=B2-dL_db2*lr
        w3=W3-dL_dw3*lr 
        b3=B3-dL_db3*lr
        
        return w1,b1,w2,b2,w3,b3

    W1 = 2 * np.random.random((D_in, D_h)) - 1
    B1 = np.zeros((1,D_h))
    W2 = 2 * np.random.random((D_h, D_h2)) - 1
    B2 = np.zeros((1,D_h2))
    W3 = 2 * np.random.random((D_h2, D_out)) - 1
    B3 = np.zeros((1,D_out))

    ## Back propagation
    loss_list=[]
    accuracy_list=[]
    for n in range(n_iter):
        O1,O2,O3,loss=forward_3layers(X, W1, B1, W2, B2, W3, B3, Y)
        W1,B1,W2,B2,W3,B3=back_propagate_3layers(X, Y, O1, O2, O3, W1, W2, W3, B1, B2, B3, lr)
        loss_list.append(loss)
        accuracy_list.append(evaluation_classifieur(Y*9, np.round(O3*9)))
    
    return loss_list, accuracy_list