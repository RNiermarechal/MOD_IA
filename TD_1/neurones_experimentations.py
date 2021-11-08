from main_neurones import *
from kppv_experimentations import *

path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/'
def change_nb_neurons(path):
    X,Y=lecture_cifar(path)
    losses=[]
    accuracies=[]
    list_nb_neurons=range(1,200,50)
    for nb_neurons in list_nb_neurons:
        loss_list,accuracy_list=train_nn_2layers(X,Y,nb_neurons,1e-3,200)
        losses.append(loss_list)
        accuracies.append(accuracy_list)
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title("Influence du nombre de neurones de la couche cachée sur la vitesse de convergence")
    for i,loss in enumerate(losses):
        plt.plot(loss,label=str(list_nb_neurons[i])+' neurones')
    plt.ylabel("Loss function")
    plt.legend()
    plt.subplot(212)
    for i,accuracy in enumerate(accuracies):
        plt.plot(accuracy,label=str(list_nb_neurons[i])+' neurones')
    plt.ylabel("Accuracy")
    plt.xlabel("Itérations")
    plt.legend()
    plt.savefig(fname='results_change_nb_neurons_1.png',format='png')
    print("Influence du nombre de neurones de la couche cachée sur la vitesse de convergence : DONE")

    #plt.show()

    losses_end=[] # loss à la fin de l'optimisation des poids
    accuracies_end=[] # accuracy à la fin de l'optimisation des poids
    list_Dh=range(1,2000,50)
    for nb_neurons in list_Dh:
        loss_list,accuracy_list=train_nn_2layers(X,Y,nb_neurons,1e-3,200)
        losses_end.append(loss_list[-1])
        accuracies_end.append(accuracy_list[-1])
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title("Influence du nombre de neurones de la couche cachée sur la performance du classifieur")
    plt.plot(list_Dh,losses_end)
    plt.ylabel("Loss function à la fin de l'entrainement")
    plt.subplot(212)
    plt.plot(list_Dh,accuracies_end)
    plt.ylabel("Accuracy à la fin de l'entrainement")
    plt.xlabel("Nombre de neurones de la couche cachée")
    plt.savefig(fname='results_change_nb_neurons_2.png',format='png')
    print("Influence du nombre de neurones de la couche cachée sur la performance du classifieur : DONE")
    print("Etude sensibilité nb de neurones effectuée")

def change_learning_rate(path):
    X,Y=lecture_cifar(path)
    losses=[]
    accuracies=[]
    lr_list=[1e-4,1e-3,1e-2,1e-1]
    for lr in lr_list:
        loss_list,accuracy_list=train_nn_2layers(X,Y,1000,lr,200)
        losses.append(loss_list)
        accuracies.append(accuracy_list)
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title("Influence du learning rate sur la vitesse d'apprentissage")
    for i,loss in enumerate(losses):
        plt.plot(loss,label='l_R = '+str(lr_list[i]))
    plt.legend()

    plt.subplot(212)
    for i,accuracy in enumerate(accuracies):
        plt.plot(accuracy,label='l_R = '+str(lr_list[i]))
    plt.legend()
    plt.savefig(fname='results_change_learning_rate.png',format='png')
    print("Etude sensibilité learning rate effectuée")
        
def use_mini_batch(X,Y,nb_batches,n_iter,D_h,lr):
    
    
    N,D_in=X.shape
    D_out=1
    # Génération aléatoire des mini batches
    mini_batches=np.split(np.random.permutation(np.arange(0,N)),nb_batches)
    D_out=1
    
    # Initialisation aléatoire des poids du réseau
    W1 = 2 * np.random.random((D_in, D_h)) - 1
    B1 = np.zeros((1,D_h))
    W2 = 2 * np.random.random((D_h, D_out)) - 1
    B2 = np.zeros((1,D_out))
    loss_list=[]
    accuracy_list=[]
    for n in range(n_iter):
        for batch in mini_batches:
        
            X_batch=X[list(batch)]/255
            Y_batch=Y[list(batch)]/9
    
        ## Back propagation
        
            O1,O2,loss=forward_2layers(X_batch, W1, B1, W2, B2, Y_batch)
            W1,B1,W2,B2=back_propagate_2layers(X_batch, Y_batch, O1, O2, W1, W2, B1, B2, lr)
            loss_list.append(loss)
            accuracy_list.append(evaluation_classifieur(Y_batch*9, np.round(O2*9)))
    
    return loss_list,accuracy_list

def mini_batches(path):
    X,Y=lecture_cifar(path)
    nb_batches=10
    n_iter=250
    D_h=500
    lr=1e-3
    loss_mini_b,accuracy_mini_b=use_mini_batch(X,Y, nb_batches, n_iter, D_h,lr)
    loss_full,accuracy_full=train_nn_2layers(X,Y, D_h, lr, n_iter)
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title("Utilisation de 10 mini - batches")
    plt.plot(loss_mini_b, label="Avec mini-batch")
    plt.plot(loss_full,label="Sans mini-batch, échantillon complet")
    plt.ylabel("Loss function")
    plt.legend()
    plt.subplot(212)
    plt.plot(accuracy_mini_b,label="Avec mini-batch")
    plt.plot(accuracy_full,label="Sans mini-batch, échantillon complet")
    plt.ylabel("Accuracy")
    plt.xlabel("Itérations")
    plt.legend()
    plt.show()
    

def change_nb_layers(path):
    X,Y=lecture_cifar(path)
    D_h=1000
    D_h2=1000
    lr=1e-3
    n_iter=300
    loss_list_2,accuracy_list_2=train_nn_2layers(X,Y, D_h, lr, n_iter)
    loss_list_3,accuracy_list_3=train_nn_3layers(X,Y, D_h,D_h2, lr, n_iter)
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title("Ajout d'une couche : effet sur la vitesse de convergence")
    plt.plot(loss_list_2,label='2 couches')
    plt.plot(loss_list_3,label='3 couches')
    plt.legend()
    plt.ylabel("Loss function")
    plt.subplot(212)
    plt.plot(accuracy_list_2,label='2 couches')
    plt.plot(accuracy_list_3,label='3 couches')
    plt.legend()
    plt.ylabel("Activation function")
    plt.xlabel("Nb d'itérations")
    plt.savefig(fname='results_change_nb_layers.png',format='png')
    print("Ajout d'une couche pour la vitesse de convergence : DONE")
    # plt.show()

## Lancement des tests
# try:
#     change_nb_neurons(path)
# try:
#     change_learning_rate(path)
# try:
#     mini_batches(path)
try:
    change_nb_layers(path)
except:
    print("Erreur dans change nb layers")
