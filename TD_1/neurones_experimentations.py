from main_neurones import *

path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/data_batch_1'
def change_nb_neurons(path):
    losses=[]
    accuracies=[]
    list_nb_neurons=range(1,2000,1000)
    for nb_neurons in list_nb_neurons:
        loss_list,accuracy_list=train_nn_2layers(path,nb_neurons,1e-2,200)
        losses.append(loss_list)
        accuracies.append(accuracy_list)
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
    #plt.show()

    losses_end=[] # loss à la fin de l'optimisation des poids
    accuracies_end=[] # accuracy à la fin de l'optimisation des poids
    list_Dh=range(1,2000,1000)
    for nb_neurons in list_Dh:
        loss_list,accuracy_list=train_nn_2layers(path,nb_neurons,1e-2,200)
        losses_end.append(loss_list[-1])
        accuracies_end.append(accuracy_list[-1])
    plt.figure()
    plt.subplot(211)
    plt.title("Influence du nombre de neurones de la couche cachée sur la performance du classifieur")
    plt.plot(list_Dh,losses_end)
    plt.ylabel("Loss function à la fin de l'entrainement")
    plt.subplot(212)
    plt.plot(list_Dh,accuracies_end)
    plt.ylabel("Accuracy à la fin de l'entrainement")
    plt.xlabel("Nombre de neurones de la couche cachée")
    plt.show()


def change_learning_rate(path):
    losses=[]
    accuracies=[]
    lr_list=[1e-4,1e-3,1e-2,1e-1]
    for lr in lr_list:
        loss_list,accuracy_list=train_nn_2layers(path,1000,lr,200)
        losses.append(loss_list)
        accuracies.append(accuracy_list)
    plt.figure()
    plt.subplot(211)
    plt.title("Influence du learning rate sur la vitesse d'apprentissage")
    for i,loss in enumerate(losses):
        plt.plot(loss,label='l_R = '+str(lr_list[i]))
    plt.legend()

    plt.subplot(212)
    for i,accuracy in enumerate(accuracies):
        plt.plot(accuracy,label='l_R = '+str(lr_list[i]))
    plt.legend()
    plt.show()

        

change_nb_neurons(path)
change_learning_rate(path)