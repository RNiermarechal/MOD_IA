from main_neurones import *

path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/data_batch_1'
def change_nb_neurons(path):
    losses=[]
    accuracies=[]
    for nb_neurons in range(1,2000,1000):
        loss_list,accuracy_list=train_nn_2layers(path,nb_neurons,1e-2,200)
        losses.append(loss_list)
        accuracies.append(accuracy_list)
    plt.subplot(211)
    for loss in losses:
        plt.plot(loss)
    plt.subplot(212)
    for accuracy in accuracies:
        plt.plot(accuracy)
    #plt.show()

    losses_end=[] # loss à la fin de l'optimisation des poids
    accuracies_end=[] # accuracy à la fin de l'optimisation des poids
    for nb_neurons in range(1,2000,1000):
        loss_list,accuracy_list=train_nn_2layers(path,nb_neurons,1e-2,200)
        losses_end.append(loss_list[-1])
        accuracies_end.append(accuracy_list[-1])
    plt.figure()
    plt.subplot(211)
    plt.plot(losses_end)
    plt.subplot(212)
    plt.plot(accuracies_end)
    plt.show()




        

change_nb_neurons(path)