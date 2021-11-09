from main_kppv import *

path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/'

## Influence nb de voisins
def influence_param_k(path,k_max):
    X,Y=lecture_cifar(path)
    X=X[:20000,:]
    Y=Y[:20000,:]
    X_app,Y_app,X_test,Y_test=decoupage_donnees(X, Y)
    Dist=kppv_distances(X_test, X_app)
    l_k=range(1,k_max,2)
    l_accuracy=[]
    for k in l_k:
        Y_pred=kppv_predict(Dist, Y_app, k)
        kppv_res=evaluation_classifieur(Y_test, Y_pred)
        l_accuracy.append(kppv_res)
    plt.figure()
    plt.plot(l_k,l_accuracy)
    plt.xlabel('k voisins')
    plt.ylabel('Accuracy (%)')
    plt.title("Variation du nombre de voisins")
    
    # plt.savefig(fname='influence_k_results.png',format='png')
    # plt.savefig(fname='influence_k_results.svg',format='svg')
    return l_k,l_accuracy

## Influence des descripteurs
def influence_param_k_LBP(path,k_max):
    X,Y=lecture_cifar(path)
    X=X[:20000,:]
    Y=Y[:20000,:]
    X=add_LPB(X)
    X_app,Y_app,X_test,Y_test=decoupage_donnees(X, Y)
    Dist=kppv_distances(X_test, X_app)
    l_k=range(1,k_max,2)
    l_accuracy=[]
    for k in l_k:
        Y_pred=kppv_predict(Dist, Y_app, k)
        kppv_res=evaluation_classifieur(Y_test, Y_pred)
        l_accuracy.append(kppv_res)
    plt.figure()
    plt.plot(l_k,l_accuracy)
    plt.xlabel('k voisins')
    plt.ylabel('Accuracy (%)')
    plt.title("Utilisation du descripteur LBP")
    
    # plt.savefig(fname='influence_k_avec_lbp_results.png',format='png')
    # plt.savefig(fname='influence_k_avec_lbp_results.svg',format='svg')
    return l_k,l_accuracy

def influence_param_k_HOG(path,k_max):
    X,Y=lecture_cifar(path)
    X=X[:20000,:]
    Y=Y[:20000,:]
    X=add_HOG(X)
    X_app,Y_app,X_test,Y_test=decoupage_donnees(X, Y)
    Dist=kppv_distances(X_test, X_app)
    l_k=range(1,k_max,2)
    l_accuracy=[]
    for k in l_k:
        Y_pred=kppv_predict(Dist, Y_app, k)
        kppv_res=evaluation_classifieur(Y_test, Y_pred)
        l_accuracy.append(kppv_res)
    plt.figure()
    plt.plot(l_k,l_accuracy)
    plt.xlabel('k voisins')
    plt.ylabel('Accuracy (%)')
    plt.title("Utilisation du descripteur HOG")
    
    # plt.savefig(fname='influence_k_avec_hog_results.png',format='png')
    # plt.savefig(fname='influence_k_avec_hog_results.svg',format='svg')

    return l_k,l_accuracy

## Influence validation croisée
def influence_param_k_avec_validation_croisee(path,k_max,nb_batches):
    X,Y=lecture_cifar(path)
    X=X[:20000,:]
    Y=Y[:20000,:]
    N,D_in=X.shape

    # Génération aléatoire des mini batches
    mini_batches=np.split(np.random.permutation(np.arange(0,N)),nb_batches)
    L=np.arange(0,nb_batches)
    acccuracy_valid_croisee=[]
    for i in range(nb_batches):
        L_indices=np.delete(L,np.where(L==i))
        batch_test=mini_batches[i]
        batch_app=np.ravel(np.array(mini_batches)[L_indices])
        X_app,Y_app=X[batch_app],Y[batch_app]
        X_test,Y_test=X[batch_test],Y[batch_test]
        Dist=kppv_distances(X_test, X_app)
        l_k=range(1,k_max,2)
        l_accuracy=[]
        for k in l_k:
            Y_pred=kppv_predict(Dist, Y_app, k)
            kppv_res=evaluation_classifieur(Y_test, Y_pred)
            l_accuracy.append(kppv_res)
        acccuracy_valid_croisee.append(l_accuracy)
    
    acccuracy_valid_croisee=np.mean(acccuracy_valid_croisee,axis=0)
    plt.figure()
    plt.plot(l_k,acccuracy_valid_croisee)
    plt.xlabel('k voisins')
    plt.ylabel('Accuracy (%)')
    plt.title("Influence k avec validation croisee pour "+str(int(nb_batches))+" répertoires")
    
    # plt.savefig(fname='influence_k_avec_valid_croisee.png',format='png')
    # plt.savefig(fname='influence_k_avec_valid_croisee.svg',format='svg')



## Lancement des tests
# try:
#     l_k,l_accuracy=influence_param_k(path, 100)
# except:
#     print("Erreur dans param k")
# try:
#     l_k_lbp,l_accuracy_lbp=influence_param_k_LBP(path, 100)
# except:
#     print("Erreur dans LBP")
# try:
#     l_k_hog,l_accuracy_hog=influence_param_k_HOG(path, 100)
# except:
#     print("Erreur dans HOG")
# try:
#     influence_param_k_avec_validation_croisee(path, 200, 5) # repartition 80-20
# except:
#     print("Erreur dans validation croisée")

## Quelques tracés
# plt.figure()
# plt.plot(l_k,l_accuracy,label="Sans LBP")
# plt.plot(l_k_lbp,l_accuracy_lbp,label="Avec LBP")
# plt.xlabel('k voisins')
# plt.ylabel('Accuracy (%)')
# plt.title("Utilisation du descripteur LBP")
# plt.legend()
# plt.savefig(fname='influence_k_avec_lbp_results_2.png',format='png')
# plt.figure()
# plt.plot(l_k,l_accuracy,label="Sans HOG")
# plt.plot(l_k_hog,l_accuracy_hog,label="Avec HOG")
# plt.xlabel('k voisins')
# plt.ylabel('Accuracy (%)')
# plt.legend
# plt.title("Utilisation du descripteur HOG")
# plt.savefig(fname='influence_k_avec_hog_results_2.png',format='png')
