import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern


path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/'

L_path=[]
for i in range(1,6):
    L_path.append('D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/data_batch_'+str(i))
    L_path.append('D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/test_batch')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

# dict=unpickle('D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/TD_1/cifar-10-batches-py/data_batch_1')
# labels=dict[b'labels']
# data=dict[b'data']

def lecture_cifar(file):
    L_path=[]
    for i in range(1,6):
        L_path.append(file+'data_batch_'+str(i))
        L_path.append(file+'test_batch')
    
    path=L_path[0]
    file_extracted=unpickle(path)
    Y=np.transpose(np.array([file_extracted[b'labels']]))
    X=np.array(file_extracted[b'data'],dtype='float32')
    for path in L_path[1:]:
        file_extracted=unpickle(path)
        y=np.transpose(np.array([file_extracted[b'labels']]))
        x=np.array(file_extracted[b'data'],dtype='float32')
        X=np.append(X,x,axis=0)
        Y=np.append(Y,y,axis=0)
    return X,Y

# X,Y=lecture_cifar(path)


def decoupage_donnees(X,Y):
    index_app=np.random.choice(len(Y),int(0.8*len(Y)),replace=False)
    index_test=[]
    # index_test=np.unique(np.array(index_app,np.arange(len(Y))))
    X_test=[]
    Y_test=[]
    X_app=[]
    Y_app=[]
    for x in range(len(Y)):
        if x not in index_app:
            index_test.append(x)
            X_test.append(X[x])
            Y_test.append(Y[x])
        else:
            X_app.append(X[x])
            Y_app.append(Y[x])

    return np.array(X_app),np.array(Y_app),np.array(X_test),np.array(Y_test)

# X_app,Y_app,X_test,Y_test=decoupage_donnees(X, Y)

def kppv_distances(X_test,X_app):
    X_test_dots=(X_test*X_test).sum(axis=1).reshape(X_test.shape[0],1)*np.ones(shape=(1,X_app.shape[0]))
    X_app_dots=(X_app*X_app).sum(axis=1)*np.ones(shape=(X_test.shape[0],1))
    Dist=X_test_dots+X_app_dots-2*X_test.dot(X_app.T)
    return Dist

# Dist=kppv_distances(X_test, X_app)
# print(Dist) 

def kppv_predict(Dist,Y_app,K):
    A=np.argsort(Dist,axis=1)[:,:K]
    B=np.take(Y_app,A)
    C=np.apply_along_axis(np.bincount,1,B,minlength=10) # effectifs pour chaque classe dans les K voisins
    Y_pred=np.array([])
    for i in range(np.shape(B)[0]):
        l_class=[]
        for j in B[i]:
            if j not in l_class:
                l_class.append(j)
        class_reco=np.column_stack((l_class,np.zeros(len(l_class),dtype=int))) # les classes reconnues, dans l'ordre de proximité, avec compteurs associés
        final_count=np.copy(class_reco) #une copie pour pouvoir modifier les compteurs
        
        for k,line in enumerate(class_reco):
            final_count[k,1]=C[i,line[0]]
        Y_pred=np.append(Y_pred,final_count[np.argmax(final_count[:,1]),0])
            
    return Y_pred

#Y_pred=kppv_predict(Dist, Y_app, 3)
# print(Y_pred)

def evaluation_classifieur(Y_test,Y_pred):
    good_pred=np.sum(np.equal(Y_test,Y_pred))
    return good_pred/len(Y_test)*100

#kppv_res=evaluation_classifieur(Y_test, Y_pred)

#print(kppv_res)


def influence_param_k(path,k_max):
    X,Y=lecture_cifar(path)
    X_app,Y_app,X_test,Y_test=decoupage_donnees(X, Y)
    Dist=kppv_distances(X_test, X_app)
    l_k=range(1,k_max)
    l_accuracy=[]
    for k in l_k:
        Y_pred=kppv_predict(Dist, Y_app, k)
        kppv_res=evaluation_classifieur(Y_test, Y_pred)
        l_accuracy.append(kppv_res)
    plt.figure()
    plt.plot(l_k,l_accuracy)
    plt.xlabel('k voisins')
    plt.ylabel('Accuracy (%)')
    plt.title(path.split('/')[-1])
    
    plt.savefig(fname=path.split('/')[-1]+'_results.png',format='png')
    plt.savefig(fname=path.split('/')[-1]+'_results.svg',format='svg')

# influence_param_k(path, 20)

#for path in L_path:
    #influence_param_k(path, 200)

def add_LPB(X):
    images_LPB = [] # liste à retourner
    # Transformer les images en noir et blanc
    for x in X:
        image_2D_grey=np.zeros((32,32))
        for i in range(32):
            for j in range(32):
                image_2D_grey[i,j]=0.2125*x[32*i + j]+0.7154*x[1*32**2 + 32*i + j]+0.0721*x[2*32**2 + 32*i + j] # image RGB en 2D, chaque triplet de couleur est un tuple pour pouvoir appliquer la méthode rgb2gray
        # plt.figure()
        # plt.imshow(image_2D_grey, cmap='gray')
        # plt.show()

        # descripteur LBP
        image_2D_LPB = local_binary_pattern(image_2D_grey,8,2,method='uniform')
        # plt.imshow(lbp_image, cmap='gray')
        # plt.show()

        #Reviens au format des images de la base de données cifar
        image_LBP_flat = image_2D_LPB[0]
        for i in range(1,len(image_2D_LPB)):
            image_LBP_flat = np.concatenate((image_LBP_flat,image_2D_LPB[i]), axis = 0)

        images_LPB.append(image_LBP_flat)
    return np.array(lbp_images)

X,Y=lecture_cifar(path)
add_LPB(X)