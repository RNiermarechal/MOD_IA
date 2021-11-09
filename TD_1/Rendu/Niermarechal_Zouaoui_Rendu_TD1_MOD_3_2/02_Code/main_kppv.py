import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern,hog
from skimage import data, exposure


path='D:/Robin Niermaréchal/Documents/ECL/3A/S9/MOD/IA/MOD_IA/TD_1/cifar-10-batches-py/'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

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
    print("Extraction des données CIFAR effectuée")
    return X,Y

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

def kppv_distances(X_test,X_app):
    X_test_dots=(X_test*X_test).sum(axis=1).reshape(X_test.shape[0],1)*np.ones(shape=(1,X_app.shape[0]))    
    X_app_dots=(X_app*X_app).sum(axis=1)*np.ones(shape=(X_test.shape[0],1))
    Dist=X_test_dots+X_app_dots-2*X_test.dot(X_app.T)
    return Dist

def kppv_predict(Dist,Y_app,K):
    A=np.argsort(Dist,axis=1)[:,:K] # les positions des K plus proches voisins de la matrice de distance
    B=np.take(Y_app,A) # les classes de ces K plus proches voisins
    C=np.apply_along_axis(np.bincount,1,B,minlength=10) # effectifs pour chaque classe dans les K voisins, triés par numéro de classe et non par distance
    Y_pred=np.array([])
    for i in range(np.shape(B)[0]):
        l_class=[]
        for j in B[i]:
            if j not in l_class:
                l_class.append(j)
        class_reco=np.column_stack((l_class,np.zeros(len(l_class),dtype=int))) # les classes reconnues, dans l'ordre de proximité, avec compteurs associés (nuls pour le moment)
        final_count=np.copy(class_reco) #une copie pour pouvoir modifier les compteurs par la suite
        for k,line in enumerate(class_reco):
            final_count[k,1]=C[i,line[0]] # calcul du nb d'apparition pour chaque classe reconnue
        Y_pred=np.append(Y_pred,final_count[np.argmax(final_count[:,1]),0]) # on retient la classe avec le + grand nb d'apparitions, et la plus proche si égalité
    return np.transpose(np.array([Y_pred]))

def evaluation_classifieur(Y_test,Y_pred):
    good_pred=np.sum(np.equal(Y_test,Y_pred))
    return good_pred/len(Y_test)*100

def conv_2D_RGB(image_1D):
    image_2D_RGB=np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            image_2D_RGB[i,j,0]=image_1D[32*i + j]
            image_2D_RGB[i,j,1]=image_1D[1024 + 32*i + j]
            image_2D_RGB[i,j,2]=image_1D[2048 + 32*i + j] # image RGB en 2D, avec RGB selon l'axe 3 pour appliquer la méthode rgb2gray
    return image_2D_RGB

def add_LPB(X):
    images_LBP = [] # liste à retourner
    for x in X:
        image_2D_RGB=conv_2D_RGB(x) # passage format 1D à 2D
        image_2D_grey=rgb2gray(image_2D_RGB) # Transformation des images en niveaux de gris
        image_2D_LPB=local_binary_pattern(image_2D_grey,8,2,method='uniform') # application descripteur
        ## Affichge si besoin
        # plt.figure()
        # plt.imshow(image_2D_LPB, cmap='gray')
        # plt.show()
        ## Retour au format CIFAR
        image_LBP_flat=np.ravel(image_2D_LPB)
        images_LBP.append(image_LBP_flat)
    print("Descripteur LBP appliqué")
    return np.array(images_LBP)

def add_HOG(X):
    images_HOG = [] # liste à retourner
    for x in X:
        image_2D_RGB=conv_2D_RGB(x) # passage format 1D à 2D
        fd,image_2D_HOG = hog(image_2D_RGB, orientations=6, visualize=True) # application descripteur
                
        ## Retour au format CIFAR
        image_HOG_flat=np.ravel(image_2D_RGB)
        images_HOG.append(image_HOG_flat)
    print("Descripteur HOG appliqué")
    return np.array(images_HOG)

