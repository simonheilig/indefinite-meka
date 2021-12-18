from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt
import pandas as pd

def generateSyntheticData():
    X1,y1 = make_blobs(n_samples=7500, n_features=2 , centers=3, cluster_std=1.9,)
    plt.scatter(X1[:,0],X1[:,1],c=y1)
    plt.show()

    X2,y2 = make_classification(n_samples=10000, n_features=15, n_informative=2, n_redundant=2, n_repeated=0, n_classes=3, n_clusters_per_class = 1, flip_y=0.01,class_sep=1.0, hypercube=True)
    plt.scatter(X2[:,0],X2[:,1],c=y2)
    plt.show()

    cols1 = ["x"+str(i) for i in range(1,3)]
    df1 = pd.DataFrame(data=X1,columns=cols1)
    df1['y']=y1
    df1.to_csv("synth1.csv",index=False, sep=';')

    cols2 = ["x"+str(i) for i in range(1,16)]
    df2 = pd.DataFrame(data=X2,columns=cols2)
    df2['Y']=y2
    df2.to_csv("synth2.csv",index=False, sep=';')

if __name__ == "__main__":
    generateSyntheticData()