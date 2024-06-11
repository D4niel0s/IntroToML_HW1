from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import scipy

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def main():
    
    print("*** Please read the comment in main() ***")
    '''
    Note to checker:
    (a) is the function "classify(images, labels, query, k)"
    (b) is "computeAccuracy(n,k)" when run with n=1000,k=10
    (c) is "plotAccAsFuncOfK()"
    (d) is "plotAccAsFuncOfn()"
    '''



def plotAccAsFuncOfn():
    n = np.arange(100,5001, 100)
    accs = np.zeros(len(n))
    for i in range(len(n)):
        accs[i] = computeAccuracy(n[i],1)
    
    plt.plot(n,accs)
    plt.title("Accuracy of KNN as a function of training set size")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.show()

def plotAccAsFuncOfK():
    k = np.arange(1,101)
    accs = np.zeros(len(k))
    for i in range(len(k)):
        accs[i] = computeAccuracy(1000,k[i])
    
    plt.plot(k,accs)
    plt.title("Accuracy of KNN as a function of K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()




def computeAccuracy(n,k):
    correctCounter = 0
    for i in range(len(test)):
        if(classify(train[:n],train_labels[:n], test[i], k) == int(test_labels[i])):
            correctCounter += 1

    return correctCounter / float(len(test))

def classify(images, labels, query, k):
    numImages = len(images)
    dists = np.zeros(numImages)

    for i in range(numImages):
        dists[i] = scipy.spatial.distance.euclidean(query,images[i])
    
    best = np.argpartition(dists,k)[:k]

    classCounters = np.zeros(10)
    #Now best has k closest images' indices
    for i in best:
        classCounters[int(labels[i])] += 1

    return int(classCounters.argmax())




if __name__ == '__main__':
    main()