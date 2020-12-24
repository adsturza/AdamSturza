import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

# Prints description of dataset
#print(cancer.DESCR) 

def createDataframe():

    columns = np.append(cancer.feature_names, 'target');   
    index = pd.RangeIndex(start=0, stop=569, step=1)
    
    data = np.column_stack((cancer.data, cancer.target))

    df = pd.DataFrame(data=data, index=index, columns=columns)
    
    return df 

def distribution(dataf):
    
    #values = dataf['target']
    #values = values.values_count()
    values = dataf.target.value_counts()
    a, b = values.iloc[0], values.iloc[1]
    
    values.iloc[0] = b
    values.iloc[1] = a
    
    indexes = ['malignant', 'benign']
    values.index = indexes
    target = pd.Series(values)

    return target

def trainTestSplit(dataf):

    copy = dataf.copy()
    X = copy.iloc[:, 0:30]
    y = copy['target']
    
    #print(X.shape, y.shape)
    #X, a pandas DataFrame, has shape (569, 30)
    #y, a pandas Series, has shape (569,)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    return X_train, X_test, y_train, y_test

def createClassifier(X_train, X_test, y_train, y_test):

    KNN = KNeighborsClassifier(n_neighbors = 1)
    return KNN.fit(X_train, y_train)

def predictAndScore(X_train, X_test, y_train, y_test, knn):

    predArr = knn.predict(X_test)
    #print(predArr)

    return knn.score(X_test, y_test)

def visualize_accuracy(X_train, X_test, y_train, y_test, knn):

    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

    plt.figure()

    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8)
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)

    return None

def main():
    dataf = createDataframe()
    #print(dataf)

    dis = distribution(dataf)
    #print(dis)

    X_train, X_test, y_train, y_test = trainTestSplit(dataf)
    knn = createClassifier(X_train, X_test, y_train, y_test)
    print(f"Out Of Sample Accuracy/Score: {predictAndScore(X_train, X_test, y_train, y_test, knn)}")

    visualize_accuracy(X_train, X_test, y_train, y_test, knn)

if __name__ == '__main__':
    main()
    




    

