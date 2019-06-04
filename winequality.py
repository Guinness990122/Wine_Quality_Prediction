import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree, svm, linear_model, neighbors
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering,KMeans
import  warnings
warnings.filterwarnings('ignore')


number = 0
data = np.genfromtxt(
    './winequality-red.csv',
    dtype=np.float32,
    delimiter=';',
    skip_header=1
)

X = data[:, 0:11]
Y = data[:, 11]


classifier1 = tree.DecisionTreeClassifier(random_state=0)
classifier2 = svm.SVC(gamma='auto', random_state=0)
classifier3 = linear_model.LogisticRegression(multi_class='auto', solver='liblinear', random_state=0)
classifier4 = neighbors.KNeighborsClassifier(n_neighbors=5)

def Accuracy(classifier):

    accuracy = cross_val_score(classifier,X,Y,cv=5).mean()
    accuracy_score = round(accuracy*100,1)

    return accuracy_score


def precision(classifier):
    y_true = data[:, 11]
    y_pred = []
    classifier = classifier.fit(X, Y)
    y_pred = classifier.predict(X)
    precision = precision_score(y_true, y_pred, average=None)

    return precision

def recall(classifier):
    y_true = data[:, 11]
    y_pred = []
    classifier = classifier.fit(X, Y)
    y_pred = classifier.predict(X)
    recall = recall_score(y_true, y_pred, average=None)

    return recall

def confusion(classifier):

    y_true = data[:, 11]
    y_pred = []
    classifier = classifier.fit(X, Y)
    y_pred=classifier.predict(X)
    confusion = confusion_matrix(y_true, y_pred)
    return confusion


def predict(input_, classifier):
    classifier = classifier.fit(X, Y)
    predicted_class = classifier.predict([input_])
    return predicted_class[0]

def hierarchical(cluster_amount,wine_num):
    wine_arr = np.array(X)
    model = AgglomerativeClustering(n_clusters = cluster_amount)
    model.fit(wine_arr)
    result=model.labels_[wine_num]
    return result

def K_means(cluster_amount,wine_num):
    wine_arr = np.array(X)
    model = KMeans(n_clusters = cluster_amount, random_state=0)
    model.fit(wine_arr)
    result=model.labels_[wine_num]
    return result

while number != 5:
    print(
        '''[Wine Quality]
        [name: Guinness]

        1. Evaluate classifiers 
        2. Input the information about a wine 
        3. Predict wine quality
        4. Cluster wines
        5. Quit '''
    )
    number = int(input())
    if number == 1:
        print('[Accuracy estimation]')
        print("Decision tree:", Accuracy(classifier1), "%")
        print("Support vector machine:", Accuracy(classifier2), "%")
        print("linear:", Accuracy(classifier3), "%")
        print("Knn:", Accuracy(classifier4), "%")
        print('\n ')
        print('[Confusion Matrix]')
        print("1. Decision tree:", '\n ',confusion(classifier1))
        print("2. Support vector machine:",'\n ', confusion(classifier2))
        print("3. Logistic:",'\n ', confusion(classifier3))
        print("4. Knn:",'\n ', confusion(classifier4))
        print('\n ')
        print('[Precision]')
        print("1. Decision tree:", precision(classifier1))
        print("2. Support vector machine:", precision(classifier2))
        print("3.Logistic:", precision(classifier3))
        print("4. Knn:", precision(classifier4))
        print('\n ')
        print('[Recall]')
        print("1. Decision tree:", recall(classifier1))
        print("2. Support vector machine:", recall(classifier2))
        print("3. Logistic:", recall(classifier3))
        print("4. Knn:", recall(classifier4))
        temp = input()

    elif number == 2:
        print('[Wine information]')
        input_data = []
        header = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH",
                  "sulphates", "alcohol"]
        length = len(header)
        for i in range(length):
            input_data.append(float(input("{}.{}: ".format(i + 1, header[i]))))

        temp = input()

    elif number == 3:
        print("[Predicted wine quality]")
        print("Decision tree:", predict(input_data, classifier1))
        print("Support vector machine:", predict(input_data, classifier2))
        print("Logistic regression:", predict(input_data, classifier3))
        print("K-NN classifier: ", predict(input_data, classifier4))
        temp = input()

    elif number == 4:
        cluster_input=input("Select the algorithm ((h)ierarchical or (k)-means):")
        cluster_amount=int(input("Input the number of cluseters:"))
        cluster_firstwine = int(input("Input the number of first wine:"))
        cluster_secondwine = int(input("Input the number of second wine:"))

        if cluster_input == 'h':
            first_result=hierarchical(cluster_amount,cluster_firstwine)
            second_result = hierarchical(cluster_amount,cluster_secondwine)

        elif cluster_input == 'k':
            first_result = K_means(cluster_amount, cluster_firstwine)
            second_result = K_means(cluster_amount, cluster_secondwine)

        if first_result == second_result:
            print("Result : ",cluster_firstwine,"and",cluster_secondwine," are in the same cluster")
        else:
            print("Result : ", cluster_firstwine, "and", cluster_secondwine, " are in the different cluster")

        temp=input()

