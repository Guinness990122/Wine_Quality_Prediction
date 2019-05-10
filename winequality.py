import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree,svm,linear_model,neighbors
number = 0
input_data = []
data =  np.genfromtxt(
    './winequality-red.csv',
    dtype = np.float32,
    delimiter=';',
    skip_header=1
)

X = data[:, 0:11]
Y = data[:, 11]

test_sample = [[7,0.7,1,1,0.056,12,34,0.9978,3.21,0.66,9]]

classifier1 = tree.DecisionTreeClassifier(random_state=0)
classifier2 = svm.SVC(gamma='auto',random_state=0)
classifier3 = linear_model.LogisticRegression(multi_class='auto',solver='liblinear',random_state=0)
classifier4 = neighbors.KNeighborsClassifier(n_neighbors=5)

def Accuracy(classifier):

    accuracy = cross_val_score(classifier,X,Y,cv=5).mean()
    accuracy_score = round(accuracy*100,1)

    return accuracy_score


def predict(input_,classifier):
    classifier = classifier.fit(X,Y)
    predicted_class = classifier.predict([input_])
    return predicted_class[0]

def Input_Wine():
    header=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
            "free sulfur dioxide","total sulfur dioxide","density","pH",
            "sulphates","alcohol"]
    length=len(header)
    for i in range(length):
        input_data.append(float(input("{}.{}: ".format(i+1, header[i]))))

    return input_data

while number!=4:
    print(
        '''[Wine Quality]
        [name: Guinness]
        
        1.Estimate the accuracy of classifiers 
        2. Input the information about a wine 
        3. Predict wine quality
        4. Quit '''
    )
    number=int(input())
    if number == 1 :
        print('[Accuracy estimation]')
        print("Decision tree:", Accuracy(classifier1),"%")
        print("Support vector machine:", Accuracy(classifier2),"%")
        print("linear:", Accuracy(classifier3),"%")
        print("Knn:",Accuracy(classifier4),"%")
        temp = input()

    elif number == 2:
        print('[Wine information]')
        Input_Wine()
        temp = input()

    elif number == 3:
        print("[Predicted wine quality]")
        print("Decision tree:", predict(input_data,classifier1))
        print("Support vector machine:",predict(input_data, classifier2))
        print("Logistic regression:",predict(input_data, classifier3))
        print("K-NN classifier: ", predict(input_data, classifier4))
        temp = input()



