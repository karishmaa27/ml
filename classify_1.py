from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# loading dataset
iris=datasets.load_iris()

#printing decription and features
# print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0],labels[0])

# training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)
pred = clf.predict([[31,1,1,1]])
print(pred)