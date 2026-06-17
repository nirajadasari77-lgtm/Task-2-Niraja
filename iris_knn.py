from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data[:5])
X = iris.data
y = iris.target
print(y[:5])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print(accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)
from sklearn.metrics import f1_score

f1 = f1_score(y_test, predictions, average='weighted')

print(f1)