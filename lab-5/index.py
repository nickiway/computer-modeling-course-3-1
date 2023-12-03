import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# initial data
file_path = './Wine.csv'
data = pd.read_csv(file_path)
data_without_wine = data.drop('Wine', axis=1) 
np.random.seed(44)

print("Task 1 -----------------------------------------------------------------")
print(data.describe())

total_objects = len(data)
print(f"Total number of objects in the sample: {total_objects}")

classification_features = data.columns[1:]
num_classification_features = len(classification_features)

print(f"Number of features for classification: {num_classification_features}")
print(f"Features for classification: {', '.join(classification_features)}")

missing_values = data.isnull().sum().sum()
if missing_values == 0:
    print("No missing values found")
else:
    data = data.apply(lambda col: col.fillna(col.mean()), axis=0) # setting mean in case we have null
    print(f"Number of missing values: {missing_values}")

duplicates = data.duplicated().sum()
if duplicates == 0:
    print("No duplicate records found")
else:
    print(f"Number of duplicate records: {duplicates}")
    data.drop_duplicates(inplace=True) # removing duplicates

# Number of instances in each class
class_counts = data.iloc[:, 0].value_counts()
print("Number of instances in each class:")
print(class_counts)


# Task 2 -----------------------------------------------------------------
font_size = 6
xticks_orientation = 'vertical'  
yticks_orientation = 'horizontal'

color_wheel={1:"red",  2:"blue", 3:"green"}
colors = data["Wine"].map(lambda x: color_wheel.get(x))

scatter_matrix(data_without_wine, color=colors)

for ax in plt.gcf().get_axes():
    ax.xaxis.label.set_rotation(xticks_orientation)
    ax.yaxis.label.set_rotation(yticks_orientation)
    ax.yaxis.label.set_ha('right') 
    ax.xaxis.label.set_fontsize(font_size)
    ax.yaxis.label.set_fontsize(font_size)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(font_size)
        
plt.show()

print("Task 3 -----------------------------------------------------------------")
clas = data.iloc[:, 0].values
data = data.iloc[:, 1:]

Size_train, Size_test, Class_train, Class_test = train_test_split(data,clas,test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=7, metric='manhattan')
classifier.fit(Size_train, Class_train)

test_predict = classifier.predict(Size_test)
score_bayes = classifier.score(Size_test, Class_test)

print("Classification results before scaling, K-Nearest Neighbors (KNN) method:")
print("Accuracy score = ", score_bayes, "\n")

print("Task 4 -----------------------------------------------------------------")
print("Classifier performance report:")
print(classification_report(Class_test, test_predict),"\n")

# Task 5 -----------------------------------------------------------------
cm = confusion_matrix(Class_test, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()

plt.title('Confusion Matrix, K-Nearest Neighbors (KNN) without scaling 7 neighb')
plt.show()

print("Task 6 -----------------------------------------------------------------")
Size_train, Size_test, Class_train, Class_test = train_test_split(data,clas,test_size=0.2)

scaler = StandardScaler()
scaler.fit(Size_train)

Size_train = scaler.transform(Size_train)
Size_test = scaler.transform(Size_test)

classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
classifier.fit(Size_train,Class_train)

test_predict = classifier.predict(Size_test)
score_bayes = classifier.score(Size_test, Class_test)

print("Classification results after scaling, K-Nearest Neighbors (KNN) method 7 neighb:")
print("Accuracy score = ", score_bayes, "\n")

print("Classifier performance report:")
print(classification_report(Class_test,test_predict),"\n")

cm2 = confusion_matrix(Class_test, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=classifier.classes_)
disp.plot()

plt.title('Confusion Matrix, K-Nearest Neighbors (KNN) with scaling 7 neighb')
plt.show()

print("Task 7 -----------------------------------------------------------------")
Size_train, Size_test, Class_train, Class_test = train_test_split(data,clas,test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=2, metric='euclidean')
classifier.fit(Size_train, Class_train)

test_predict = classifier.predict(Size_test)
score = classifier.score(Size_test,Class_test)

print("Accuracy score for Decision Tree method with max_depth=3, min_samples_split =2 = ", score, "\n")
print("Classifier performance report:")

print(classification_report(Class_test,test_predict),"\n")
unique_classes = sorted(set(Class_train) | set(Class_test))

cm = confusion_matrix(Class_test, test_predict, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot()

plt.title('Confusion Matrix, K-Nearest Neighbors (KNN) without scaling 2 neighb')
plt.show()

print("Task 9-----------------------------------------------------------------")
scaler = StandardScaler()
scaler.fit(Size_train)

Size_train = scaler.transform(Size_train)
Size_test = scaler.transform(Size_test)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def find_nearest_neighbors(test_data, train_data, train_labels):
    nearest_neighbors = []
    for test_instance in test_data:
        distances = [euclidean_distance(test_instance, train_instance) for train_instance in train_data]
        nearest_neighbor_index = np.argmin(distances)
        nearest_neighbor_label = train_labels[nearest_neighbor_index]
        nearest_neighbors.append(nearest_neighbor_label)
    return nearest_neighbors

selected_objects = []
selected_labels = []

for class_label in set(Class_test):
    class_indices = np.where(Class_test == class_label)[0][:3]
    selected_objects.extend(Size_test[class_indices])
    selected_labels.extend(Class_test[class_indices])

nearest_neighbors = find_nearest_neighbors(selected_objects, Size_train, Class_train)

for i, (test_label, nearest_neighbor) in enumerate(zip(selected_labels, nearest_neighbors)):
    print(f"Object {i+1} from class {test_label} has the nearest neighbor from class {nearest_neighbor}")
