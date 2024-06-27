#Load the dataset and predict the survival of passengers using Logistic Regression.
import pandas as PD
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#Load and preprocess data
dataframe = PD.read_csv('Titanic-Dataset.csv')
dataframe.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
dataframe['Age'].fillna(dataframe['Age'].median(), inplace=True)
dataframe['Embarked'].fillna(dataframe['Embarked'].mode()[0], inplace=True)
dataframe=PD.get_dummies(dataframe, columns=['Sex', 'Embarked'], drop_first=True)

#Split data
feature=dataframe.drop('Survived', axis=1)
target=dataframe['Survived']
X_train,X_test,Y_train,Y_test=TTS(feature,target,test_size=0.5,random_state=45)

#Scale features
SCALAR = SS()
X_train_scaled=SCALAR.fit_transform(X_train)
X_test_scaled=SCALAR.transform(X_test)

#Train and predict
Lr=LR(random_state=45)
Lr.fit(X_train_scaled,Y_train)
predict=Lr.predict(X_test_scaled)

#Evaluate
Acc=accuracy_score(Y_test,predict)
Con_Mat=confusion_matrix(Y_test,predict)
Class_Report=classification_report(Y_test,predict)

print('Accuracy: ' + str(round(accuracy, 4)))
print('Confusion Matrix:')
print(Con_Mat)
print('Classification Report:')
print(Class_Report)