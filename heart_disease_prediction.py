import pandas as pd
from keras.layers import Dense
from keras.models import Sequential


"""reading the data frame through path"""

df= pd.read_csv("C:\\Users\\kanni\\OneDrive\\Documents\\verzeo\\data set\\heart.csv")

"""assigning the target value to y """
y=df["target"]
y=pd.get_dummies(y)

"""assigning the input values to x"""
x=df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak"]]

"""specifing the architecture"""
model= Sequential()
model.add(Dense(100,activation="relu",input_shape=(10,)))
model.add(Dense(2,activation="softmax"))
"""compiling the model"""
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

"""training the model"""
model.fit(x,y,epochs=20)

"""prediction using the model"""
data = pd.DataFrame({"age":[63],"sex":[1],"cp":[3],"trestbps":[145],"chol":[233],"fbs":[1],"restecg":[0],"thalach":[150],"exang":[0],"oldpeak":[2.3]})
result=model.predict(data)
print(result)




