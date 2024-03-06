import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import cv2

achulu_files = os.listdir(os.getcwd()+"/achulu")
#print(achulu_files)
'''
reading the matrices
'''

A = np.zeros((1024,len(achulu_files)))
for i in range (len(achulu_files)):
	img = cv2.imread("./achulu/"+achulu_files[i])
	if np.array(img).shape != ():
		resized = cv2.resize(img,(32,32))
		np_array = np.array(resized)
	#convert it to gray
		gray = np_array[:,:,0] * 0.3 + np_array[:,:,1]*0.6 + np_array[:,:,2]*0.1
		A[:,i] = gray.reshape([-1])

print("achulu")
print(A)
print(A.shape)

hallulu_files = os.listdir(os.getcwd()+"/hallulu/")
B = np.zeros((1024,len(hallulu_files)))
for j in range(len(hallulu_files)):
	img = cv2.imread("./hallulu/"+hallulu_files[j])
	if np.array(img).shape != ():
		resized_img = cv2.resize(img,(32,32))
		array = np.array(resized_img)
		gray_img = array[:,:,0]*0.3 + array[:,:,1]*0.6 + array[:,:,2]*0.1
		B[:,j] = gray_img.reshape([-1])
print("hallulu")
print(B)
print(B.shape)

#putting their transposes together and adding a column of labels
x = np.zeros((len(achulu_files)+len(hallulu_files),1024))
x[:len(achulu_files),:] = A.T
x[len(achulu_files):,:] = B.T

y = np.zeros((len(achulu_files)+len(hallulu_files),1))
y[:len(achulu_files)] = 0
y[len(achulu_files):] = 1

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4, random_state = 26)
print("checking number of 1s and 0s and y_test")
ones = 0;zeros=0
for i in y_test:
	if i == 1:
		ones+=1
	else:
		zeros+=1
print(f"ones : {ones} && zeros :{zeros}")
#Linear Regression
linear_classifier = linear_model.LinearRegression()
linear_classifier.fit(x_train,y_train)
predictions = linear_classifier.predict(x_test)
print("the accuracy score of this linear model is ",accuracy_score(y_test,np.round_(predictions)))
print("The mean squared error : %.2f" % mean_squared_error(y_test,predictions))
