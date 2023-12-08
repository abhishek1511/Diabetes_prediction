import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

inputdata = (2,	122, 70, 27, 0, 36.8,	0.34,	27)
#convert the input data to numpy array
inputdata_as_nparray = np.asarray(inputdata)

#reshaping the array as we are predciting for 1 instance
input_data_reshape = inputdata_as_nparray.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction[0])

if (prediction[0] == 0):
  print("The patience is not diabetic")
else:
  print("The patience is diabetic")