import numpy as np
import pickle
#lodaing the saved model
loaded_model=pickle.load(open("C:/Users/Aryan Sinha/Desktop/Wine_pred/trained_model.sav","rb"))

input_data=[[7,0.32,0.34,1.3,0.042,20,69,0.9912,3.31,0.65,12]]

np_array=np.asarray(input_data)

np_array_reshape=np_array.reshape(-1,1)
prediction=loaded_model.predict(np_array)
print(prediction)