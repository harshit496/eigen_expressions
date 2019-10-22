
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import operator

#%%
def show_image_and_label(x, y):
    x_reshaped = x.reshape(48,48)
    plt.imshow(x_reshaped, cmap= "gray",
              interpolation="nearest")
#    plt.imshow(x_reshaped)
    plt.axis("off")
    plt.show()
    print(y)

#%%
raw_data_csv_file_name = 'fer2013.csv'
raw_data = pd.read_csv(raw_data_csv_file_name)
raw_data.info()
raw_data.head()
#%%

EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
i = 20
# x_pixels
img = raw_data["pixels"][i]
val = img.split(" ")
x_pixels = np.array(val, 'float32')
#print (x_pixels)
x_pixels /= 255
#print (x_pixels)
print (np.shape(x_pixels),len(raw_data.loc[raw_data['Usage'] == 'Training']))

show_image_and_label(x_pixels, EMOTIONS[raw_data["emotion"][i]])

#%%
dataset = pd.DataFrame(columns=['name','expression','index','pixels'])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#face_img = cv2.imread('jaffe/KA.AN1.39.tiff',0)
#gray = cv2.imread('jaffe/KA.AN1.39.tiff',0)
#
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
#    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = gray[y:y+h, x:x+w]
#plt.imshow(roi_gray)
for i,files in enumerate(os.listdir("jaffe")):
#    print (files)
    name,expression,index,extension = files.split('.')
    
#    if "HA" in expression:
#        dataset['expression'] = 'HAPPY'
#    elif 'AN' in expression:
#        dataset['expression'] = 'ANGRY'
#    elif 'DI' in expression:
#        dataset['expression'] = 'DISGUST'
#    elif 'SU' in expression:
#        dataset['expression'] = 'SURPRISED'
#    elif 'SA' in expression:
#        dataset['expression'] = 'SAD'
#    elif 'FE' in expression:
#        dataset['expression'] = 'FEAR'
#    elif 'NE' in expression:
#        dataset['expression'] = 'NEUTRAL'
    
    
    filename = 'jaffe/'+files
    gray = cv2.imread(filename,0)
    dim = (52,52)
    

#    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    for (x,y,w,h) in faces:
#        roi_gray = gray[y:y+h, x:x+w]
#    roi_gray = (roi_gray - np.mean(roi_gray))/np.std(roi_gray)
#    resized = cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
    #normalize for brightness and contrast
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    dataset.loc[i] = [name, expression,index,resized.ravel()]
#    plt.imshow(resized)
#    plt.show()
#%%
#msk = np.random.rand(len(dataset)) < 0.8
#train_data = dataset[msk]
#test_data = dataset[~msk]
train_data = dataset
#%%
#train_data = raw_data.loc[raw_data['Usage'] == 'Training']
vectors_rowsize = len(np.array(dataset['pixels'][1]))
vectors_colsize = len(train_data)
vectors = np.zeros((vectors_rowsize,vectors_colsize))
centered_vectors = np.zeros((vectors_rowsize,vectors_colsize))

for i,img in enumerate(train_data['pixels']):
#    x_pixels = np.array(img, 'float32')
    vectors[:,i] = img
#, 
average = (np.mean(vectors, axis = 1))
#%%

plt.imshow(average.reshape(dim),cmap= "gray",
              interpolation="nearest")
plt.show()

#%%
for i in range(np.shape(vectors)[1]):
    
    centered_vectors[:,i] = vectors[:,i] - average
  
#%%    
covariance_matrix = np.dot(centered_vectors,centered_vectors.T)

#%%
eigenvalues,eigenvectors = np.linalg.eigh(covariance_matrix.T)
indices = np.argsort(eigenvalues)
sortedeigenvectors = eigenvectors[:,indices]
#eigenvalues = np.sort(eigenvalues,axis = 1)
#print (covariance_matrix)

#%%
proj_eigenvector = sortedeigenvectors[:,1:2]
test_img = test_data.iloc[1]
print (test_img['expression'])
test_img = test_img["pixels"]
#test_pixels = np.array(test_img, 'float32')

centered = test_img - average
projection = np.dot(proj_eigenvector.T,average)

test_img = test_img.reshape(dim)
plt.imshow(test_img,cmap= "gray",
              interpolation="nearest")
plt.show()

#%%

centered_result = np.dot(proj_eigenvector,projection)
result = centered_result + average
result = result.reshape(dim)
plt.imshow(result,cmap= "gray",
              interpolation="nearest")
plt.show()

#%%
res_list = []
result = result.ravel()
for img,exp in zip(train_data['pixels'],train_data['expression']):
#    train = np.array(img, 'float32')
    res_list.append((exp,np.linalg.norm(result-img)))
res_list = sorted(res_list , key = operator.itemgetter(1))