from keras.engine.saving import model_from_json
import cv2


json_file = open("model.json",'r').read()


model = model_from_json(json_file)
model.load_weights('model_weights.h5')

#compile model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#predicting


image = cv2.imread('/Users/user/Desktop/nokia.jpeg')

image = cv2.resize(image,(64,64))

image = image.reshape(1,64,64,3)

result =  model.predict_classes(image)

if result[0,0]==0:
    print("nokia")
else:
    print("samsung")
