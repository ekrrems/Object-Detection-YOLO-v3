import numpy as np
import cv2
import time

img = cv2.imread('woman-working-in-the-office.jpg')

print(img.shape)
h,w = img.shape[0:2]

# cv2.imshow('img',img)
# cv2.waitKey()

blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)

with open('yolo-coco-data\coco.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolo-coco-data\yolov3.cfg',
                                     'yolo-coco-data\yolov3.weights')
layer_names_all = network.getLayerNames()
# print(layer_names_all)
output_layers = ['yolo_82','yolo_94','yolo_106']
prob_min = 0.5
threshold =0.3
colors = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

network.setInput(blob)
start=time.time()
output_from_network = network.forward(output_layers)
end = time.time()
print(end-start)

#Bounding Boxes
bounding_boxes = []
confidences = []
class_numbers=[]

for result in output_from_network:
    print('result')
    print(len(result))
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]

        if confidence_current > prob_min:
            print('detected Objects:')
            print(detected_objects[0:8])
            box_current = detected_objects[0:4]*np.array([w,h,w,h])

            x_center,y_center,box_width,box_height = box_current
            x_min = int(x_center-(box_width/2))
            y_min = int(y_center-(box_height/2))

            bounding_boxes.append([x_min,y_min,int(box_width),int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

results = cv2.dnn.NMSBoxes(bounding_boxes,confidences,prob_min,threshold)
print('results')
print(results)
counter = 1

if len(results)>0:
    for i in results.flatten():
        print('Object{}: {}'.format(counter,labels[int(class_numbers[i])]))
        counter+=1
        x_min,y_min = bounding_boxes[i][0],bounding_boxes[i][1]
        box_width,box_height = bounding_boxes[i][2],bounding_boxes[i][3]

        color_box_current = colors[class_numbers[i]].tolist()

        cv2.rectangle(img,(x_min,y_min),(x_min+box_width,y_min+box_height),color_box_current,2)
        text_box_current = '{} : {}'.format(labels[int(class_numbers[i])],confidences[i])

        cv2.putText(img,text_box_current,(x_min,y_min-5),cv2.FONT_HERSHEY_COMPLEX,0.6,color_box_current,2)
print('bounding boxes :{}'.format(len(bounding_boxes)))
print('after nmsboxes :{}'.format(counter-1))
cv2.imshow('img',img)
cv2.waitKey()
