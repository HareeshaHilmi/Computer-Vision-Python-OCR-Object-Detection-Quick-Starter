# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 07:55:13 2021

@author: user
"""


import numpy as  np
import cv2

# get the video stream from the video file
webcam_video_stream = cv2.VideoCapture('E:\\Practice\\OCR\\code\\images\\testing\\video_sample.mp4')

while True:
    ret,current_frame = webcam_video_stream.read()    
    
    # load the image to detect, get width, height 
    # resize to match input size, convert to blob to pass into model
    #img_to_detect = cv2.imread('E:\\Practice\\OCR\\code\\images\\testing\\scene1.jpg')
    img_to_detect = current_frame
    
    
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    
    # convert to blob to pass into model
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (608, 608), swapRB=True, crop=False)
    #recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
    #accepted sizes are 320×320,416×416,608×608. More size means more accuracy but less speed
    
    # set of 80 class labels 
    class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    
    #Declare List of colors as an array
    #Green, Blue, Red, cyan, yellow, purple
    #Split based on ',' and for every split, change type to int
    #convert that to a numpy array to apply color mask to the image numpy array
    class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(16,1))
    
    # Loading pretrained model 
    # input preprocessed blob into model and pass through the model
    # obtain the detection predictions by the model using forward() method
    yolo_model = cv2.dnn.readNetFromDarknet('E:\\Practice\\OCR\\code\\dataset\\yolov3.cfg','E:\\Practice\\OCR\\code\\dataset\\yolov3.weights')
    
    # Get all layers from the yolo network
    # Loop and find the last layer (output layer) of the yolo network 
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
    
    # input preprocessed blob into model and pass through the model
    yolo_model.setInput(img_blob)
    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = yolo_model.forward(yolo_output_layer)
    
    
    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
    	# loop over the detections
        for object_detection in object_detection_layer:
            
            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]
        
            # take only predictions with confidence more than 50%
            if prediction_confidence > 0.50:
                #get the predicted label
                predicted_class_label = class_labels[predicted_class_id]
                #obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                end_x_pt = start_x_pt + box_width
                end_y_pt = start_y_pt + box_height
                
                #get a random mask color from the numpy array of colors
                box_color = class_colors[predicted_class_id]
                
                #convert the color numpy array as a list and apply to text and box
                box_color = [int(c) for c in box_color]
                
                # print the prediction in console
                predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                print("predicted object {}".format(predicted_class_label))
                
                # draw rectangle and text in the image
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    cv2.imshow("Detection Output", img_to_detect)
    # terminate while loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing he stream and the camera
webcam_video_stream.release()
# close all opencv windows
cv2.destroyAllWindows()