import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

class YOLOv4():
    def __init__(self, conf_thresh = 0.4, nms_thresh = 0.4, input_size = (416,416), scale = 1.0/255):

    	# Initializing the parameters 
    	self.conf_thresh = conf_thresh
    	self.nms_thresh = nms_thresh
    	self.input_size = input_size
    	self.scale = scale

    	self.model = None
    	self.names = None 
    	self.layer = None

    	cmap = plt.cm.get_cmap("hsv", 256)
    	self.cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    def read_names(self, names_file):
    	file = open(names_file, "r")
    	names = [line.split('\n'[0]) for line in file.readlines()]
    	return names

    def load_model(self, weights_path, config_path, names_path):

    	self.model = cv2.dnn.readNet(weights_path, config_path)
    	self.names = self.read_names(names_path)
    	ln = self.model.getLayerNames()
    	self.layers = [ln[i-1] for i in self.model.getUnconnectedOutLayers()]

    def detect(self, image, write_class = False):

    	height, width = image.shape[:2]
    	blob = cv2.dnn.blobFromImage(image, self.scale, self.input_size, swapRB = True, crop = False)
    	self.model.setInput(blob)

    	output = self.model.forward(self.layers)
    	boxes = []
    	confidences = []
    	classes = []

    	for out in output:
    		for detection in out:
    			
    			scores = detection[5:]
    			classID = np.argmax(scores)
    			conf = score[classID]

    			if(conf>self.conf_thresh):

    				box = detection[0:4]*np.array((width,height,width,height))
    				cx,cy,w,h = ox.astype('int')
    				x = int(cx - w/2)
    				y = int(cy - h/2)
    				boxes.append([x,y,int(w), int(h)])
    				confidences.append(conf)
    				classes.append(classID)

    	index = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
    	detections = []

    	if len(idxs) > 0:
    		for i in idxs.flatten():

    			class_id = classes[i]
    			conf = confidences[i]
    			(x,y) = (boxes[i][0], boxes[i][1])
    			(w,h) = (boxes[i][2], boxes[i][3])
    			bbox = [x,y,w,h]

    			if True:

    				color = self.cmap[int(255.0 / (class_id + 1)), :]
    				cv2.rectangle(image, (x, y), (x + w, y + h), color=tuple(color), thickness=2)

    				if write_class:
    					cv2.putText(image, str(self.names[class_id]), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 2, 16)
    					cv2.putText(image, str(self.names[class_id]), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 1, 16)

    			detections.append([class_id, bbox, conf])

    	cv2.imshow("image", image)
    	cv2.waitKey()
    	return image, np.array(detections)



