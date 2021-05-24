import cv2

config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="frozen_inference_graph.pb"

model=cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize((320,320))
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
classlabels=[]
file_name="labels.txt"
with open(file_name,'r') as t:
    classlabels=t.read().rstrip('\n').split('\n')
print(classlabels)
video=cv2.VideoCapture(0)
font_scale=2
font=cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame=video.read()
    classindex,confidence,bbox=model.detect(frame,confThreshold=0.55)

    if len(classindex)!=0:
        for classind,conf,boxes in zip(classindex.flatten(),confidence.flatten(),bbox):
            if classind<=80:
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classlabels[classind-1],(boxes[0]+10,boxes[1]+30),font,fontScale=font_scale,color=(0,255,0))
                cv2.putText(frame, str(round(conf*100,2))+"%", (boxes[0] + 150, boxes[1] + 30), cv2.FONT_HERSHEY_COMPLEX,1,
                            color=(0, 255, 0))
    cv2.imshow("Object detection",frame)

    if ord('d')==0xff & cv2.waitKey(2):
        break
video.release()
cv2.destroyAllWindows()