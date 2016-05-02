import math
import numpy as np
import cv2
import FaceRecognizer
reload(FaceRecognizer)



face_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./Cascades/smiled_01.xml')
nose_cascade = cv2.CascadeClassifier('./Cascades/Nariz.xml')

def selectCascades():
    global face_cascade, eye_cascade, mouth_cascade, nose_cascade



def processImage(pathToImageFile,trainingData):
    global face_cascade, eye_cascade, mouth_cascade, nose_cascade
    
    #Get the filename from the path - This will be the key for the DS storing the image features
    filename = pathToImageFile.split("/")[len(pathToImageFile.split("/"))-1].strip()
    print "File Name : ", filename
    
    # Selecting the image and other setup for cv2
    img = cv2.imread(pathToImageFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Input Image',img)
    # Identify the face in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print "Face is : ",faces
    # Capture the cascade data in detectMultiScale, then draw the rectangles
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # Identify the eye, mouth and nose in the image
        eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(30,30))
        mouth = mouth_cascade.detectMultiScale(roi_gray,minSize=(30,40))
        nose = nose_cascade.detectMultiScale(roi_gray,minSize=(60,60))
        print "Eyes are : ", eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)        
        
        for (ex,ey,ew,eh) in mouth:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            
        for (ex,ey,ew,eh) in nose:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(50,50,50),2)  
        
        # Store the Face, Eye, Nose and Mouth in their coressponding objects
        face     = FaceRecognizer.FR_Face(faces)
        leftEye  = FaceRecognizer.FR_Eye(eyes[1])
        rightEye = FaceRecognizer.FR_Eye(eyes[0])
        nose     = FaceRecognizer.FR_Nose(nose)
        mouth    = FaceRecognizer.FR_Mouth(mouth)
        
        # Printing Debug information
        print "Leye Upper Left:        ", leftEye.upperLeftPoint
        print "Leye Upper Right:        ", leftEye.upperRightPoint
        print "Leye Lower Left:        ", leftEye.lowerLeftPoint
        print "Leye Lower Right:        ", leftEye.lowerRightPoint                        

        print "Reye Upper Left:        ", rightEye.upperLeftPoint
        print "Reye Upper Right:        ", rightEye.upperRightPoint
        print "Reye Lower Left:        ", rightEye.lowerLeftPoint
        print "Reye Lower Right:        ", rightEye.lowerRightPoint                        

        # Calculating features
        distLeftRightEyeCenter  = calculateDistance(leftEye.center, rightEye.center)
        distLefEyeNoseCenter    = calculateDistance(leftEye.center, nose.center)
        distRightEyeNoseCenter  = calculateDistance(rightEye.center, nose.center)
        distLefEyeMouthCenter   = calculateDistance(leftEye.center, mouth.center)
        distRightEyeMouthCenter = calculateDistance(rightEye.center, mouth.center)
        print "Distance : ", distLeftRightEyeCenter
        #Storing features in object of class ImageFeatureSet
        trainingData.addToTrainingDataSet(filename,leftEye.center,rightEye.center,nose.center,mouth.center,distLeftRightEyeCenter,distLefEyeNoseCenter,distRightEyeNoseCenter,distLefEyeMouthCenter,distRightEyeMouthCenter,face.width)
    
        # Printing details from the object
        trainingData.printDataSet("Training Data")
        
        
        
        cv2.imshow("Image with identification", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return trainingData.trainingDataSet,trainingData.testDataSet
   
def calculateDistance(point1, point2, method="euclidean"):
    print "Inside Dis function.\nP1 : ",point1,type(point1)
    if method == "euclidean":
            return ( math.sqrt( pow( (point1[0] - point2[0]),2 ) + pow( (point1[1] - point2[1]),2 )) )
            
def scoring(train,test):
    
    
    #Sample testing data delete later
    test = {"testing": ((48.0, 60.0), (93.0, 59.0), (84.0, 88.5), (82.0, 105.0), 45.0111097397076, 45.9156835950419, 30.84234102658227, 56.4003546088143, 47.29693436154187, 49.0)}
    
    #The dictionary will store the confidence score for a particular test image when trained by a particular image
    confidence_score ={}
    
    
    #Iterating through each test data and calculating the confidence score for each data in the training set
    for key_test,value_test in test.iteritems():  
                
                for key_train,value_train in train.iteritems():
                    confi={}
                    i=0;
                    #confident increments when there is a matching attribute 
                    confident=0;
                    #The number of matching features gives a count of the number of features which are matched

                    number_of_matching=0;
                    
                    
                    for y in value_train:
                        if(i<4):
                            
                            # Matching each feature and calculating the confidence score 
                            
                            if(y[0]==value_test[i][0] and y[1]==value_test[i][1]) :
                                    confident = confident+10
                                    number_of_matching=number_of_matching+1
                            i=i+1
                        else:
                            if(y==value_test[i]):
                                confident = confident+10
                                number_of_matching=number_of_matching+1

                            i=i+1
                            
                            
                            
                    confi[key_train] =[confident,number_of_matching]
                    confidence_score[key_test]=confi
                 
                       
    return confidence_score
 
#Printing out the output to a text file output.txt            
def print_output_file(output):
    
    with open ("output.txt","a") as fh:
                for x,y in output.iteritems():
                    for h,w in y.iteritems():
                        out="Testing Image-->   %s   Training Image-->   %s   Confidence Score-->    %d    Number of Matching attributes-->   %d\n" % (x,h,w[0],w[1])
                        fh.write(out)             
    
                        
                        
                        
    
def main():
    
    selectCascades()
    
    trainingData = FaceRecognizer.ImageFeatureSet()
    # Call the below method in a loop for every file in the training data set
    train,test=processImage('./Images/test.jpg',trainingData)
    
    #Scoring the test data with the training dataset
    score=scoring(train,test)  
    
    #Printing to the output file
    print_output_file(score)

                        
if __name__ == '__main__':
    main()
