import cv2
import os
import label_pickle as lp
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def TakeImages(Id,name):
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    sampleNum=0
    img_dir="testImage/"+name
    try:
        os.rmdir(img_dir)
    except:
    	pass
    os.mkdir(img_dir)
    while(True):
        ret, img = cam.read()
        gray=img
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
            #incrementing sample number 
            sampleNum=sampleNum+1
            #saving the captured face in the dataset folder TrainingImage
            cv2.imwrite(img_dir+"/"+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
            # print("CAPTURES!!!!!!!!!!!!!!!")
            #display the frame
            cv2.imshow('frame',img)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 100
        elif sampleNum>30:
            break
    cam.release()
    cv2.destroyAllWindows()
    imgPath="testImage/features/"+name 
    lp.generatePickle(imgPath)
def take_input():
	TakeImages(178,"priyank")