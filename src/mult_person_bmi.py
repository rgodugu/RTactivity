import numpy as np
import cv2
import os
import imutils
from BGSubstraction import GMM_zivkovic
import pickle
path = '/home/ravi/python/ML/data/userinput/'
single_person_files = [path+'ashwin_walk.mp4' , path+'sandeep_run.mp4']
mult_person_files = [path+'walk_run.mp4' , path+'walk_walk.mp4']
mult_person_frm_data = [path+'walk_run.pk1' , path+'walk_walk.pk1']

def readVideo(path_index):
    file_path = mult_person_files[path_index]
    cap = cv2.VideoCapture(file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # show frame
        frame = imutils.resize(frame, width =500)
        cv2.imshow('video frame' , frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
def readImgFile(path_index):
    file_path = mult_person_files[path_index]
    output_path = mult_person_frm_data[path_index]
    
    frames, rate = GMM_zivkovic(file_path)
    frame_list = []
    for f in frames:
        f = imutils.resize(f, width =500)
        #f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        f = cv2.GaussianBlur(f, (5,5),0)
        f = cv2.medianBlur(f,5)
        #gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        ret,gray = cv2.threshold(f,127,255,0)
        
        frame_list.append(gray)
    #write to file
    with open(output_path , 'wb') as out:
        pickle.dump(frame_list , out , pickle.HIGHEST_PROTOCOL)

def getContours(frames):
    contour_list = []
    patch_list = []
    avg_h = 0
    avg_w = 0
    for gray in frames:
        gray = imutils.resize(gray , height=400)
        
        
        bgFrame = frames[0]
        for frame in frames[2:]:
            
            grayscale = cv2.GaussianBlur(frame, (21, 21), 0)
            frameDelta = cv2.absdiff(bgFrame, grayscale)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            (_ ,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts_toshow = []
            for c in cnts:
                if cv2.contourArea(c) < 2000:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                avg_h += h
                avg_w += w
                cnts_toshow.append(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 0), 2)

            if len(cnts_toshow) > 0:
                contour_list.append([cnts_toshow ])
                patch_list.extend(getSeperatedPatches(frame , cnts_toshow , bgFrame))

            #cv2.imshow("Security Feed", frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #getSeperatedImages(frame,cnts_toshow , bgFrame , True)
            
        avg_h = avg_h // len(patch_list)
        avg_w = avg_w // len(patch_list)
        for i in range(len(patch_list)):
            #patch_list[i] = imutils.resize(patch_list[i] , height = avg_h )
            print(patch_list[i].shape)

        # apply kmeans
        patch_list_input = np.float32(np.vstack(patch_list))
        
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #ret,label,center=cv2.kmeans(patch_list_input,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        #return patch_list_input   
        #cv2.imshow('' , gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        


            
def getSeperatedImages(frame , contours , bgFrame , show = False):
    frame_list = []
    for cnt in contours:
        newframe  = bgFrame.copy()
        x,y,w,h = cv2.boundingRect(cnt)
        newframe[y:(y+h) , x:(x+w)] = frame[y:(y+h) , x:(x+w)]
        if show:
            cv2.imshow("Seperated image", newframe)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        frame_list.append(newframe)
    return frame_list

def getSeperatedPatches(frame , contours , bgFrame):
    patch_list = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        patch = frame[y:(y+h) , x:(x+w)]
        patch_list.append(patch)
    return patch_list
    
    


def getFrames_frm_file(path_index):
    input_path = mult_person_frm_data[path_index]
    with open(input_path , 'rb') as inp:
        frames = pickle.load(inp)
    return frames

#readImgFile(1)
frames = getFrames_frm_file(1)
getContours(frames)


#readVideo(1)
    

    
    
