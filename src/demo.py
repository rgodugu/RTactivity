import cv2
import pickle
from sklearn.mixture import GMM

path = '/home/ravi/python/ML/src/contour.pk1'
with open(path ,'rb') as inp:
    contour_lis,movement_frame_list ,bgframe = pickle.load(inp)

#fgbg = cv2.createBackgroundSubtractorMOG2()


def readImage_frm_path():
    for p in plist:
        frame = cv2.imread(p)
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame' , fgmask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def readImages_frm_pk(lis):
    for frame in lis :
        cv2.imshow('frame' , frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def showimg(img):
    cv2.imshow('img' , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def build_duplicate_sequence(contour_lis , mv_frame_lis , bgframe):
    dup_seq = []
    for i in range(len(mv_frame_lis)):
        
        frame = bgframe.copy()
        x,y,w,h = contour_lis[i]
        x = x-30
        #w = w +30
        orig_frame = mv_frame_lis[i]
        patch = orig_frame[0:(0+375) , x:(x+w)]
        #showimg(patch)
        frame[0:(0+375) , x:(x+w)] = patch
        showimg(frame)
        
        dup_seq.append(frame)
    return dup_seq

def cluster_images():
    p = '/home/ravi/python/ML/data/walk/daria_walk.avi'
    cap = cv2.VideoCapture(p)
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        showimg(frame)
        
    
        
    
                        
                        
    
        


    
#readImages_frm_pk(movement_frame_list)
#dup = build_duplicate_sequence(contour_lis , movement_frame_list , bgframe)
cluster_images()
        


