import cv2
import pickle
import os



train_data_path = '../data/train_data/pose_recog_train_data.pk1'
path = '../data/'
train_data_folders = ['walk' , 'run' , 'jump' , 'side' , 'bend' , 'wave1' , 'wave2' , 'pjump' , 'jack' , 'skip']

def GMM_zivkovic(path , show = False):
    if not os.path.exists(path):
        print('function : GMM_zivkovic .' + path + ' Does not Exist')
        return
    cap = cv2.VideoCapture(path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    framerate = cap.get(5)
    processed_frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            processed_frames.append(fgmask)
            if show:
                cv2.imshow('frame' , fgmask)
        else:
            break
        
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return processed_frames , framerate
        
    

def createBMI(frames , framerate , function = 'medianBlur'):
    timecounter = 0
    delta_t = 1  / (framerate  )
    BMI = 0
    for frame in frames :
        BMI += timecounter * timecounter * frame
        cv2.imshow('' , BMI)
        cv2.waitKey(100)
        timecounter += delta_t
    # blurr the image (Normalizing to remove noise)
    
    if function == 'medianBlur':
        BMI = cv2.medianBlur(BMI,5)
    elif function == 'GaussianBlur':
        BMI = cv2.GaussianBlur(BMI , (5,5) , 0)
    return BMI


def save_bmi():
    inp = open(train_data_path , 'rb')
    [train_bmi,train_labels] = pickle.load(inp)
    inp.close()
    train_dir_paths = []
    for dir in train_data_folders:
        dir_path = path + dir + '/'
        train_dir_paths.append(dir_path)
        i = 0
    for bmi in train_bmi:
        lab = train_labels[i] + str(i) + '.jpg'
        lab = path + lab
        cv2.imwrite(lab , bmi)
        i += 1



frames , framerate = GMM_zivkovic(path+'side/daria_side.avi' )

bmi = createBMI(frames , framerate )
#cv2.imwrite(path + 'side.jpg' , bmi)
#cv2.imshow('' ,bmi)
#cv2.waitKey(1000)

      
      
      
