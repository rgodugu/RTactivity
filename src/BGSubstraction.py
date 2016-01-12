import numpy as np
import os
import cv2
import pickle

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


def GMM_zivkovic_list(frame_list ):
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    processed_frames = []
    for frame in frame_list:
        fgmask = fgbg.apply(frame)
        processed_frames.append(fgmask)
        
    
    return processed_frames

def generate_BMI(frame_list , framerate):
    processed_frames = GMM_zivkovic_list(frame_list )
    BMI = createBMI(processed_frames , framerate)
    return BMI
    

def createBMI(frames , framerate ,polynomial_order = 2 , function = 'medianBlur' ):
    timecounter = 0
    delta_t = 1 / framerate
    BMI = 0
    for frame in frames :
        BMI += (timecounter ** polynomial_order ) * frame
        #BMI += timecounter * timecounter *timecounter * frame
        timecounter += delta_t
    # blurr the image (Normalizing to remove noise)
    
    if function == 'medianBlur':
        BMI = cv2.medianBlur(BMI,5)
    elif function == 'GaussianBlur':
        BMI = cv2.GaussianBlur(BMI , (5,5) , 0)
    return BMI

def process_training_data(poly_order = 2 , bg_substractor = 'GMM_zivkovic' , normalize = 'medianBlur'):
    if not os.path.exists(path):
        print(path + ' :path does not exist')
        return
    train_dir_paths = []
    for dir in train_data_folders:
        dir_path = path + dir + '/'
        if os.path.exists(dir_path) :
            print('adding ' + dir_path + ' to train directories')
            train_dir_paths.append(dir_path)
        else:
            print('path : ' + dir_path + 'does not exist')

    train_bmi = []
    train_bmi_labels = []
    # collect all *.avi files in each folder
    for dir_path in train_dir_paths:
        action = os.path.basename(os.path.normpath(dir_path))
        train_files = os.listdir(dir_path)
        for file in train_files:
            file_path = dir_path + file
            if bg_substractor == 'GMM_zivkovic':
                frames , framerate = GMM_zivkovic(file_path)

            bmi = createBMI(frames , framerate ,poly_order , normalize)

            # add bmi and label to lists
            train_bmi.append(bmi)
            train_bmi_labels.append(action)

    return train_bmi , train_bmi_labels
            
        

def main():
    train_bmi , train_lables = process_training_data(0)
    train_data_path = path + 'train_data/'
    file_to_store = train_data_path + 'pose_recog_train_data_0.pk1'
    
    if not os.path.exists(train_data_path):
        print('creating directory for training data @: '+ train_data_path)
        os.makedirs(train_data_path)
    print('saving data as pickle object')
    print('saving to file : ' + file_to_store)
    with open(file_to_store , 'wb') as output:
        pickle.dump([train_bmi , train_lables] , output , pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    main()



    
