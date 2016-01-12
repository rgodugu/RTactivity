import imutils
import cv2
import time
import pickle
from BGSubstraction import generate_BMI

min_area = 500


def start_video_cam():
    i = 0
    movement_frame_list = []
    camera = cv2.VideoCapture(0)
    print(camera.get(5))
    time.sleep(0.25)
    bgFrame = None

    while True:
        ret, frame = camera.read()
        text = 'Unoccupied'

        if not ret:
            break

        frame = imutils.resize(frame, width=500)
        frame_to_be_processed = frame.copy()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (21, 21), 0)

        if bgFrame is None:
            bgFrame = grayscale
            continue

        frameDelta = cv2.absdiff(bgFrame, grayscale)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        (_ ,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		

        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        
        if text == 'Occupied':
            print('adding a frame ')
            movement_frame_list.append(frame_to_be_processed)
        elif text == 'Unoccupied' and len(movement_frame_list) > 0:
            BMI = generate_BMI(movement_frame_list ,25)
            '''
            with open('demo.pk1' , 'wb') as out:
                pickle.dump([movement_frame_list , BMI] , out ,  pickle.HIGHEST_PROTOCOL)
            '''
            cv2.imwrite('wave1'+str(i)+'.jpg' , BMI)
            i += 1
            movement_frame_list = []
            
            #print( nn.predict(np.asarray([BMI])) )

        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

def start_video_cam2():
    movement_frame_list = []
    contour_list = []
    camera = cv2.VideoCapture(0)
    print(camera.get(5))
    time.sleep(0.25)
    bgFrame = None
    bgOrig = None

    while True:
        ret, frame = camera.read()
        text = 'Unoccupied'
        

        if not ret:
            break

        frame = imutils.resize(frame, width=500)
        if bgOrig is None:
            bgOrig = frame.copy()
        frame_to_be_processed = frame.copy()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (21, 21), 0)

        if bgFrame is None:
            bgFrame = grayscale
            continue

        frameDelta = cv2.absdiff(bgFrame, grayscale)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        (_ ,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		

        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            contour_list.append([x,y,w,h])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        
        if text == 'Occupied':
            movement_frame_list.append(frame_to_be_processed)
        elif text == 'Unoccupied' and len(movement_frame_list) > 0:
            #BMI = generate_BMI(movement_frame_list ,30)
            seperate_objects(contour_list,movement_frame_list,bgOrig)
            movement_frame_list = []
            contout_list = []

        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

def seperate_objects(contour_list ,movement_frame_list, bgFrame):
    with open('contour.pk1' ,'wb') as out:
        pickle.dump([contour_list,movement_frame_list ,bgFrame] , out , pickle.HIGHEST_PROTOCOL)
    
        
    
    
        

    
def main():
    start_video_cam()

if __name__ == '__main__':
    main()
    

                            
                
		
            
    
