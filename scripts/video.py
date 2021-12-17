import imageio
import cv2
import numpy as np


class RealTimeDetector():
    def __init__(self):
        pass
   
    def detectar(self,clasificador,detector):
        capture = cv2.VideoCapture(0)
        fps = 10

        frame_counter = 0
        video_wfp = 'video.avi'
        writer = imageio.get_writer(video_wfp, fps=fps)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while (capture.isOpened()):
                ret, frame = capture.read()
                if (ret == True):            
                        
                    frame = cv2.flip(frame, 1)
                    
                    detectadas,img_aux,pred = detector.contar_detectar(frame,clasificador)
                    
                    cv2.putText(img_aux,'Clase:{} Detectadas:{}'.format(pred,detectadas), 
                    (0, 20), 
                    font, 1, 
                    (0, 255, 255), 
                     2, 
                    cv2.LINE_4)
                        
                    cv2.imshow('image', img_aux)
                    writer.append_data(img_aux[...,::-1])
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.close()
        capture.release()
        cv2.destroyAllWindows()