import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats as scp



def stats(aplicar=False,fruta="bananas",separar=False):
    def decorator(func):
        if aplicar and separar:
            def wrapper(self,ds,**kwargs):
                
                detectadas = []
                totales = []
                error = []
                mape = []
                acierto=0

                cont=1
                for img,label in zip (ds[0],ds[1]):
                    
                    print("{}/{}".format(cont,len(ds[1])),end='\r',flush=True)
                    
                    frutas_detectadas,_ = func(self,img,**kwargs)

                    detectadas.append(frutas_detectadas)
                    totales.append(label[0])
                    error.append(frutas_detectadas-label[0])
                    mape.append(np.abs(frutas_detectadas-label[0])/label[0])
                    acierto+= (frutas_detectadas == label[0])/len(ds[1])
                    cont+=1

                print("tercer momento {:.2f}".format(scp.skew(error)))
                print("mape {:.2f}%".format(np.mean(mape)*100))
                print("porcentaje de acierto {:.2f}%".format(acierto*100))

                plt.subplots(figsize = (20,8))
                plt.title("Detector de " + fruta,size=20)
                plt.plot(list(range(len(totales))),totales)
                plt.plot(list(range(len(detectadas))),detectadas)

                plt.xlabel("Imagen",size = 20)
                plt.ylabel("Cantidad de frutas",size=20)

                plt.rcParams["xtick.labelsize"] = 20
                plt.rcParams["ytick.labelsize"] = 20

                plt.legend(['labels','detectados'],prop ={"size":15})
                plt.show()

        elif (aplicar and (not separar)):
            def wrapper(self,img,label,totales,detectadas,error,mape,acierto,**kwargs):
             
                frutas_detectadas,_ = func(self,img,**kwargs)
                
                detectadas.append(frutas_detectadas)
                totales.append(label)
                error.append(frutas_detectadas-label)
                mape.append(np.abs(frutas_detectadas-label)/label)
                acierto+= (frutas_detectadas == label)
                
                return totales,detectadas,error,mape,acierto      
     
            
        else:
            def wrapper(self,img,**kwargs):
                frutas_detectadas,img_aux = func(self,img,**kwargs)
                return frutas_detectadas,img_aux

        return wrapper
    return decorator
            
    