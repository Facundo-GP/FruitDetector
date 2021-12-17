from scripts.detector_utils import *
from scripts.estadisticas import stats
import scipy.stats as scp

class FruitDetector():
    def __init__(self,analysis=False,split=True):
        
        class FruitDetectorMode():

            def __init__(self_aux):
                self.dim = (320,258)
                self.dim_manzanas = (480,322)
                

            @stats(aplicar=analysis,fruta="bananas",separar=split)
            def detectar_banana(self_aux,img,th = 86,th_filtro_area = 343,th1_canny =247,th2_canny=6, erode_kernel = 8,
                                close_kernel = 1,base_area=2845, blur_dim_fondo = 3,morph_kernel_fondo = 11, 
                                area_th_fondo = 1829, eps = 0.004685190931126466, show_img=False):


                img = cv2.resize(img,self.dim)
                
                bk = eliminar_fondo(img,
                                    area_th = area_th_fondo,
                                    blur_dim = blur_dim_fondo,
                                    morph_kernel = morph_kernel_fondo)

                bk = cv2.cvtColor(bk,cv2.COLOR_BGR2GRAY)
                _,cl = cv2.threshold(bk, th, 255,cv2.THRESH_BINARY)

                kernel = np.ones((erode_kernel,erode_kernel))  
                er = cv2.morphologyEx(cl.astype(np.uint8), cv2.MORPH_ERODE, kernel)


                cc = filtrar_areas(er,th_filtro_area)


                kernel = np.ones((close_kernel,close_kernel))
                cl = cv2.morphologyEx(cc.astype(np.uint8), cv2.MORPH_CLOSE, kernel,iterations=7)
                
    
                img_aux = img.copy()
                cnts,_ = cv2.findContours(cl,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
                detectados = 0
                for c in cnts:

                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    area = cv2.contourArea(approx)
                    if area < base_area:
                        detectados+=1
                    else:
                        detectados+= int(area/base_area)
                    
                    cv2.drawContours(img_aux, [approx], -1, (0, 255, 0), 3)
                    
                if show_img:
                    show(img_aux[...,::-1])
       
                    
                return detectados,img_aux



            @stats(aplicar=analysis,fruta="ciruelas",separar=split)
            def detectar_ciruela(self_aux,img,th = 35,kernel_blur = 3,minDist = 30,param1=24,param2=22,
                                 minRadius=12,maxRadius=31, close_kernel = 1,show_img = False):
        
                
                img = cv2.resize(img,self.dim)
                
                cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
                blur = cv2.medianBlur(cimg,kernel_blur)

                _,blur = cv2.threshold(blur, th, 255,cv2.THRESH_BINARY)

                kernel = np.ones((close_kernel,close_kernel))
                cl = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)


                circles = cv2.HoughCircles(cl,cv2.HOUGH_GRADIENT,1.5,minDist,param1=param1,param2=param2,minRadius=minRadius,
                                           maxRadius=maxRadius) 
                try:
                    img_aux = img.copy()
                    circles = np.uint16(np.around(circles)) 
                    for i in circles[0,:]: 
                        cv2.circle(img_aux,(i[0],i[1]),i[2],(0,255,0),2) 
                        cv2.circle(img_aux,(i[0],i[1]),2,(0,0,255),3)

                    if show_img:
                        show(img_aux[...,::-1])
                except:
                    pass

                try:
                    detectadas = len(circles[0])
                except:
                    detectadas = 0

                return detectadas,img_aux

            
            @stats(aplicar=analysis,fruta="manzanas",separar=split)
            def detectar_manzana(self_aux,img,minDist=88,param1=31,param2=35,minRadius=18,maxRadius=37,blur_dim=1, 
                             blur_dim_fondo = 21,morph_kernel_fondo = 2,area_th_fondo = 2653,show_img = False):
                
                
                
                img = cv2.resize(img,self.dim_manzanas)
                
                bk = eliminar_fondo(img,
                                    area_th = area_th_fondo,
                                    blur_dim = blur_dim_fondo,
                                    morph_kernel = morph_kernel_fondo)

                I = cv2.medianBlur(bk,blur_dim) 
                cimg = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)


                circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1.5,minDist, param1=param1,
                                               param2=param2,minRadius=minRadius,maxRadius=maxRadius) 

                try:
                    img_aux = img.copy()
                    circles = np.uint16(np.around(circles)) 
                    for i in circles[0,:]: 
                        cv2.circle(img_aux,(i[0],i[1]),i[2],(0,255,0),2) 
                        cv2.circle(img_aux,(i[0],i[1]),2,(0,0,255),3)

                    if show_img:
                        show(img_aux[...,::-1])
                except:
                    pass

                try:
                    detectadas = len(circles[0])
                except:
                    detectadas = 0

                return detectadas,img_aux



            @stats(aplicar=analysis,fruta="mangos",separar=split)
            def detectar_mango(self_aux,img,th=105,erode_kernel=17,th_areas=562,close_kernel=3,base_area=3225,
                           th1_canny=206,th2_canny=17,show_img=False):
                
                
                
                img = cv2.resize(img,self.dim)
                
                bk = eliminar_fondo(img)
                bk = cv2.cvtColor(bk,cv2.COLOR_BGR2GRAY)
                _,cl = cv2.threshold(bk, th, 255,cv2.THRESH_BINARY)

                kernel = np.ones((erode_kernel,erode_kernel))  
                er = cv2.morphologyEx(cl.astype(np.uint8), cv2.MORPH_ERODE, kernel)


                th = th_areas
                cc = filtrar_areas(er,th)


                kernel = np.ones((close_kernel,close_kernel))
                cl = cv2.morphologyEx(cc.astype(np.uint8), cv2.MORPH_CLOSE, kernel,iterations=7)

                dr = suavizar(cl,th1_canny,th2_canny)
                out = cv2.connectedComponentsWithStats(dr.copy(),8)
                area = out[2][:,-1]

                contours, hierarchy = cv2.findContours(dr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                try:
                    base_area = base_area
                    detectados = 0
                    for area in area[1:]:
                        if area < base_area:
                            detectados+=1
                        else:
                            detectados+= int(area/base_area)

                    img_aux = img.copy()
                    cv2.drawContours(img_aux, contours, -1, (0,0,255), 3)
                    if show_img:
                        show(img_aux[...,::-1])
                except:
                    detectados = 0

                return detectados,img_aux



            @stats(aplicar=analysis,fruta="kiwis",separar=split)
            def detectar_kiwi(self_aux,img,th=59,kernel_blur = 7,minDist = 28,param1=49,param2=15,minRadius=23,maxRadius=41,
                          area_th_fondo = 1508,blur_dim_fondo = 17,morph_kernel_fondo = 14,show_img = False):

                
                img = cv2.resize(img,self.dim)
                
                bk = eliminar_fondo(img,
                                    area_th = area_th_fondo,
                                    blur_dim = blur_dim_fondo,
                                    morph_kernel = morph_kernel_fondo)

                cimg = cv2.cvtColor(bk,cv2.COLOR_BGR2GRAY) 

                _,cl = cv2.threshold(cimg, th, 255,cv2.THRESH_BINARY)
                blur = cv2.medianBlur(cl,kernel_blur)

                circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1.5,minDist, param1=param1,param2=param2,
                                           minRadius=minRadius,maxRadius=maxRadius) 
                try:
                    img_aux = img.copy()
                    circles = np.uint16(np.around(circles)) 
                    for i in circles[0,:]: 
                        cv2.circle(img_aux,(i[0],i[1]),i[2],(0,255,0),2) 
                        cv2.circle(img_aux,(i[0],i[1]),2,(0,0,255),3)

                    if show_img:
                        show(img_aux[...,::-1])
                except:
                    pass

                try:
                    detectadas = len(circles[0])
                except:
                    detectadas = 0

                return detectadas,img_aux

            @stats(aplicar=analysis,fruta="naranjas",separar=split)
            def detectar_naranja(self_aux,img,area_th_fondo = 2600,blur_dim_fondo =9,morph_kernel_fondo=16,
                                 medianblur_dim=3,minDist=51,param1=3,param2=41,minRadius=22,maxRadius=38,
                                 show_img = False):
                
                img = cv2.resize(img,self.dim)
                
                bk = eliminar_fondo(img,area_th = area_th_fondo,blur_dim  = blur_dim_fondo,
                                    morph_kernel = morph_kernel_fondo)

                I = cv2.medianBlur(bk,medianblur_dim) 
                cimg = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1.5,minDist, param1=param1,param2=param2,
                                           minRadius=minRadius,maxRadius=maxRadius) 

                img_aux = img.copy()
                try:
                    circles = np.uint16(np.around(circles)) 
                    for i in circles[0,:]: 
                        cv2.circle(img_aux,(i[0],i[1]),i[2],(0,255,0),2) 
                        cv2.circle(img_aux,(i[0],i[1]),2,(0,0,255),3)

                    if show_img:
                        show(img_aux[...,::-1])
                except:
                    pass

                try:
                    detectadas = len(circles[0])
                except:
                    detectadas = 0

                return detectadas,img_aux
            
            def contar_detectar(self,img,clasificador):
                pred = clasificador.pred(img)
                
        
            
                if pred == "Banana":
                    detectadas,img_aux = self.detectar_banana(img)
                elif pred == "Manzana":
                    detectadas,img_aux = self.detectar_manzana(img)
                elif pred == "Naranja":
                    detectadas,img_aux = self.detectar_naranja(img)
                elif pred == "Kiwi":
                    detectadas,img_aux = self.detectar_kiwi(img)
                elif pred == "Ciruela":
                    detectadas,img_aux = self.detectar_ciruela(img)
                elif pred == "Mango":
                    detectadas,img_aux = self.detectar_mango(img)
                
                else:
                    detectadas = 0
                    img_aux=I
                
                return detectadas,img_aux,pred
            
            
            def analizar_todas(self,data_loader):
                
                totales = []
                detectadas = []
                error = []
                mape = []
                acierto=0
        
                
                bananas_ds,ciruelas_ds,kiwis_ds,mangos_ds,manzanas_ds,naranjas_ds = data_loader.todos()
                total = len(manzanas_ds[1]) + len(naranjas_ds[1]) + len(bananas_ds[1]) + len(ciruelas_ds[1]) + len(mangos_ds[1]) + len(kiwis_ds[1])
 
                
               
                cont = 1
                for img,label in zip (manzanas_ds[0],manzanas_ds[1]):
                    print("{}/{}".format(cont,total),end='\r',flush=True)
                    totales,detectadas,error,mape,acierto = self.detectar_manzana(img,label[0],totales,detectadas,
                                                                                   error,mape,acierto) 
                    cont+=1
                
  
                for img,label in zip (naranjas_ds[0],naranjas_ds[1]):
                    print("{}/{}".format(cont,total),end='\r',flush=True)
                    totales,detectadas,error,mape,acierto = self.detectar_naranja(img,label[0],totales,detectadas,
                                                                                  error,mape,acierto)
                    cont+=1
  
             
                for img,label in zip (bananas_ds[0],bananas_ds[1]):
                    print("{}/{}".format(cont,total),end='\r',flush=True)
                    totales,detectadas,error,mape,acierto = self.detectar_banana(img,label[0],totales,detectadas,
                                                                                 error,mape,acierto)
                    cont+=1

                for img,label in zip (ciruelas_ds[0],ciruelas_ds[1]):
                    print("{}/{}".format(cont,total),end='\r',flush=True)
                    totales,detectadas,error,mape,acierto = self.detectar_ciruela(img,label[0],totales,detectadas,error,
                                                                                  mape,acierto)  
                    cont+=1
  
                for img,label in zip (mangos_ds[0],mangos_ds[1]):
                    print("{}/{}".format(cont,total),end='\r',flush=True)
                    totales,detectadas,error,mape,acierto = self.detectar_mango(img,label[0],totales,detectadas,error,
                                                                                mape,acierto)
                    cont+=1
                    
                for img,label in zip (kiwis_ds[0],kiwis_ds[1]):
                    print("{}/{}".format(cont,total),end='\r',flush=True)
                    totales,detectadas,error,mape,acierto = self.detectar_kiwi(img,label[0],totales,detectadas,error,
                                                                               mape,acierto)
                    cont+=1
              
                print("tercer momento {:.2f}".format(scp.skew(error)))
                print("mape {:.2f}%".format(np.mean(mape)*100))
                print("porcentaje de acierto {:.2f}%".format(acierto*100/total))

                plt.subplots(figsize = (20,8))
                plt.title("Detector de frutas",size=20)
                plt.plot(list(range(len(totales))),totales)
                plt.plot(list(range(len(detectadas))),detectadas)

                plt.xlabel("Imagen",size = 20)
                plt.ylabel("Cantidad de frutas",size=20)

                plt.rcParams["xtick.labelsize"] = 20
                plt.rcParams["ytick.labelsize"] = 20

                plt.legend(['labels','detectados'],prop ={"size":15})
                plt.show()
         
        self.detector = FruitDetectorMode()
        
    def get(self):
        return self.detector
        




        