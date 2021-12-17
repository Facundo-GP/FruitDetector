import numpy as np
import cv2
import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img,cmap='gray')
    plt.show()
    

def filtrar_areas(img,th):
    connectivity = 8
    out = cv2.connectedComponentsWithStats(img.copy(), connectivity)
    (numLabels, labels, stats, centroids) = out
    mask = np.zeros(img.shape, dtype="uint8")
    
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > th:
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
    
    return mask




def suavizar(img,th1,th2):
    
    canny_output = cv2.Canny(img, th1,th2)
    
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 1), dtype=np.uint8)
    
    for i in range(len(contours)):
        color = (np.random.randint(0,256))
        cv2.drawContours(drawing, hull_list, i, color)
        
    cv2.fillPoly(drawing, pts =hull_list, color=(255))
    
    
    return drawing


def background(I,Ib):
    return I*np.moveaxis(np.array([Ib,Ib,Ib]),0,-1)

def eliminar_fondo(img,area_th = 2000,blur_dim = 7,morph_kernel = 7):
    
    I=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
    Iy=cv2.cvtColor(I.copy(),cv2.COLOR_RGB2YUV)
    u=Iy[...,1]
    v=Iy[...,2]
    
    _, thru = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thrv = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    ub=u>thru
    vb=v<thrv
    orb=ub+vb
    
    kernel = np.ones((morph_kernel,morph_kernel),np.uint8)
    orb_c=cv2.morphologyEx(np.array(orb,np.uint8), cv2.MORPH_OPEN, kernel)
    
    orb_c_f= (filtrar_areas(orb_c,area_th)/255).astype(np.uint8)
    
    blur = cv2.medianBlur(orb_c_f,blur_dim)
    bk=background(img.copy(),blur)
    
    return bk
