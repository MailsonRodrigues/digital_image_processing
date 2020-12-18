# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:41:37 2020

@author: Mailson
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Função de controle do Trackbar de controle do limite inferior do algoritmo de Canny
def on_trackbar_canny_inf(canny_inf_slider):
    global canny_inf
    canny_inf = canny_inf_slider

#Função de controle do Trackbar de controle do limite superior do algoritmo de Canny
def on_trackbar_canny_sup(canny_sup_slider):
    global canny_sup
    canny_sup = canny_sup_slider

#Função que calcula os pontos da equação da reta sobre uma matriz de pixels
def straightEquation(m, step, cnt, angle, point):
    if angle%90 == 0:
        if angle == 0:
            x = point[0] + cnt
            y = point[1]
        elif angle == 90:
            x = point[0]
            y = point[1] + cnt
        elif angle == 180:
            x = point[0] - cnt
            y = point[1]
        elif angle == 270:
            x = point[0]
            y = point[1] - cnt
    else:
        x = step*cnt
        y = m*x
        if angle > 0 and angle < 136:
            x = int(x) + point[0]
            y = int(y) + point[1]       
            
        elif angle > 135 and angle < 316:
            x = -int(x) + point[0]
            y = -int(y) + point[1]
        
        elif angle > 315 and angle < 360:
            x = int(x) + point[0]
            y = int(y) + point[1]
    
    return int(x),int(y)

#Função que, partindo de um ponto central, encontra a borda do objeto
#em uma determinada direção
def findEndPoint(angle,center,mat):
    m = np.tan(np.deg2rad(angle))
    
    stop_criteria = int(np.sqrt(2)*max(mat.shape))*2
    
    if m > 100:
        m = 100
    
    if abs(m) > 1:
        step = 1/m
    elif abs(m) == 0:
        step = 0.1
    else:
        step = 0.1    
    cnt = 0
    p = 255
    pos = (0,0)
    while p == 255 and cnt < stop_criteria:                
        x,y = straightEquation(m, step, cnt, angle, center)
        if y < mat.shape[0] and x < mat.shape[1]:
            p = mat[y,x]  
        else:
            break
        pos = (x,y)
        cnt+=1
    return pos

#Função que retorna todos os pontos de uma reta que liga dois
#pontos determinados
def getLinePoints(begin_point, end_point,mat):
    points = []
    points.append(begin_point)
    if end_point[0] == begin_point[0]:
        for i in range(min(begin_point[1],end_point[1]),max(begin_point[1],end_point[1])):
            if end_point[0]+i < mat.shape[0]-1:
                points.append((end_point[0]+i,end_point[0]))
            else:
                return points

    elif end_point[1] == begin_point[1]:
        for i in range(min(begin_point[0],end_point[0]),max(begin_point[0],end_point[0])):
            if end_point[0]+i < mat.shape[1]-1:
                points.append((end_point[1],end_point[0]+i))
            else:
                return points
    else:
        cnt = 0
        m = (end_point[1]-begin_point[1])/(end_point[0]-begin_point[0])
        angle = int(np.rad2deg(np.arctan(m)))
        if angle < 0:
            angle = angle + 180
        dist = 11
        stop_criteria = int(np.sqrt(2)*max(mat.shape))*30

        if m > 100:
            m = 100        
        if abs(m) > 1:
            step = 1/m
        elif abs(m) == 0:
            step = 0.1
        else:
            step = 0.1  
        global c
        while dist > 2 and cnt < stop_criteria:
            
            y,x = straightEquation(m, step, cnt, angle, begin_point)
            
            dist = np.sqrt((end_point[0]-y)**2+(end_point[1]-x)**2)
            
            if y < mat.shape[1]-1 and x < mat.shape[0]-1:
                points.append((y,x))
            else:
                return points                
            
            cnt += 1
            cv2.circle(mat, (y, x), 1, (255, 0, 0), -1)
        if cnt == stop_criteria:
            points = []
    return points

captura = cv2.VideoCapture(0)

if not captura.isOpened():
    print("Erro ao abrir a câmera")
else:
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = captura.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    cv2.namedWindow("original",cv2.WINDOW_NORMAL)
    cv2.namedWindow("canny",cv2.WINDOW_NORMAL)
    cv2.namedWindow("saida",cv2.WINDOW_NORMAL)
    
    canny_inf = 120
    canny_inf_max = 550
    canny_inf_slider = 120
    
    canny_sup = 170
    canny_sup_max = 600
    canny_sup_slider = 170
    
    cv2.createTrackbar("canny inf","canny", canny_inf_slider, canny_inf_max, on_trackbar_canny_inf)
    on_trackbar_canny_inf(canny_inf)
    
    cv2.createTrackbar("canny sup","canny", canny_sup_slider, canny_sup_max, on_trackbar_canny_sup)
    on_trackbar_canny_sup(canny_sup)
    sub = False   
    filt = False
    while(1):
        ret, img = captura.read()
        
        #Remoção das primeiras linhas da imagem que continham textos
        #com a data e hora da captura
        img = img[20:img.shape[1],:,:]
        
        #Função de remoção do fundo da cena, porém, não funcionou muito bem
        if sub:
            background = cv2.imread("background.jpg")
            img = abs(img-background)
        
        #Habilitação do filtro
        if filt:
            img = cv2.medianBlur(img,5)
        
        #Normalização e conversão da imagem para grayscale
        norm = np.zeros(img.shape,np.uint8)    
        norm = cv2.normalize(img, norm, 0, 127, cv2.NORM_MINMAX)        
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        
        #Adaptação da imagem para o destaque dos resistores
        if canny_sup > canny_inf:
            canny = cv2.Canny(gray, canny_inf, canny_sup)        
        kernel = np.ones((5,5), np.uint8)                
        canny_modified = cv2.dilate(canny,kernel, iterations = 3)
        
        #Função que encontra os contornos dos resistores
        contours, hierarchy = cv2.findContours(canny_modified,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #Criação da imagem de saída
        out = np.copy(img)
        
        centers = []
        
        #Processo para encontrar os centroides
        cnt1 = 1
        for c in contours:
            if len(c) > 100: #Limitar para descartar contornos pequenos que não são resistores
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX,cY])
                cv2.circle(out, (cX, cY), 5, (255, 0, 0), -1)
                cv2.putText(out, "Resistor " + str(cnt1), (centers[-1][0] - 25, centers[-1][1] - 25),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cnt1 += 1
        
        #Análise dos centros para encontrar o eixo
        if len(centers) > 0:
            #Listas que irão armazenar as componentes de cores que passam pelo
            #eixo de todos os resistores
            red = []
            green = []
            blue = []
            for c in range(len(centers)):
                #Listas que irão armazenar os pontos finais das retas 
                r1 = []
                r2 = []
                r3 = []
                r4 = []
                
                #Passo do ângulo das retas
                angle_step = 5
                angles = np.arange(0,90,3)
                
                #Método para encontrar os pontos finais das retas para todos
                #os ângulos de acordo com o passo determinado
                for ang in angles:
                    r1.append(findEndPoint(ang, centers[c], canny_modified))
                    r2.append(findEndPoint(ang+90, centers[c], canny_modified))
                    r3.append(findEndPoint(ang+180, centers[c], canny_modified))
                    r4.append(findEndPoint(ang+270, centers[c], canny_modified))                    
                
                #Listas que irão armazenar a largura e o comprimento encontrados
                #dos resistores de acordo com as retas
                w = []
                l = []
                
                #Cálculo da distância euclidiana entre os pontos finais das retas
                #encontradas e o centroide do resistor
                
                #Os comprimentos das retas que são defasadas 180° (mesma orientação)
                #são somados e acrescentados na lista
                for i in range(len(r3)):
                    l1 = np.sqrt((centers[c][0]-r1[i][0])**2+(centers[c][1]-r1[i][1])**2)
                    l2 = np.sqrt((centers[c][0]-r2[i][0])**2+(centers[c][1]-r2[i][1])**2)
                    l3 = np.sqrt((centers[c][0]-r3[i][0])**2+(centers[c][1]-r3[i][1])**2)
                    l4 = np.sqrt((centers[c][0]-r4[i][0])**2+(centers[c][1]-r4[i][1])**2)                    
                    w.append(l1+l3)
                    l.append(l2+l3)
                
                #Verificação da medida mais longa, indicando a orientação do 
                #eix principal e os pontos final e inicial da reta do eixo
                if len(np.where(max((max(w),max(l))) == w)[0]) != 0:
                    i = np.where(max((max(w),max(l))) == w)[0][0]
                    begin_point = r3[i]
                    end_point = r1[i]
                elif len(np.where(max((max(w),max(l))) == l)[0]) != 0:
                    i = np.where(max((max(w),max(l))) == l)[0][0]
                    begin_point = r4[i]
                    end_point = r2[i]
                
                orientation = angles[i]
                
                #Exibição do ponto final e inicial da reta na imagem
                cv2.circle(out, begin_point, 5, (255, 0, 0), -1)
                cv2.circle(out, end_point, 5, (0, 0, 255), -1)
                
                '''cv2.line(canny,(centers[c][0],centers[c][1]), (r1[i][0],r1[i][1]), (255,255,255), 1)
                cv2.line(canny,(centers[c][0],centers[c][1]), (r2[i][0],r2[i][1]), (255,255,255), 1)
                cv2.line(canny,(centers[c][0],centers[c][1]), (r3[i][0],r3[i][1]), (255,255,255), 1)
                cv2.line(canny,(centers[c][0],centers[c][1]), (r4[i][0],r4[i][1]), (255,255,255), 1)'''
                
                #Pontos da reta que conecta o ponto final e o inicial
                pontos = getLinePoints(begin_point, end_point,out)
                r = np.zeros(len(pontos))
                g = np.zeros(len(pontos))
                b = np.zeros(len(pontos))
                
                #Obtenção das cores nos pontos da reta
                for i in range(len(r)):
                    r[i] = img[pontos[i][1],pontos[i][0],2]
                    g[i] = img[pontos[i][1],pontos[i][0],1]
                    b[i] = img[pontos[i][1],pontos[i][0],0]
                
                red.append(r)
                green.append(g)
                blue.append(b)
                

            '''cv2.line(out, r1[i], r2[i], (0,255,0), 2)
            cv2.line(out,r2[i], r3[i], (0,255,0), 2)
            cv2.line(out,r3[i], r4[i], (0,255,0), 2)
            cv2.line(out,r4[i], r1[i], (0,255,0), 2)'''
            
        #Exibição das imagens
        cv2.imshow("original", img)
        cv2.imshow("canny", canny_modified)
        cv2.imshow("saida", out)
        
        #Funções do teclado
        k = chr(cv2.waitKey(10) & 0xff)
        if ord(k) == 27:
            cv2.destroyAllWindows()
            plt.close('all')
            break
        elif k == 'b':
            cv2.imwrite("./background.jpg",img)
        elif k == 's':
            sub = not sub
        elif k == 'f':
            filt = not filt

#Função para plotar as cores do eixo principal. Recebe como parâmetro as lista
#das componentes de cores no eixo e o índice indicando qual resistor da cena
#deve ser exibido. Além disso, indica-se o número da figura para que se possa
#ter múltiplas janelas de gráficos
def plotRGB(red,green,blue,index,fignum):
    plt.figure(fignum,[15,15])
    plt.subplot(311)
    plt.plot(red[index], color = "red")
    plt.title("Componente R do eixo")
    
    plt.subplot(312)
    plt.plot(green[index], color = "green")
    plt.title("Componente G do eixo")
    
    plt.subplot(313)
    plt.plot(blue[index], color = "blue")
    plt.title("Componente B do eixo")

