import cv2
import numpy as np
from collections import deque
#mavi nesneyi algılayıp hareketini çizen renk takibi uygulaması
dq=deque(maxlen=3)

#%%
buffer_size=16 #hafizada kac konumun tutulacagını belirler
pts=deque(maxlen=buffer_size)#her karede tespit edilen merkezler burada birikir hareket çizgisi cizilir

bluelower=(84,98,90)#hsv renk uzayında mavi rengin alt ve ust sınırları
blueupper=(179,150,150)

cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)#kamera cozunurluk ayarlanır 3 genislik 4 yukseklık
while True:
    success,imorginal=cap.read()
    #success başarılı olup olmadığı imprginal alınan goruntu
    
    #göruntu alımı basarılı ise 
    if success:
        blurred=cv2.GaussianBlur(imorginal,(11,11),0)
        #gurultuyu azaltmak icib bulanıklastırma uygula 11,11: filtre buyuklugu 0 : standart sapma
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        #bgr den hsv ye cevir
        cv2.imshow("hsv penceresi",hsv)
        
        mask=cv2.inRange(hsv,bluelower,blueupper)
        #belirlenen mavi aralıgındaki pikselleri beyaz yapar digerlerini siyah yapar
        mask=cv2.erode(mask,None,iterations=2)
        #beyaz bölgeleri kucultur kucuk gurultulerı temizler
        mask=cv2.dilate(mask,None,iterations=2)
        #nesnenin boyutunu geri büyütür
        cv2.imshow("maskeli pencere",mask)
        #beyaz:mavi nesne siyah:diger alanlar 

        (contours,_)=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center=None
        # beyaz alanları yani mavi nesne olanlarınn dıs hatları konturları bulunur .retr_externel sadece dıs konturları alır 
       
        if len(contours)>0:
            c=max(contours,key=cv2.contourArea)
           #en buyuk konturu sec.bu genellikle nesnenin kendisidir.
            rect=cv2.minAreaRect(c)
            #secilen konturun etrafına minimum döndürülebilir dikdörtgen seçilir
            ((x,y),(width,height),rotation)=rect
            #merkezi x,y boyutları ve döndürülme açısı alınır
            s="x:{},y:{},width:{},height:{},rotation{}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            #bilgileri yazdır. np.round  ile sayılar tam hale getirlir
           
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            #boxPoints: dikdörtgenin kose noktlarını verir çizmek icin tam sayı tipine cevrilir

            M=cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            #konturun merkezini hesaplamak icin moment kullanılr. ilki x ekseninde merkez ikincisi y ekseninde merkez
            cv2.drawContours(imorginal,[box],0,(0,255,255),2)
            #mavi nesne etrafına dikdörtgen çizilir sarı renkte  
            cv2.circle(imorginal,center,5,(255,0,0))
            #merekze mavi bir daire çiz 
        pts.append(center)
        #hareket gecmisi iin merkez noktasını pts dizisine ekle 
        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:continue
            cv2.line(imorginal,pts[i-1],pts[i],(0,255,255),3)
       #pts listesiden birden fazla merkez varsa ardsık merkezler arası cizgi çiz bu sayede nesnenin hareketini izleyebilirsin sarı renkte
        cv2.imshow("orijinal",imorginal)   



    if cv2.waitKey(1) & 0xFF==ord("q"):
       break

cv2.destroyAllWindows()




