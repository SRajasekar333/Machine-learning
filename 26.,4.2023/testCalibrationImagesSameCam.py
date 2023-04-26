import cv2
import time
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

num = 0
start_time = time.time()
#print(start_time)
elapsed_time = 0

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()
 
    elapsed_time = time.time() - start_time
    #print(elapsed_time)

    if round(elapsed_time,1) == 5.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(5)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(5)' + str(num) + '.png', img2)
        print("T5images saved!")
        num += 1
    if round(elapsed_time,1) == 10.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(10)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(10)' + str(num) + '.png', img2)
        print("T10images saved!")
        num += 1
    if round(elapsed_time,1) == 15.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(15)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(15)' + str(num) + '.png', img2)
        print("T15images saved!")
        num += 1
    if round(elapsed_time,1) == 20.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(20)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(20)' + str(num) + '.png', img2)
        print("T20images saved!")
        num += 1
    if round(elapsed_time,1) == 25.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(25)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(25)' + str(num) + '.png', img2)
        print("T25images saved!")
        num += 1
    if round(elapsed_time,1) == 30.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(30)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(30)' + str(num) + '.png', img2)
        print("T30images saved!")
        num += 1
    if round(elapsed_time,1) == 35.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(35)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(35)' + str(num) + '.png', img2)
        print("T35images saved!")
        num += 1
    if round(elapsed_time,1) == 40.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(40)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(40)' + str(num) + '.png', img2)
        print("T40images saved!")
        num += 1
    if round(elapsed_time,1) == 45.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(45)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(45)' + str(num) + '.png', img2)
        print("T45images saved!")
        num += 1
    if round(elapsed_time,1) == 50.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(50)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(50)' + str(num) + '.png', img2)
        print("T50images saved!")
        num += 1
    if round(elapsed_time,1) == 55.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(55)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(55)' + str(num) + '.png', img2)
        print("T55images saved!")
        num += 1
    if round(elapsed_time,1) == 60.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(60)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(60)' + str(num) + '.png', img2)
        print("T60images saved!")
        num += 1
    if round(elapsed_time,1) == 65.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(65)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(65)' + str(num) + '.png', img2)
        print("T65images saved!")
        num += 1
    if round(elapsed_time,1) == 70.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(70)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(70)' + str(num) + '.png', img2)
        print("T70images saved!")
        num += 1
    if round(elapsed_time,1) == 75.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(75)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(75)' + str(num) + '.png', img2)
        print("T75images saved!")
        num += 1
    if round(elapsed_time,1) == 80.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(80)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(80)' + str(num) + '.png', img2)
        print("T80images saved!")
        num += 1
    if round(elapsed_time,1) == 85.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(85)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(85)' + str(num) + '.png', img2)
        print("T85images saved!")
        num += 1
    if round(elapsed_time,1) == 90.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(90)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(90)' + str(num) + '.png', img2)
        print("T90images saved!")
        num += 1
    if round(elapsed_time,1) == 95.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(95)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(95)' + str(num) + '.png', img2)
        print("T95images saved!")
        num += 1
    if round(elapsed_time,1) == 100.0: # wait for 's' key to save and exit
        cv2.imwrite('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(100)' + str(num) + '.png', img)
        cv2.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_sphere/imageR(100)' + str(num) + '.png', img2)
        print("T100images saved!")
        num += 1
   

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)

# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()