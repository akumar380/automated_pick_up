import numpy as np
import cv2
from matplotlib import pyplot as plt
import Tkinter
import tkMessageBox

import datetime

class ImageProcessor(object):
    """This is the base class for all classes that deal with segmentation for the robot"""

    def __init__(self):
        self.currentTime = datetime.datetime.now() #stores current time during initialization of the ImageProcessor class
        root = Tkinter.Tk()
        root.withdraw()
        
    def displayImage(self, windowname, img_array):
        tkMessageBox.showinfo('Image Display', 'Press esc to exit')
        while(1):
            cv2.imshow(windowname, img_array)
            if cv2.waitKey(0) == 27:
                #esc is pressed
                break
        cv2.destroyAllWindows()
        
    def whatime(self):
        """Convert currentTime completely into seconds"""
        self.currentTimeSec = self.currentTime.year * 3.154e+7 + self.currentTime.month * 2628336.2137829 + self.currentTime.day * 86410.958906880114228 \
                              + self.currentTime.hour * 3600 + self.currentTime.minute * 60 + self.currentTime.second
        #print(self.currentTimeSec)

    def captureVideo(self):
        '''Initialize the camera and begin recording. When q is pressed, feed is stopped and the video file is saved.'''
        self.vid = cv2.VideoCapture(0) #begins video capture from default camera (webcam in this case)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        img_counter = 0        

        while (self.vid.read()):
            #capture frame-by-frame
            self.ret, self.frame = self.vid.read()
            out.write(self.frame) #write each frame to video

            if not self.ret:
                break
            
            if cv2.waitKey(67) == 27: #stop feed when 'esc' is pressed
                print('Q hit...Feed Stopped')
                break
            elif cv2.waitKey(67) == 115:
                self.take_screenshot(img_counter) #takes a screenshot of video feed when 's' is pressed
                img_counter += 1
        self.vid.release()
        out.release()
        cv2.destroyAllWindows()

    def loadSavedVideo(self, vidname):
        '''Play already saved videos
           Inputs: vidname - string name and path of video to be played
           Outputs: name - name of image snapshot from video'''
        self.vid = cv2.VideoCapture(vidname) #plays video from file 'vidname'
        tkMessageBox.showinfo('Video Playing', 'Video will begin. Press \'s\' to take a screenshot or \'esc\' to quit')

        img_counter = 0
        name = None
        while(self.vid.read()):
            self.ret, self.frame = self.vid.read() #each frame is read            
            self.globalthresh, self.originalimg = self.getLocation(self.frame)
            cv2.imshow('Global Threshold', self.globalthresh)
            cv2.imshow('Actual Video', self.originalimg)
            if not self.ret:
                break
            
            if cv2.waitKey(67) == 27: #waitKey(67) because frame rate is 15 so ~67 ms waiting time for each frame
                tkMessageBox.showinfo('Quitting', 'Esc hit...')
                break
            elif cv2.waitKey(67) == 115:
                name = self.take_screenshot(img_counter) #takes a screenshot of video feed when 's' is pressed
                img_counter += 1

        self.vid.release()
        cv2.destroyAllWindows()

        if name is not None:
            self.workWithMouse(name)
        else:
            tkMessageBox.showinfo('Quit Program', 'No screenshots were saved')
        
    def take_screenshot(self, i):
        '''Take a snapshot during video recording
           Inputs:
           i - snapshot number
           Outputs:
           img_name - name of image where screenshot is saved
           '''
        img_name = "opencv_frame_{}.jpg".format(i) 
        cv2.imwrite(img_name, self.frame) #write frame to an image and save image
        tkMessageBox.showinfo('Screenshot Saved.', img_name + ' was saved')
        return img_name

    #def take_pic(self, i, real_coordinates, reference_image):
        #real_image = cv2.imread(img_name, 1)
        #copy_image = real_image
        #img = crop(real_coordinates, copy_image)
        #time_of_acq = self.whatime()
        #filename = 'img' + str(i)
        #imagename = filename + '.png'
        #self.getLocation(img, imagename, reference_image, real_image)
        
    
class FeatureManager(ImageProcessor):
    """This is a subclass of ImageProcessor. It contains all methods for recognizing features, tracking objects, and manipulating color/size"""

    def __init__(self):
        super(FeatureManager, self).__init__()

    def detectCircles(self, imagename):
        '''Detect circles in input image
           Inputs:
           imagename - name of image in which circles have to be detected
           Outputs:
           circles - detected circles stored as an array
           circles_img - image array with detected circles drawn'''
        circles_img = cv2.imread(imagename, 0)
        circles_img2 = circles_img #create a duplicate copy of input image
        circles_img = cv2.medianBlur(circles_img, 5) #used to blur original image so that circles drawn are visible
        circles = cv2.HoughCircles(circles_img, cv2.HOUGH_GRADIENT, 1, 75, param1=50, param2=35) #circle detection
        circles = np.array(circles) #convert output into a 3-D array [(x-center, y-center, radius)...]
        for i in np.arange(0, circles.shape[1]):
            circles_img = cv2.circle(circles_img, (circles[0, i, 0], circles[0, i, 1]), circles[0, i, 2], [0, 0, 255], 0) #draw each circle
        return circles_img, circles, circles_img2
    
class CoordinateCalculator(FeatureManager):
    """This is a subclass of FeatureManager. It contains all methods for calculating coordinates"""
    mouse_x, mouse_y = -1, -1

    def __init__(self):
        super(CoordinateCalculator, self).__init__()

    def mouseCoordinates(self, event,x,y,flags,param):
        '''Function to obtain coordinates of mouse click'''
        global mouse_x, mouse_y #coordinates are global so can be read by any function in this class
        if event == cv2.EVENT_LBUTTONDBLCLK: #DOUBLE CLICK on left mouse button
            mouse_x, mouse_y = x, y
            print(mouse_x, mouse_y)
            return [mouse_x, mouse_y]

    def workWithMouse(self, imagename):
        '''Function to use mouseCoordinates() function to get mouse's position
           Inputs:
           imagename - name of the image where mouse coordinates need to be found'''
        cv2.namedWindow('Detected Circles')
        cv2.setMouseCallback('Detected Circles', self.mouseCoordinates) #use mouseCoordinates function to get mouse's position
        circles_img, circles, cleanimage = self.detectCircles(imagename) #detect circles in input image
        tkMessageBox.showinfo('Screenshot Saved', "On this image, click on center of end effector loop and press esc")
        self.displayImage('Detected Circles', circles_img)
        circles_img2 = self.closestCircle(cleanimage, circles)
        self.displayImage('Image with User-Selected Circle', circles_img2) #shows image with user-selected circle

    def crop(self, coordinates, img_array):
        '''Function to crop an image to specified coordinates
           Inputs:
           coordinates - vector of coordinates of image to crop to. In the form [x1, y1, x2, y2]
           img_array - the array form of the image to crop
           Outputs:
           newimg_array - cropped image array'''
        x1 = coordinates[1]
        y1 = coordinates[2]
        x2 = coordinates[3]
        y2 = coordinates[4]

        margin = 30
        shapevec = np.shape(img_array)

        if len(shapevec) == 2:
            #DIRECT TRANSLATION OF MATLAB CODE
            #top left corner of pool
            img_array[:, 0:x1+margin] = 0
            img_array[0:y1+margin, :] = 0
            #top right corner of pool
            img_array[:, x2-margin:shapevec[1]] = 0
            img_array[0:y2+margin, :] = 0

        elif len(shapevec) == 3:
            #top left corner of pool
            img_array[:, 0:x1+margin, :] = 0
            img_array[0:y1+margin,:, :] = 0
            #top right corner of pool
            img_array[:, x2-margin:shapevec[1], 0] = 0
            img_array[0:y2+margin, :, :] = 0

        #Is there a better way (in 1 line) to index these?

        newimg_array = img_array
        return newimg_array

    def colorManip(self, img_array, lr, lg, lb, maxval):
        '''
           This function takes in a color image, takes each color channel and thresholds each color channel - according to
           its own threshold value - to convert each channel to a black and white image. Then the 3 images are recombined
           and the minimum common set of pixels is retained.
           Inputs:
           img_array - array of image to manipulate color
           lr - threshold value for red channel
           lg - threshold value for green channel
           lb - threshold value for blue channel
           maxval - maximum value that pixels with values greater than threshold will be set to
           Outputs:
           newimg_array - array form of manipulated image'''
        # Crop the image
        totalrows = img_array.shape[0]
        totalcols = img_array.shape[1]

        img_array = img_array[0:totalrows - np.round(totalrows/8), np.round(totalcols/8):totalcols - np.round(totalcols/8), :]         
        bmat = img_array[:, :, 0]
        gmat = img_array[:, :, 1]
        rmat = img_array[:, :, 2]

        #Applied binary thresholding to each color channel separately
        self.newimg_array = np.array(img_array)
        
        cr = cv2.threshold(rmat, lr, maxval, cv2.THRESH_BINARY)
        cg = cv2.threshold(gmat, lg, maxval, cv2.THRESH_BINARY)
        cb = cv2.threshold(bmat, lb, maxval, cv2.THRESH_BINARY)

        self.newimg_array = cr[1] & cg[1] & cb[1]                            
        return self.newimg_array 
    

    def closestCircle(self, originalimage, circles):
        """
        This function calculates the distance between the circle centers found by program and the point selected by user.
        Outputs xc and yc correspond to minimum distance between centers provided by program and point selected by user.
        Inputs: circles - vector of circles found by HoughCircles() method. Contains coordinates of center and radius (x. y, r)
                mouse_x - x coordinate of point selected by user
                mouse_y - y coordinate of point selected by user of center of end effector loop]"""
        distances = np.zeros(circles.shape[1], dtype=np.float) #create a zero array
        lowest = 500
        lowestInd = 0
        
        for i in np.arange(0, circles.shape[1]): #loop through values of center points of circles obtained from HoughCircles()
            distances[i] = np.sqrt((circles[0, i, 0]-mouse_x)**2 + (circles[0, i, 1]-mouse_y)**2) #distance formula
            if distances[i] < lowest:
                lowest = distances[i] #calculates lowest distance
                lowestInd = i #index of lowest distance
        print(lowest)
        circles_img2 = cv2.circle(originalimage, (circles[0, lowestInd, 0], circles[0, lowestInd, 1]), circles[0, lowestInd, 2], [0, 0, 255], 0)
        #draws circle closest to user-selected point on duplicate copy of original image
        #print(circles[0, lowestInd, 0], self.circles[0, lowestInd, 1])
        return circles_img2


    def getLocation(self, originalimg):
        '''Finds centroid of section from thresholded image.
           Inputs:
           originalimg - original image that needs to be thresholded
           Outputs:
           globalthresh - cropped binary thresholded image
           originalimg - original image after cropping and drawing corners and centroids'''

        outlines = self.colorManip(originalimg, 50, 60, 50, 255) #threshold the input image
        kernel = np.ones((6,6),np.uint8) #6x6 kernel of 1s
        erosion = cv2.erode(outlines,kernel,iterations = 1) #noise reduction by erosion
        globalthresh = cv2.dilate(erosion, kernel, iterations = 1) #get back what was lost by erosion using dilation

        #crop original image to same size as thresholded image        
        totalrows = originalimg.shape[0]
        totalcols = originalimg.shape[1]
        originalimg = originalimg[0:totalrows - np.round(totalrows/8), np.round(totalcols/8):totalcols - np.round(totalcols/8), :]

        #find contours in thresholded image
        image, contours, hierarchy = cv2.findContours(globalthresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for cont in contours:
            if cv2.contourArea(cont) >= 500 and cv2.contourArea(cont) <= 1000: #area must be between 500 and 1000
                if cv2.arcLength(cont, True) <= 500: #perimeter must be greater than 500
                    M = cv2.moments(cont) #find moment of contour
                    cx = int(M['m10']/M['m00']) #x coordinate of centroid
                    cy = int(M['m01']/M['m00']) #y coordinate of centroid
                    originalimg = cv2.drawContours(originalimg, cont, -1, (255, 0, 0), 5) #draw contours in blue
                    originalimg = cv2.circle(originalimg,(cx,cy), 5, (0,0,255), -1) #draw centroid in red
            
        return globalthresh, originalimg


#SCRIPT TO RUN PROGRAM
cc = CoordinateCalculator()
savedimage = cc.loadSavedVideo('C:\Users\HP\Documents\Research\Tim#2\section_recording_17_8_15_14_54.avi')

        
        
