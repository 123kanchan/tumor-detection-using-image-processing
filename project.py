from tkinter import filedialog
from tkinter import *
import tkinter as tk
import PIL
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageTk
from predictTumor import *
class Window(Frame):
     
     def __init__(self,master=None):
         Frame.__init__(self,master)
         
         self.master=master 
         self.init_Window()
     def init_Window(self):
         self.master.title("Brain Tumor Detection")
         
         takeImg=tk.Button(root,text="BrowseImages",command=self.BrowseImages, fg="red",bg="white",width=20,height=1,activebackground="Red",font=('times',15,'bold'))
         takeImg.place(x=500,y=100)
         area=tk.Button(root,text="Result",command=self.predict, fg="red",bg="white",width=20,height=1,activebackground="Red",font=('times',15,'bold'))
         area.place(x=1000,y=600)  

     def BrowseImages(self):
          global panelA, panelB,panelC,panelD,panelE,panelF,panelG
          global mrimg
          self.path = filedialog.askopenfilename()
    
          if len(self.path)>0:
              image=Image.open(self.path)
              img=str(self.path)
              img=cv2.imread(img,1)
              mrimg=img
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #CHECK
              ret, thresh = cv2.threshold(gray,120,255,cv2.THRESH_TOZERO)
              kernel = np.ones((3,3),np.uint8)
              opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
              sure_bg = cv2.dilate(opening,kernel,iterations=3)
              sure_fg=cv2.erode(opening,kernel,iterations=3)
        
              unknown = cv2.subtract(sure_bg,sure_fg)
         # Marker labelling
              ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
              markers = markers+1

# Now, mark the region of unknown with zero
              markers[unknown==255] = 0

              markers=markers.astype('int32')#change
              img = cv2.imread(self.path)

              markers = cv2.watershed(img,markers)
              img[markers == -1] = [255,0,0]
              tumorimage=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
              tumorimage = cv2.cvtColor(tumorimage, cv2.COLOR_BGR2RGB) #CHECK
         
              img = Image.fromarray(img)
              thresh=Image.fromarray(thresh)
              opening=Image.fromarray(opening)
              sure_bg = Image.fromarray(sure_bg)
              sure_fg = Image.fromarray(sure_fg)
              unknown = Image.fromarray(unknown)
              tumorimage= Image.fromarray(tumorimage)
         
              img=img.resize((160, 160), Image.ANTIALIAS)
              thresh=thresh.resize((160, 160), Image.ANTIALIAS)
              opening=opening.resize((160, 160), Image.ANTIALIAS)
              sure_bg=sure_bg.resize((160,160),Image.ANTIALIAS)
              sure_fg=sure_fg.resize((160,160),Image.ANTIALIAS)
              unknown=unknown.resize((160,160),Image.ANTIALIAS)
              tumorimage=tumorimage.resize((160,160),Image.ANTIALIAS)
         
              img = ImageTk.PhotoImage(img)
              thresh = ImageTk.PhotoImage(thresh)
              opening= ImageTk.PhotoImage(opening)
              sure_bg= ImageTk.PhotoImage(sure_bg)
              sure_fg= ImageTk.PhotoImage(sure_fg)
              unknown= ImageTk.PhotoImage(unknown)
              tumorimage=ImageTk.PhotoImage(tumorimage)
              if panelA is None or panelB is None  or panelC is None or panelD is None or panelE is None or panelF is None or panelG is None:
			# the first panel will store our original image
                panelA = Label(image=img)
                panelA.image = img
                panelA.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Original",font=('times',20,'italic bold underline'))
                message.place(x=30,y=500) 
            
                panelB = Label(image=thresh)
                panelB.image = thresh
                panelB.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Threshold",font=('times',20,'italic bold underline'))
                message.place(x=200,y=500) 
            
                panelC = Label(image=opening)
                panelC.image = opening
                panelC.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Morphology",font=('times',20,'italic bold underline'))
                message.place(x=380,y=500) 
            
                panelD = Label(image=sure_bg)
                panelD.image = sure_bg
                panelD.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Background",font=('times',20,'italic bold underline'))
                message.place(x=580,y=500) 
            
                panelE = Label(image=sure_fg)
                panelE.image = sure_fg
                panelE.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Foreground",font=('times',20,'italic bold underline'))
                message.place(x=760,y=500) 
            
                panelF = Label(image=tumorimage)
                panelF.image = tumorimage
                panelF.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Marker",font=('times',20,'italic bold underline'))
                message.place(x=960,y=500) 
            
            
                panelG = Label(image=unknown)
                panelG.image = unknown
                panelG.pack(side=LEFT,padx=10,pady=15)
                message=tk.Label(root,text="Final",font=('times',20,'italic bold underline'))
                message.place(x=1140,y=500) 
            
              else:
            # update the pannels
                panelA.configure(image=img)
                panelB.configure(image=thresh)
                panelC.configure(image=opening)
                panelD.configure(image=sure_bg)
                panelE.configure(image=sure_fg)
                panelF.configure(image=tumorimage)
                panelG.configure(image=unknown)
                panelA.image = img
                panelB.image = thresh
                panelC.image = opening
                panelD.image = sure_bg
                panelE.image = sure_fg
                panelF.image = tumorimage
                panelG.image = unknown

     
     def predict(self):
           global mrimg
           res=predictTumor(mrimg)

           if res > 0.5:
                message=tk.Label(root,text="Detected",width=10,height=1,font=('times',30,'italic bold underline'))
                message.place(x=300,y=600) 
           else:
                message=tk.Label(root,text="Not Detected",width=10,height=1,font=('times',30,'italic bold underline'))
                message.place(x=300,y=600)       

        


root=Tk()
root.geometry("1280x720")
app=Window(root)
root.configure(background='gray')
panelA = None
panelB = None
panelC = None         
panelD = None
panelE = None
panelF = None        
panelG = None        

message=tk.Label(root,text="Brain Tumor Detection Using Image Processing",bg="Green",width=50,height=1,font=('times',30,'italic bold underline'))
message.place(x=100,y=20)    
   

root.mainloop()       