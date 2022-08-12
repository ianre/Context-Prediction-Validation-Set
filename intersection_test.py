from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
from scipy import ndimage


def isIntersecting(label1, label2):
    pass
def main():
    '''
    A = [1,0,0,0,1]
    B = [0,0,0,1,0]
    print("",A,"\n",B,np.intersect1d(A,B))
    return
    '''
    

    task = "Knot_Tying"
    I = Iterator(task)
    I.generateMasks();

class Iterator:

    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")
        self.gdir = os.path.join(self.CWD, task,"deeplab_grasper_v1")
        self.tdir = os.path.join(self.CWD, task,"deeplab_thread_v1")
        self.ndir = os.path.join(self.CWD, task,"deeplab_needle_v1")
        self.rdir = os.path.join(self.CWD, task,"deeplab_ring_v1")

    def imageToNPY(self,file):
        fileArr = file.split(".")
        temp = "".join(fileArr[:-1]) + ".npy"
        return temp.replace(".npy","_gt_pred.npy") 

    def generateMasks(self):
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue
                #if "Suturing_S02_T05" not in os.path.basename(root):
                #    continue
                print("Proc:", os.path.basename(root),file+".txt" )

                '''
                If we replace "images" by "labels" then the image source should be the same as the label source, 
                which is the same as the output destination
                '''
                imageRoot = root
                npyRoot = root.replace("images","deeplab_grasper_v1")
                threadRoot = root.replace("images","deeplab_thread_v1")
                outputRoot = root.replace("images","deeplab_masks")

                imageSource = os.path.join(imageRoot, file)
                npySource = os.path.join(npyRoot, self.imageToNPY(file))
                threadSource = os.path.join(threadRoot, self.imageToNPY(file))
                outputDest = os.path.join(outputRoot, file)
                if(not os.path.isdir(outputRoot)):
                    path = pathlib.Path(outputRoot)
                    path.mkdir(parents=True, exist_ok=True)

                (x_p,y_p), inter = self.getIntersection(npySource, threadSource ,outputDest,imageSource)
                print((x_p,y_p), inter)
                #if(not os.path.isdir(outputRoot)):
                #    path = pathlib.Path(outputRoot)
                #    path.mkdir(parents=True, exist_ok=True)

                #if os.path.exists(outputDest):
                #    os.remove(outputDest)
                

                count += 1
                
        print(count,"images processed!")

    def getIntersection(self, grasperSource,threadSource, outputMask, imagePath):
        [grasper_gt,grasper] = np.load(grasperSource,allow_pickle=True)
        [thread_gt,thread] = np.load(threadSource,allow_pickle=True)
        #image = cv2.imread(imagePath)
        grasper[grasper>0.95]=1 #! instead of 0.97
        grasper[grasper<0.95]=0 #! instead of 0.97

        thread[thread>0.95]=1 
        thread[thread<0.95]=0 
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = ndimage.center_of_mass(grasper)
        #drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter = self.isIntersecting2(grasper,thread)
        print(inter)
        return (y_center,x_center), inter

    def savePredLite(self, grasperSource,threadSource, outputMask, imagePath):
        [grasper_gt,grasper] = np.load(grasperSource,allow_pickle=True)
        [thread_gt,thread] = np.load(threadSource,allow_pickle=True)
        image = cv2.imread(imagePath)
        grasper[grasper>0.95]=1 #! instead of 0.97
        grasper[grasper<0.95]=0 #! instead of 0.97

        thread[thread>0.95]=1 
        thread[thread<0.95]=0 

        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        #plt.figure(figsize=(136,240))

        fig, ax = plt.subplots( nrows=1, ncols=1 )
        #plt.subplot(121)
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),interpolation=None) # I would add 
        #plt.imshow(pred, cmap='hot', alpha=0.8,interpolation=None)
        #plt.subplot(122)
        ax.imshow(grasper,cmap='hot', alpha=0.2,interpolation=None) # I would add 
        ax.imshow(thread, cmap='hot', alpha=0.2,interpolation=None)
        (x_center, y_center) = ndimage.center_of_mass(grasper)
        #x_center, y_center = np.argwhere(grasper==1).mean()
        #xis, yis = np.nonzero(grasper)
        #x_center, y_center= xis.mean(), yis.mean()

        #ax.Circle((center_x,center_y),radius=50,color='red') 
        drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        ax.add_patch(drawObject)

        inter = self.isIntersecting2(grasper,thread)
        print(inter)
        #inter_s = ', '.join(['%.2f']*len(inter))
        #print(inters)
        inter_msg = ""
        if(inter):
            inter_msg = "Intersection: True"
        else:
            inter_msg = "Intersection: False"
        ax.text(50,50,inter_msg,fontsize=30,color='red')
        #plt.imshow(gt.numpy(), cmap='hot', alpha=0.2,interpolation=None)
        fig.savefig(outputMask) 
        plt.close(fig)

    def isIntersecting2(self,maskA,maskB):
        rows = len(maskA)
        cols = len(maskA[0])
        for i in range(rows):
            for j in range(cols):
                if(maskA[i][j] == 1 and maskB[i][j] == 1):
                    return True
        return False

    def isIntersecting(self, maskA, maskB):
        return np.intersect1d(maskA,maskB)

    def savePred(self, sourceMask, outputMask, imagePath):
        [gt,pred] = np.load(sourceMask,allow_pickle=True)
        image = cv2.imread(imagePath)
        pred[pred>0.95]=1 #! instead of 0.97
        pred[pred<0.95]=0 #! instead of 0.97
        gt,pred = np.squeeze(gt),np.squeeze(pred)
        plt.figure(figsize=(136,240))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),interpolation=None) # I would add 
        plt.imshow(pred, cmap='hot', alpha=0.8,interpolation=None)
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB),interpolation=None) # I would add 
        plt.imshow(gt.numpy(), cmap='hot', alpha=0.2,interpolation=None)
        plt.savefig(outputMask)


    def refInter():
        
        CWD = os.path.dirname(os.path.realpath(__file__))      
        KT_pred_label1 = os.path.join(CWD,"Knot_Tying","deeplab_grasper_v1","Knot_Tying_S03_T02","frame_0737_gt_pred.npy")
        KT_pred_label2 = os.path.join(CWD,"Knot_Tying","deeplab_grasper_v1","Knot_Tying_S03_T02","frame_0737_gt_pred.npy")
        ImagePath = os.path.join(CWD,"Knot_Tying","images","Knot_Tying_S03_T02","frame_0737.png")

        [gt,pred] = np.load(KT_pred_label1,allow_pickle=True)
        image = cv2.imread(ImagePath)
        pred[pred>0.95]=1 #! instead of 0.97
        pred[pred<0.95]=0 #! instead of 0.97
        gt,pred = np.squeeze(gt),np.squeeze(pred)
        plt.figure(figsize=(136,240))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),interpolation=None) # I would add 
        plt.imshow(pred, cmap='hot', alpha=0.3,interpolation=None)
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB),interpolation=None) # I would add 
        plt.imshow(gt.numpy(), cmap='hot', alpha=0.3,interpolation=None)

        plt.show()
main()