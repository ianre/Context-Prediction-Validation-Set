import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


CWD = os.path.dirname(os.path.realpath(__file__))      
KT_pred_label = os.path.join(CWD,"Knot_Tying","deeplab_grasper_v1","Knot_Tying_S03_T02","frame_0737_gt_pred.npy")
ImagePath = os.path.join(CWD,"Knot_Tying","images","Knot_Tying_S03_T02","frame_0737.png")

[gt,pred] = np.load(KT_pred_label,allow_pickle=True)
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