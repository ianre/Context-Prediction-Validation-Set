from asyncio import Task
import os, sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import krippendorff
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pkg_resources import invalid_marker
import statistics
from numpy.linalg import matrix_power


global unique_labels
unique_labels = {}

global invalid_States
invalid_States = {}

from scipy.spatial import distance

'''
This program combines labels in "kay", "ian", and "volunteer" folders into three types of consensus data:

"output" contains the consensus labels
"match" contains a comparison between all labels as well as kappa for all 4 sets (kay vs. ian vs. volunteer) and (kay vs. ian vs. volunteer vs. consensus)
"kappa" 
'''

mathematicaColors = {
    "blue":"#5E81B5",
    "orange":"#E09C24",
    "red":"#EA5536",
    "purple":"#A5609D",
    "green":"#8FB131",
}
# list of colors for the annotations
colors =["#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D"]
# opacity of the annotation masks. Values range from (0 to 255) Type tuple
opacity = (180,)
# radius of keypoint
radius = 4

label_classes = {}

global class_num
class_num = 0

def main(): 
    
    dir=os.getcwd()
    '''
    try:
        task=sys.argv[1]
        #print(task)
    except:
        print("Error: no task provided \nUsage: python draw_labels.py <task>")
        available_tasks = next(os.walk(os.path.join(dir, "images")))[1]
        print("Available task images: ", available_tasks)
        available_labels = next(os.walk(os.path.join(dir, "labels")))[1]
        print("Available task labels: ", available_labels)
        sys.exit()
    '''

    #print(task)
    task = "Knot_Tying" #then suturing to b done w/ jigs
    I = Iterator(task)
    #I.fixStates()
    #I.poll()
    #I.verifyOutput()
    #I.showAllPlots()


    #I.poll()
    #I.metrics()
    I.IOU()
    quit(); 

class Iterator:
    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.outputDir = os.path.join(self.CWD, self.task,"ctx_output")
        self.ian = os.path.join(self.CWD,self.task, "ctx_ian")
        self.kay = os.path.join(self.CWD,self.task, "ctx_kay")
        self.pred = os.path.join(self.CWD, self.task,"vis_context_labels_deeplab")
        self.consensus = os.path.join(self.CWD, self.task,"ctx_consensus")    
        self.surgeon =      os.path.join(self.CWD, self.task,"ctx_surgeon")    
        self.alpha = os.path.join(self.CWD, self.task,"k_alpha")

    def unrollContext(self, lines):
        n_lines = []
        start = lines[0].split(" ")[0] 
        start = int(start)
        MAX = lines[-1].split(" ")[0]
        #MAX = int(MAX)
        if(start > 0):
            for j in range(0, start):
                n_lines.append(str(j)+ " 0 0 0 0 0")


        for i in range(0,len(lines)-1):
            n_lines.append(lines[i])
            currentIndex = lines[i].split(" ")[0]
            currentIndex = int(currentIndex)
            nextIndex = lines[i+1].split(" ")[0]
            nextIndex = int(nextIndex)
            for k in range(currentIndex+1,nextIndex):
                n_lines.append(str(k) + " " + " ".join(lines[i].split(" ")[1:] ))
        
        n_lines.append(lines[len(lines)-1])
        return n_lines

        
    def poll(self):
        # get a filename from Kay's set:
        count = 0
        for root, dirs, files in os.walk(self.kay):
            for file in files:
                if("99"  in file):
                    continue
                #print(file)
                count=count+1

                kay_file =  os.path.join(self.kay, file)
                ian_file =  os.path.join(self.ian, file)
                out_file = os.path.join(self.outputDir, file)

                pred_file = os.path.join(self.pred, file)
                consensus_file = os.path.join(self.consensus, file)
                alpha_file = os.path.join(self.alpha, file)
                
                print("\tFile: ",file)

                kay_lines = []
                with open(kay_file) as kay_data:
                    for line in kay_data:
                        kay_lines.append(line.strip())
                
                ian_lines = []
                with open(ian_file) as ian_data:
                    for line in ian_data:
                        ian_lines.append(line.strip())
                
                pred_lines = []
                with open(pred_file) as pred_data:
                    for line in pred_data:
                        pred_lines.append(line.strip())

                consensus_lines = []
                with open(consensus_file) as consensus_data:
                    for line in consensus_data:
                        consensus_lines.append(line.strip())
                

                kay_lines_u = self.unrollContext(kay_lines)
                ian_lines_u = self.unrollContext(ian_lines)
                pred_lines_u = self.unrollContext(pred_lines)                
                consensus_lines_u = self.unrollContext(consensus_lines)
                print("\tkay",len(kay_lines_u),"ian",len(ian_lines_u),"pred",len(pred_lines_u),"consensus",len(consensus_lines_u),)
                

                LinesMin = min(len(kay_lines_u),len(pred_lines_u))
                kay_lines_u = kay_lines_u[0:LinesMin]
                ian_lines_u = ian_lines_u[0:LinesMin]
                pred_lines_u = pred_lines_u[0:LinesMin]
                consensus_lines_u = pred_lines_u[0:LinesMin]


                out_lines = []
                alpha_lines = []

                out_lines.append("Annotation Consensus")
                alpha_lines.append("Frame Kay Ian Predicted Nominal Ordinal Interval Ratio Nominal Ordinal Interval Ratio")
                #               0 00000 00000 00000 00000 0 0 0 0
                #match_lines.append("Frame Consensus")

                i = 0                
                for line in kay_lines_u:
                    if i >= len(pred_lines_u) or i >= len(ian_lines):
                        continue
                    
                    line = line.replace("\n","")
                    ian_line = ian_lines_u[i].replace("\n","")
                    pred_line = pred_lines_u[i].replace("\n","")
                    kappaPollRow = self.pollLine_cohen_kappa(line, ian_line, pred_line,i,DEBUG_INFO=file+" "+str(1))
                    #probsPollRow = self.pollLine_probs(line, ian_line, v_line)
                    outPollRow = self.majority_voting(line, ian_line, pred_line,i)
                    out_lines.append(outPollRow)
                    alpha_lines.append(kappaPollRow)
                    #match_lines.append(probsPollRow)
                    i=i+1
                self.save(out_file,out_lines)
                self.save(alpha_file, alpha_lines)
               
        print(count,"files processed!")

    def IOU(self):
        global class_num
        # get a filename from Kay's set:
        count = 0
        intersections = []
        IOU_frame = []
        for root, dirs, files in os.walk(self.consensus): # self.surgeon
            for file in files:
                if("99"  in file):
                    continue
                #print(file)
                count=count+1
                out_file = os.path.join(self.outputDir, file)

                #! IOU Consensus vs Surgeon
                '''
                pred_file = os.path.join(self.consensus, file)
                consensus_file = os.path.join(self.surgeon, file)
                '''
                
                #! IOU Automated vs Consensus
                pred_file = os.path.join(self.pred, file)
                consensus_file = os.path.join(self.consensus, file)


                
                pred_lines = []
                with open(pred_file) as pred_data:
                    for line in pred_data:
                        pred_lines.append(line.strip())

                consensus_lines = []
                with open(consensus_file) as consensus_data:
                    for line in consensus_data:
                        consensus_lines.append(line.strip())
                

                pred_lines_u = self.unrollContext(pred_lines)                
                consensus_lines_u = self.unrollContext(consensus_lines)
                
                LinesMin = min(len(consensus_lines_u),len(pred_lines_u))
                pred_lines_u = pred_lines_u[0:LinesMin]
                consensus_lines_u = consensus_lines_u[0:LinesMin]

                pred_arr = []
                con_arr = []

                i = 0      
                frame_IOU = []        
                for line in pred_lines_u:
                    if i >= len(pred_lines_u) or i >= len(consensus_lines_u):
                        continue
                    #! boolean array for distance
                    
                    pred = line.replace("\n","")
                    consensus_line = consensus_lines_u[i].replace("\n","")

                    p_num =[int(x)+1 for x in pred.split(" ")[1:5]]
                    con_num = [int(x)+1 for x in consensus_line.split(" ")[1:5]]
                    frame_IOU.append(1-distance.jaccard(p_num,con_num))

                    p_l = "".join(pred.split(" ")[1:5])
                    con_l = "".join(consensus_line.split(" ")[1:5])
                    if(p_l in label_classes):
                        pred_arr.append(label_classes[p_l])
                    else:                        
                        label_classes[p_l] = class_num
                        pred_arr.append(class_num)
                        class_num = class_num+1

                    if(con_l in label_classes):
                        con_arr.append(label_classes[con_l])
                    else:
                        label_classes[con_l] = class_num
                        con_arr.append(class_num)
                        class_num = class_num+1
                    #kappaPollRow = self.pollLine_cohen_kappa(line, ian_line, pred_line,i,DEBUG_INFO=file+" "+str(1))                    
                    #outPollRow = self.majority_voting(line, ian_line, pred_line,i)

                    #out_lines.append(outPollRow)
                    #alpha_lines.append(kappaPollRow)
                    i=i+1
                dis = distance.jaccard(pred_arr,con_arr)
                intersections.append(1-dis)
                IOU_frame.append(statistics.mean(frame_IOU))
                print(file,dis)
                #self.save(out_file,out_lines)
                #self.save(alpha_file, alpha_lines)
        
        print("Frame level:",statistics.mean(intersections)) 
        print("Annotation level:",statistics.mean(IOU_frame))
        print(count,"files processed!")

    '''Usage
    def get_overlap_f1(self, overlap, bg_class):
        return utils.get_overlap_f1_colin(self.result, self.label,
                                          n_classes=self.class_num,
                                          overlap=overlap,
                                          bg_class=bg_class)

    f_scores = []
    for overlap in [0.1, 0.25, 0.5, 0.75]:
        f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
                                        n_classes=gesture_class_num, 
                                        bg_class=bg_class, 
                                        overlap=overlap))
    '''
        
    def segment_labels(self, Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
        Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
        return Yi_split

    def segment_intervals(self,Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
        intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
        return intervals

    def get_overlap_f1_colin(self,P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
        def overlap_(self,p,y, n_classes, bg_class, overlap):

            true_intervals = np.array(self.segment_intervals(y))
            true_labels = self.segment_labels(y)
            pred_intervals = np.array(self.segment_intervals(p))
            pred_labels = self.segment_labels(p)

            # Remove background labels
            if bg_class is not None:
                true_intervals = true_intervals[true_labels!=bg_class]
                true_labels = true_labels[true_labels!=bg_class]
                pred_intervals = pred_intervals[pred_labels!=bg_class]
                pred_labels = pred_labels[pred_labels!=bg_class]

            n_true = true_labels.shape[0]
            n_pred = pred_labels.shape[0]

            # We keep track of the per-class TPs, and FPs.
            # In the end we just sum over them though.
            TP = np.zeros(n_classes, np.float)
            FP = np.zeros(n_classes, np.float)
            true_used = np.zeros(n_true, np.float)

            for j in range(n_pred):
                # Compute IoU against all others
                intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
                union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
                IoU = (intersection / union)*(pred_labels[j]==true_labels)

                # Get the best scoring segment
                idx = IoU.argmax()

                # If the IoU is high enough and the true segment isn't already used
                # Then it is a true positive. Otherwise is it a false positive.
                if IoU[idx] >= overlap and not true_used[idx]:
                    TP[pred_labels[j]] += 1
                    true_used[idx] = 1
                else:
                    FP[pred_labels[j]] += 1


            TP = TP.sum()
            FP = FP.sum()
            # False negatives are any unused true segment (i.e. "miss")
            FN = n_true - true_used.sum()
            
            precision = TP / (TP+FP)
            recall = TP / (TP+FN)
            F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

            # If the prec+recall=0, it is a NaN. Set these to 0.
            F1 = np.nan_to_num(F1)

            return F1*100

        if type(P) == list:
            return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
        else:
            return overlap_(P, Y, n_classes, bg_class, overlap)



    def showAllPlots(self):
        plots = []
        saved_Files = []
        fileCount = 0;
        count = 0
        pdfCount = 0
        try:
            os.remove(self.task+"_"+str(pdfCount)+".pdf")
        except OSError:
            pass
        pp = PdfPages(self.task+"_"+str(pdfCount)+".pdf")
        for root, dirs, files in os.walk(self.kappa):
            for file in files:
                if("99" in file):
                    continue
                if(fileCount %15 == 0):
                    pp.close();
                    pdfCount = pdfCount +1
                    try:
                        os.remove(self.task+"_"+str(pdfCount)+".pdf")
                    except OSError:
                        pass
                    pp = PdfPages(self.task+"_"+str(pdfCount)+".pdf")

                kappa_file = os.path.join(self.kappa, file)
                #files.append(kappa_file)
                plot = self.getPlot(kappa_file,file)
                saved_Files.append(file)
                plots.append(plot)   
                #print(fileCount, "created plot for file:",saved_Files[fileCount])
                #fileCount=fileCount+1 
                pp.savefig(plot) 
                fileCount = fileCount + 1;           
            
        '''
        for p in plots:
            print("count",count)
            pp.savefig(p)
            try:
                pp.savefig(p)
                print("saved file:",saved_Files[count])
            except:
                print("something went wrong with graph", saved_Files[count])
            
            count=count+1;
        '''
        pp.close();
        return;

    def metrics(self):
        fileCount = 0
        pre = 0
        post = 0
        preAggregate = 0
        postAggregate = 0
        allSigma = 0;
        sigmas = []
        lineCount = 0;
        for root, dirs, files in os.walk(self.alpha):
            for file in files:
                if("99" in file):
                    continue 
                kappa_file = os.path.join(self.alpha, file)
                #print(len(ka))
                #plot = self.getPlot(kappa_file,file)
                l_c, sigma, preAvg, postAvg, preAgr, postAgr = self.getAverage(kappa_file,file)
                pre = pre+preAvg
                post= post+postAvg
                preAggregate=preAggregate+preAgr
                postAggregate=postAggregate+postAgr
                #allSigma = sigma+allSigma
                sigmas.append(sigma)
                print(file, str(preAvg), str(postAvg), str(sigma)) 
                fileCount = fileCount + 1; 
                lineCount=lineCount+l_c;

        sigma2 = sum(i*i for i in sigmas)
        sigma2 = sigma2 / (fileCount+1)
        sigma2 = math.sqrt(sigma2)
        print(self.task)
        print( "Length:",str(lineCount))
        print( "Pre Unweight:",str( pre / (fileCount+1.0)))
        print( "Post Unweight:",str( post  / (fileCount+1.0)))
        print( "Sigma:",str( sigma2 ))
        print( "Pre Weight:",str( preAggregate / lineCount ))
        print( "Post Weight:",str(  postAggregate / lineCount ))

        return;
    
    def getAverage(self,file,fileName,x_col=0, y_col=5,y_c_col=9):
        preAvg = 0
        postAvg = 0
        preAgr = 0
        postAgr = 0

        f = open(file)
        lines=f.readlines()        
        x, y, y_c= [], [], []
        lineCount = 0;
        for line in lines:
            try:
                x.append(float(line.split(" ")[x_col]))
                y.append(float(line.split(" ")[y_col]))
                y_c.append(float(line.split(" ")[y_c_col]))
            except:
                pass;
            lineCount=lineCount+1
        f.close()
        
        for yy in y:
            preAgr = preAgr + yy
        for y_c_y_c in y_c:
            postAgr = postAgr + y_c_y_c
        
        #postAgr = postAgr / lineCount;
        #preAgr = preAgr / lineCount;        

        preAvg = statistics.mean(y);
        postAvg = statistics.mean(y_c)
        sigma = statistics.stdev(y);
        
        return lineCount, sigma, preAvg, postAvg, preAgr, postAgr

    def getPlot(self,file,fileName,x_col=0, y_col=5,y_c_col=9):
        f = open(file)
        lines=f.readlines()        
        x, y, y_c= [], [], []
        for line in lines:
            try:
                x.append(float(line.split(" ")[x_col]))
                y.append(float(line.split(" ")[y_col]))
                y_c.append(float(line.split(" ")[y_c_col]))
            except:
                pass;
        f.close()
        #print(x, y)
        fig = plt.figure()
        #fig = plt.plot(x,y)
        #fName = fileName.split["."][0]
        plt.title(fileName)
        plt.plot(x,y)
        plt.plot(x,y_c)
        plt.xlabel("Frames")
        plt.ylabel("Krippendorff Kappa")
        #plt.show();
        return fig

    def showPlot(self, file, x_col=0, y_col=5):
        f = open(file)
        lines=f.readlines()        
        x, y = [], []
        for line in lines:
            try:
                x.append(float(line.split(" ")[x_col]))
                y.append(float(line.split(" ")[y_col]))
            except:
                pass;
        f.close()
        print(x, y)
        plt.plot(x,y)
        #plt.show()        
    def graphs(self):
        count = 0
        for root, dirs, files in os.walk(self.kay):
            for file in files:
                kappa_file = os.path.join(self.kappa, file)
                self.showPlot(kappa_file)

    def takeAwayAngles(self, lines):
        i =0
        for line in lines:
            x = line.split(" ")
            y = " ".join(x[0:-2])
            lines[i] = y
            i = i + 1
        return lines;
    
    def getListOfInts(self, line):
        stateNums = []
        for state in line:
            stateNums.append(int(state) +1)
        return stateNums
    
    def getListOfInts_no_plus_1_offset(self, line):
        stateNums = []
        for state in line:
            stateNums.append(int(state))
        return stateNums
    
    def lineToStr(self, line):
        l_s = line.replace("\n","").split(" ")
        return "".join(l_s[1:])

    def getK_Kappa(self, arr, type,ZERO_ROW):
        kappa = self.rawK_Kappa(arr, type)
        if(kappa ==0 & ZERO_ROW):
            return " 1.0"
        else:
            return " " + "{:.2f}".format(kappa).lstrip('0');  
        
    def rawK_Kappa(self, arr, type):
        res = 0;
        try:
            res = krippendorff.alpha(reliability_data=arr,level_of_measurement=type)
        except:
            pass 
        return res
    
    def testZeroRow(self, o_s):
        for o in o_s:
            if( o != 0):
                return False
        return True

    def pollLine_cohen_kappa(self, k_line, i_line, v_line,lineNumber, DEBUG=True,DEBUG_INFO="none", ):
        # X_S means array of "S"trings from x's line
        k_s = k_line.split(" ")
        i_s = i_line.split(" ")
        v_s = v_line.split(" ")
        o_s = k_line.split(" ")
        probs = k_line.split(" ")
        line = ""
        # to do math
        for i in range (1,len(k_s)):
            self.checkUniqueness(k_s[i], i_s[i], v_s[i],lineNumber);
            
            candidate = self.majority(k_s[i], i_s[i], v_s[i]);
            o_s[i] = candidate

        if(DEBUG):
            line = k_s[0] + " " + "".join(k_s[1:])            
            line = line + " "+self.lineToStr(i_line)+" "+self.lineToStr(v_line)+" "+"".join(o_s[1:])      #+ str(res)
        else:
            line = " ".join(k_s)

        k_n = self.getListOfInts_no_plus_1_offset(k_s[1:])
        i_n = self.getListOfInts_no_plus_1_offset(i_s[1:])
        v_n = self.getListOfInts_no_plus_1_offset(v_s[1:])
        o_n = self.getListOfInts_no_plus_1_offset(o_s[1:])
        arr = [k_n,i_n,v_n]
        ZERO_ROW = self.testZeroRow(o_n)
        line = line + self.getK_Kappa(arr, "nominal",ZERO_ROW)
        line = line + self.getK_Kappa(arr, "ordinal",ZERO_ROW)
        line = line + self.getK_Kappa(arr, "interval",ZERO_ROW)
        line = line + self.getK_Kappa(arr, "ratio",ZERO_ROW)
        arr = [k_n,i_n,v_n,o_n]
        line = line + self.getK_Kappa(arr, "nominal",ZERO_ROW)
        line = line + self.getK_Kappa(arr, "ordinal",ZERO_ROW)
        line = line + self.getK_Kappa(arr, "interval",ZERO_ROW)
        line = line + self.getK_Kappa(arr, "ratio",ZERO_ROW)
       
        return line

    def pollLine_output(self, k_line, i_line, v_line,DEBUG=True):
        k_s = k_line.split(" ")
        i_s = i_line.split(" ")
        v_s = v_line.split(" ")
        o_s = k_line.split(" ")
        probs = k_line.split(" ")
        accepted = 0;
        for i in range (1,len(k_s)):
            try: 
                prob = (int(k_s[i]) + int(i_s[i]) + int(v_s[i])) /3.0;
            except:
                print("Error parsing bit ",i, " in line ", " in file ")
                continue
            prob = (int(k_s[i]) + int(i_s[i]) + int(v_s[i])) /3.0;
            if( prob.is_integer() ):
                probs[i] = str(int(prob))
            else:
                probs[i] = "{:.1f}".format(prob).lstrip('0')
            if( prob >=0.5 ):

                o_s[i] = "1";
            else:
                o_s[i] ="0";  
        line = " ".join(o_s)
        return line
        #line = line +"\t"+ " ".join(probs)       
    def checkUniqueness(self, k, i, v, line):
        global unique_labels
        if( k in unique_labels.keys()):
            pass
        else:
            unique_labels[k] = line;
        if( i in unique_labels.keys()):
            pass
        else:
            unique_labels[i] = line;
        if( v in unique_labels.keys()):
            pass
        else:
            unique_labels[v] = line;
        #print("uniue",unique_labels);     
   
    def majority(self, k, i, v):
        
        k_n = int(k)
        i_n = int(i)
        v_n = int(v)
        arr = [k_n,i_n,v_n]
        candidate = self.Most_Common(arr);

        return str(int(candidate))
    
    def Most_Common(self, lst):
        data = Counter(lst)
        obj = data.most_common(1)
        return data.most_common(1)[0][0]
        
    def majority_voting(self, k_line, i_line, v_line,lineNumber):     
        k_s = k_line.split(" ")
        i_s = i_line.split(" ")
        v_s = v_line.split(" ")
        o_s = k_line.split(" ") #output
        for i in range (1,len(k_s)):
            self.checkUniqueness(k_s[i], i_s[i], v_s[i],lineNumber);
            candidate = self.majority(k_s[i], i_s[i], v_s[i]);
            o_s[i] = candidate            
        line = " ".join(o_s)
        return line
    
    def pollLine_probs(self, k_line, i_line, v_line):
        k_s = k_line.split(" ")
        i_s = i_line.split(" ")
        v_s = v_line.split(" ")
        o_s = k_line.split(" ")
        probs = k_line.split(" ")
        accepted = 0;

        for i in range (1,len(k_s)):
            try: 
                prob = (int(k_s[i]) + int(i_s[i]) + int(v_s[i])) /3.0;
            except:
                print("Error parsing bit ",i, " in line ", " in file ")
                continue
            prob = (int(k_s[i]) + int(i_s[i]) + int(v_s[i])) /3.0;
            if( prob.is_integer() ):
                probs[i] = str(int(prob))
            else:
                probs[i] = "{:.1f}".format(prob).lstrip('0')
            if( prob >=0.5 ):

                o_s[i] = "1";
            else:
                o_s[i] ="0";  
        line = " ".join(o_s)
        line = line +"\t"+ " ".join(probs)
        return line
    
    def save(self, x_file, x_lines):
        with open(x_file, 'w+') as f:
            for item in x_lines[1:]:
                f.write("%s\n" % item)      


main();

