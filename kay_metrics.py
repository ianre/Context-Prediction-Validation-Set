# Kay Hutchinson 8/9/22
# Compare gesture transcripts in terms of acc, edit score, f1, iou (micro and macro)

import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Group all rows with the same MP and return as a new df as <start, end, MP>
def group(dfMP):
    # Find start and end indices of each group of rows with the same context label
    dfMP['subgroup'] = (dfMP['Y'] != dfMP['Y'].shift(1)).cumsum()
    print(dfMP)

    # Create 'subgroup' column indicating groups of consecutive rows with same MP label
    dfGrouped = dfMP.groupby('subgroup').apply(lambda x: (x['Sample'].iloc[0], x['Sample'].iloc[-1], x['Y'].iloc[0]))
    #print(dfGrouped)

    # Split list from lambda function into columns again
    dfGrouped = pd.DataFrame(dfGrouped.tolist(), index=dfGrouped.index)
    #print(dfGrouped)

    return dfGrouped

# Convert list of labels to a transcript (intermediate step uses dataframes)
def listToTranscript(list):
    dfMP = pd.DataFrame(list, columns=["Y"])
    dfMP.insert(0, 'Sample', range(0, len(list)))

    # Group MPs in a dataframe with start, end, and MP label
    #mps = group(dfMP)

    # convert MPs dataframe to list
    #transcript = mps.values.tolist()
    transcript = list

    return transcript

# Convert transcript to list of labels
def transcriptToList(transcript):
    list = []
    #print(transcript)
    # For each label, fill in the list with that label from start to end sample number
    for t in transcript:
        fill = [t[2]]*(int(t[1])-int(t[0])+1)
        list[int(t[0]):int(t[1])+1] = fill

    return list

# Convert transcript to sequence (one way conversion)
def transcriptToSequence(transcript):
    sequence = []
    for i in transcript:
        sequence.append(i[2])
    return sequence

# Read an MP transcript from a given file path
def readMPTranscript(filePath):
    # Read in file
    with open(filePath) as f:
        lines = f.readlines()
    # Drop header
    lines = lines[1:]

    # Reformat and take only the verb
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("(")[0]
        lines[i] = lines[i].split(" ")

    # Return MP transcript
    return lines


# Read a gesture transcript from a given file path
def readGTranscript(filePath):
    # Read in file
    with open(filePath) as f:
        lines = f.readlines()

    # Reformat and take only the verb
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split("(")[0]
        lines[i] = lines[i].split(" ")

    # Return gesture transcript
    return lines


# Calculate edit score
def levenstein_(p,y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)

    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score


# Visualize MP segments
def drawLists(list1, list2):
    # Cast list to df
    df1 = pd.DataFrame(list1, columns = ['labels'])
    df2 = pd.DataFrame(list2, columns = ['labels'])

    # Encode labels and add to df
    le = preprocessing.LabelEncoder()
    le.fit(df1['labels'])
    df1['encoded'] = le.transform(df1['labels'])

    le.fit(df2['labels'])
    df2['encoded'] = le.transform(df2['labels'])

    # Get list of classes from encoding
    le_mapping = list(le.classes_)

    # Look at sequences
    #print(transcriptToSequence(listToTranscript(list1)))
    #print(transcriptToSequence(listToTranscript(list2)))


    # Get color range for vmin and vmax in pcolorfast
    cmin = min(min(df1['encoded'].unique()), min(df2['encoded'].unique()))
    cmax = max(max(df1['encoded'].unique()), max(df2['encoded'].unique()))

    # Graph
    fig, axs = plt.subplots(2, 1)  #create two subplots

    # First plot
    ax = axs[0]
    c = ax.pcolorfast(df1.index[:], ax.get_ylim(), df1['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Left')

    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Primitive')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_yticklabels(le_mapping)

    # Second plot
    ax = axs[1]
    c = ax.pcolorfast(df2.index[:], ax.get_ylim(), df2['encoded'].values[np.newaxis], cmap='Set1', alpha=0.3, vmin=cmin, vmax=cmax)
    ax.set_title('Left (K)')

    # Labels and legend
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Primitive')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_yticklabels(le_mapping)

    font = {'size'   : 20}

    # Show plot
    fig.tight_layout()
    plt.show()


    '''
    # Old code from Zoey, also plots lines/levels for different labels
    plt.subplot(2, 1, 1)


    ax=df1['encoded'].plot(label='MP') #.apply(lambda x:x+6).plot(label = 'MP')

    print(df1['encoded'].values[np.newaxis])
    print(df2['encoded'].values[np.newaxis])

    #ax.set_xticks(np.arange(855, 855+len(table_process['Motion Primative_c'][855:2410]),150))
    #ax.set_xticklabels(np.arange(855//30,(855+len(table_process['Motion Primative_c'][855:2410]))//30,5))
    ax.pcolorfast(df1.index[:], ax.get_ylim(),df1['encoded'].values[np.newaxis],cmap='Set1', alpha=0.3)
    #ax.set_xlabel('Time (sec)',fontname='Arial')
    #ax.set_ylabel('Motion Primitive/Gesture',fontname='Arial',fontsize=20)
    #ax.set_yticks(np.arange(0,14))
    #ax.set_yticklabels(["Grasp","Pull","Push","Release","Touch","Untouch","G1","G11","G2",\
                        #"G3","G4","G5","G6","G8"])
    ax.legend()
    #ax.set_ylim([-0.2,13.2])
    #font = {'size'   : 20}

    #matplotlib.rc('font', **font)
    #fig.show()

    plt.subplot(2, 1, 2)
    ax=df2['encoded'].plot(label='MP') #.apply(lambda x:x+6).plot(label = 'MP')
    ax.pcolorfast(df2.index[:], ax.get_ylim(),df2['encoded'].values[np.newaxis],cmap='Set1', alpha=0.3)
    ax.legend()

    plt.show()
    '''
    #print(df)




# Lea's utils code
def segment_labels(Yi):   # returns an array of labels, basically seq
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

def segment_intervals(Yi):   # returns start and end indices for each segment
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals

def get_overlap_f1_colin(P, Y, n_classes=0, overlap=.1):
    def overlap_(p,y, n_classes, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]
        
        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float64)
        FP = np.zeros(n_classes, np.float64)
        true_used = np.zeros(n_true, np.float64)

        # Sum IoUs for each class
        IoUs = np.zeros(n_classes, np.float64)
        nIoUs = np.zeros(n_classes, np.float64)

        for j in range(n_pred):
            #print(n_pred[j])
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)
            
            #print("G"+str(pred_labels[j]+1))
            #if not true_used[idx]:   # might need this?
            IoUs[pred_labels[j]] += max(max(IoU), 0)
            nIoUs[pred_labels[j]] += 1

            # Get the best scoring segment (index)
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

        # Take average IoU of each class in this trial wrt number of correct labels
        classIoUs = (IoUs/nIoUs)
        classIoUs = np.nan_to_num(classIoUs)
        macroIoU = sum(classIoUs)/np.count_nonzero(nIoUs)

        # avg over trial:
        microIoU = np.sum(IoUs)/np.sum(nIoUs)
        #print(avgIoUs)
        #sys.exit()

        return F1*100, microIoU, macroIoU, nIoUs

    # if type(P) == list:
    #     return np.mean([overlap_(P[i],Y[i], n_classes, overlap) for i in range(len(P))])
    # else:
    return overlap_(P, Y, n_classes, overlap)



if __name__ == "__main__":

    # Get task from command line
    '''
    try:
        task=sys.argv[1]
        #print(task)
    except:
        print("Error: invalid task\nUsage: python context_to_gestures.py <task>\nTasks: Suturing, Needle_Passing, Knot_Tying, Peg_Transfer")
        sys.exit()
    ''' 
    task = "Knot_Tying"
    CWD = os.path.dirname(os.path.realpath(__file__))        
    task = task
    imagesDir = os.path.join(CWD, task,"images")
    predDir = os.path.join(CWD,task,"vis_context_labels_deeplab")
    consensus = os.path.join(CWD, task,"ctx_consensus")    


    baseDir = os.path.dirname(os.getcwd())
    # Transcript and video directories
    taskDir = os.path.join(baseDir, "Datasets", "dV", task)
    #gDir = os.path.join(taskDir, "gestures")
    #gccDir = os.path.join(taskDir, "gestures_consensus_context")
    gDir =  consensus
    gccDir = predDir


    # Counters
    tacc = 0
    tedit = 0
    tn = 0
    tf1 = 0
    tmicroiou = 0
    tmacroiou = 0
    tnious = 0

    bestacc = 0
    worstacc = 100
    bestedit = 0
    worstedit = 100

    # For each transcript
    for f in os.listdir(gDir): #[0:1]:
        print(f)
        # Paths to the transcripts to compare
        gPath = os.path.join(gDir, f)
        gccPath = os.path.join(gccDir, f)

        # Read in transcripts
        #linesg = readGTranscript(gPath)
        #linesgcc = readGTranscript(gccPath)

        linesg = []
        linesgcc = []

        gPathF = open(gPath,'r')
        #print gPathF.readlines()
        for i in gPathF.readlines():
            linesg.append(i)
            #print i
        gccPathF = open(gccPath,'r')
        #print gPathF.readlines()
        for i in gccPathF.readlines():
            linesgcc.append(i)
            #print i


        # Convert to lists
        listg = transcriptToList(linesg)
        listgcc = transcriptToList(linesgcc)
        #listg = []
        #listgcc = []
        

        #drawLists(listl, listk)
        #sys.exit()

        # Get accuracy
        acc = (np.mean([np.mean(listg[i]==listgcc[i]) for i in range(min(len(listg), len(listgcc)))])*100)
        #print(acc)

        # Get edit score
        linesgcc = listToTranscript(listgcc)  # combine consecutive identical gesture labels (increases F1 score)
        seqg = transcriptToSequence(linesg)
        seqgcc = transcriptToSequence(linesgcc)
        edit = (levenstein_(seqg, seqgcc, norm=True))
        #print(edit)

        tn = tn+1
        tacc = tacc+acc
        tedit = tedit+edit
        # print(acc)
        # print(edit)
        if acc > bestacc:
            bestacc = acc
        if edit > bestedit:
            bestedit = edit
        if acc < worstacc:
            worstacc = acc
        if edit < worstedit:
            worstedit = edit

        # Encode gesture labels to numbers by dropping the 'G' to use Lea's code
        # also subtract 1 from each label so it can be used for indexing
        listG = [int(listg[g][1:])-1 for g in range(len(listg))]
        listGCC =  [int(listgcc[g][1:])-1 for g in range(len(listgcc))]

        # Get number of unique classes in the labels 
        n_classes = max(max(np.unique(listG)), max(np.unique(listGCC)))+1
        
        # Get IoUs
        #ious = IOUs(listg, listgcc)
        # args: (P, Y, n_classes=0, overlap=.1)
        f1, microIoU, macroIoU, nious, = get_overlap_f1_colin(listGCC, listG, n_classes, 0.1)
        tf1 = tf1 + f1
        #print(ious)
        tmicroiou = tmicroiou + microIoU
        tmacroiou = tmacroiou + macroIoU
        tnious = tnious + nious

        #drawLists(listg, listgcc)

    print("Avg acc: " + str(tacc/tn))
    print("Avg edit: " + str(tedit/tn))
    print("Avg f1: " + str(tf1/tn))
    print("Avg micro iou: " + str(tmicroiou/tn))
    print("Avg macro iou: " + str(tmacroiou/tn))
    print("Best acc: " + str(bestacc) + " worst acc: " + str(worstacc))
    print("Best edit " + str(bestedit) + " worst edit: " + str(worstedit))

    