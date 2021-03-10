import numpy as np
import traceback
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------    
def convert_corr2sign(corr, ground_truth, threshold):
    a = np.multiply(corr, ground_truth)
    shape = a.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if a[i,j] > threshold:
                a[i,j] = 1
            elif a[i,j] < -threshold:
                a[i,j] = -1
            else:
                a[i,j] = 0
    return a
  
#-----------------------------------------------------------------------------------------------------    
def calcScore(estimation, ground_truth_sign, threshold, prediction):

    a = np.multiply(estimation, ground_truth_sign)
    b = np.where( a > threshold )
    c = np.where( abs(ground_truth_sign) > 0 )
    cnt = len(b[0])
    total = len(c[0])
    # print cnt, total, cnt/float(total)

    false = prediction - abs(ground_truth_sign)
    d = np.multiply(estimation, false)
    e = np.where( abs(d) > threshold )
    f = np.where( false > 0 )
    false_cnt = len(e[0])
    false_total = len(f[0])

    true_positive = cnt/float(total)
    false_positive = false_cnt/float(false_total)
    
    return true_positive, false_positive
#-----------------------------------------------------------------------------------------------------        
def calcScore2(estimation, ground_truth_sign, threshold, prediction):

    a = np.multiply(estimation, ground_truth_sign)
    b = np.where( a > threshold )
    c = np.where( abs(ground_truth_sign) > 0 )
    tp = len(b[0])
    total = len(c[0])
    fn = total - tp

    negative = prediction - abs(ground_truth_sign)
    d = np.multiply(estimation, negative)
    e = np.where( abs(d) > threshold )
    f = np.where( negative > 0 )
    fp = len(e[0])
    negative_total = len(f[0])
    tn = negative_total - fp

    # print(tp, fp, tn, fn)
    precision, recall, f1, accuracy = 0.0, 0.0, 0.0, 0.0
    try:
        precision = tp/float(tp + fp + np.finfo(float).eps)
        recall = tp/float(tp + fn + np.finfo(float).eps) # = true positive rate
        f1 = 2*precision*recall/(precision+recall+ np.finfo(float).eps) # harmonic mean of precision and recall
        accuracy = (tp + tn)/float(total + negative_total + np.finfo(float).eps)
    except BaseException as e:
        print("exception by %s " % (repr(e)))
        traceback.print_exc()
        pass

    return precision, recall, f1, accuracy, tp, fp, tn, fn
#-----------------------------------------------------------------------------------------------------        
def calcScore3(estimation, ground_truth_sign, threshold):

    cnt_total = 0
    cnt_true = 0
    for i in range(len(ground_truth_sign[:,0])):
        for j in range(len(ground_truth_sign[0,:])):
            if ground_truth_sign[i,j] != 0:
                cnt_total += 1                
                r = estimation[i,j]*ground_truth_sign[i,j] > threshold
                if r == True:
                    cnt_true += 1
    rate = cnt_true/float(cnt_total)
    return rate
#-----------------------------------------------------------------------------------------------------        
def calcScore4(estimation, pred_sign, threshold, miRNA_list, mRNA_list):

    a = np.multiply(estimation, pred_sign)

    
    result = {}
    for i, m in enumerate(mRNA_list):
        for j, mi in enumerate(miRNA_list):
            val = a[j,i]
            if val > threshold:
                est = estimation[j,i]
                if m not in result: #not result.has_key(m):
                    result[m] = []
                result[m].append((mi, est))
    return result
#-----------------------------------------------------------------------------------------------------        
def showCorr(estimation, corr, gt_mi, gt_m):
    point = []
    pcc = []
    est = []
    for mi, m in zip(gt_mi, gt_m):        
        point.append((mi,m))
        pcc.append(corr[mi, m])
        est.append(estimation[mi,m])
    x = range(len(point))
    
    plt.figure()
    plt.plot(x, pcc, 'r-', x, est, 'b-')
   
    plt.xlabel('miRNA/mRNA pair', fontsize=20)
    plt.ylabel('strength', fontsize=20)

    plt.grid(linestyle=':', linewidth='0.5', color='black')
    plt.xticks(x, point, rotation='vertical')
    plt.legend(['Ground truth', 'Tiresias'])
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.17)
    figname = 'corr.pdf'
    # plt.show()
    plt.savefig(figname)     
#-----------------------------------------------------------------------------------------------------        
def showCorr2(estimation, corr, gt_mi, gt_m, pred):
    point = []
    pcc = []
    est = []
    point2 = []
    pcc2 = []
    est2 = []

    for i in range(len(pred[:,0])):
        for j in range(len(pred[0,:])):
            if pred[i,j] == 1:
                if (i in gt_mi) and (j in gt_m):
                    point.append((i,j))
                    pcc.append(corr[i,j])
                    est.append(estimation[i,j])
                else:    
                    point2.append((i,j))
                    pcc2.append(corr[i,j])
                    est2.append(estimation[i,j])

    x = range(len(point+point2))
    
    plt.figure()
    plt.plot(x, pcc+pcc2, 'r-', x, est+est2, 'b-')
   
    plt.xlabel('miRNA/mRNA pair', fontsize=20)
    plt.ylabel('strength', fontsize=20)

    plt.grid(linestyle=':', linewidth='0.5', color='black')
    plt.xticks(x, point, rotation='vertical')
    plt.legend(['Ground truth', 'Tiresias'])
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.17)
    figname = 'corr2.pdf'
    # plt.show()
    plt.savefig(figname)   
#-----------------------------------------------------------------------------------------------------        
def corrDist(corr, gt_mi, gt_m):
    point = []
    pcc_gt = []
    pcc_none = []
    pcc_pos = []
    pcc_neg = []
    for mi, m in zip(gt_mi, gt_m):        
        point.append((mi,m))

    s = corr.shape
    
    for i in range(s[0]):
        for j in range(s[1]):
            if (i,j) in point:
                pcc_gt.append(corr[i, j])
                if corr[i, j]>0:
                    pcc_pos.append(corr[i, j])
                else:
                    pcc_neg.append(corr[i, j])
            else:
                pcc_none.append(corr[i, j])
    # print pcc_gt
    # print
    # print pcc_none
    # print(np.average(pcc_pos))
    # print(np.average(pcc_neg))
    # print(np.average(pcc_none))
    a = np.histogram(pcc_gt)
    # print(a)
    # plt.hist(a)

    # plt.show()
#-----------------------------------------------------------------------------------------------------        


if __name__ == '__main__':
    est = np.array([[1, -8],[3, 4]])
    gt = np.array([[0, 1],[1, 0]])
    pred = np.array([[1, 1],[1, 0]])
    th = 0.1
    calcScore(est, gt, th, pred)
    s = convert_corr2sign(est, gt, th)
    


