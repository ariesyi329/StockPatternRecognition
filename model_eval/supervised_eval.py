import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def evaluate_matrix(true_y, predicted_y):
    
    #accuracy
    
    accu = accuracy_score(true_y, predicted_y)
    print 
    print "===========ACCURACY==========="
    print
    print "{0:.2%}".format(accu)
    
    #confusion matrix
    
    cm = confusion_matrix(true_y, predicted_y)
    label = ['F ','IT','C ','I ']
    print 
    print "===========Comfusion Matrix==========="
    print 
    print "    F  IT  C  I"
    
    for i in range(len(label)):
        print "{0} {1}".format(label[i], cm[i])
    
    #precision recall fscore
    
    prf = precision_recall_fscore_support(true_y, predicted_y)
    
    print
    print "===========Precision, Recall, F-score==========="
    print
    print "               F           IT         C          I "
    name = ['Precision', 'Recall   ', 'F-score  ']
    for i in range(len(name)):
        print "{0} {1}".format(name[i], prf[i])
        
    overall_prf = precision_recall_fscore_support(true_y, predicted_y,average='macro')
    print
    print "Average precision: {0:.2%}, recall: {1:.2%}, F-score: {2:.2%}".format(overall_prf[0],
                                                                                 overall_prf[1],
                                                                                 overall_prf[2])
    

