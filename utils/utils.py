# import resource
# from bs4 import BeautifulSoup
# import urllib.request as urllib

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score
from tabulate import tabulate

# def get_resource_id(url):
#     soup = BeautifulSoup(urllib.urlopen(url), 'html.parser')
#     resource_div = soup.find_all(class_="clearfix resource_details gallery")
#     resource_id_html_string = str(resource_div[0]).split('\n')[0]
#     resource_div = BeautifulSoup(resource_id_html_string, 'html.parser')
#     return resource_div.div['ref']

def get_all_scores(actual, pred, labels):
    ((tp,fn),(fp,tn))= confusion_matrix(actual, pred, labels=labels)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fpr = fp/(fp+tn)

    acc = (tp+tn)/(tp+fp+fn+tn)
    gmean = geometric_mean_score(actual, pred, labels=labels)
    mcc = matthews_corrcoef(actual, pred)
    # f1 = f1_score(actual, pred, labels=labels)
    f1 = 2* precision * recall /(precision + recall)

    all_scores = [tp, tn, fp, fn, precision, recall, acc, f1, gmean, mcc, fpr]
    # print(all_scores)
    all_scores = [all_scores]
    all_scores_header = ['TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'Accuracy', 'F1-score', 'G-mean', 'MCC', 'FPR']
    return tabulate(all_scores, headers=all_scores_header)