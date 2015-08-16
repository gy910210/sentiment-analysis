from utils.feature_select import select_feature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import bsr_matrix
import numpy as np
from utils.TextPreprocess import review_to_words
import pandas as pd
from sklearn.svm import SVC
from utils.sample_data import sample


# read training data ...
print "read data ..."
train = pd.read_csv("data\\labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3, error_bad_lines=False)
num_reviews_train = train["review"].size

clean_train_reviews = []
for i in xrange(0, num_reviews_train):
    clean_train_reviews.append(review_to_words(train["review"][i]))

test = pd.read_csv("data\\testData.tsv", header = 0, delimiter = "\t", quoting = 3)
num_reviews_test = len(test["review"])
clean_test_reviews = []

for i in xrange(0, num_reviews_test):
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)


train_data_features_d2v = []
test_data_features_d2v = []

sentence_vector = open("data\\sentence_vectors_org.txt", 'r')
index = 1
for line in sentence_vector:
    line = line.strip()
    strs = line.split(' ')[1:]
    vector = [float(item) for item in strs]
    if index == 1:
        print vector
    
    if index > 50000:
        break
    elif index <= 25000:
        train_data_features_d2v.append(vector)
    else:
        test_data_features_d2v.append(vector)
    index += 1

lab_fea = select_feature('data\\feature_chi.txt', 19000)["1"]

result = [0.0 for i in xrange(num_reviews_test)]

max_iter = 5
for epoch in xrange(max_iter):
    print "epoch: " + str(epoch)
    
    l1_train_bow, l1_train_d2v, l2_train_bow, l2_train_d2v, l1_label, l2_label = sample(clean_train_reviews, train_data_features_d2v, 
                                                    list(train["sentiment"]))
    
    # train logistic regression ...
    print "training bow ..."
    vectorizer_bow = TfidfVectorizer(analyzer = "word",
                                     tokenizer = None,
                                     preprocessor = None,
                                     stop_words = None,
                                     vocabulary = lab_fea,
                                     max_features = 19000)
    
    l1_train_features_bow = vectorizer_bow.fit_transform(l1_train_bow)
    l1_train_features_bow = bsr_matrix(l1_train_features_bow)
    
    l1_lr_bow = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None) 
    l1_lr_bow = l1_lr_bow.fit(l1_train_features_bow, l1_label)
    
    l2_test_features_bow = vectorizer_bow.transform(l2_train_bow)
    l2_test_features_bow = bsr_matrix(l2_test_features_bow)
    
    l2_result_bow = l1_lr_bow.predict_proba(l2_test_features_bow)[:,1]
    
    
    print "train doc2vec ..."
    
    l1_train_features_d2v = bsr_matrix(l1_train_d2v)
    l2_test_features_d2v = bsr_matrix(l2_train_d2v)
    
    l1_svm_d2v = SVC(C = 1.0, kernel='rbf', gamma = 0, probability=True)
    l1_svm_d2v = l1_svm_d2v.fit(l1_train_features_d2v, l1_label)
    
    l2_result_d2v = l1_svm_d2v.predict_proba(l2_test_features_d2v)[:,1]
    
    print "train ensemble ..."
    
    train_data_features_ens = []
    
    for i in xrange(len(l2_result_bow)):
        vector = []
        vector.append(l2_result_bow[i])
        vector.append(l2_result_d2v[i])
        
        train_data_features_ens.append(vector)
    
    train_data_features_ens = np.asarray(train_data_features_ens)
    
    lr_ens = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
    lr_ens = lr_ens.fit(train_data_features_ens, l2_label)
    
    print "final predict ..."
    train_bow = vectorizer_bow.fit_transform(clean_train_reviews)
    train_bow = bsr_matrix(train_bow)
    
    test_bow = vectorizer_bow.transform(clean_test_reviews)
    test_bow = bsr_matrix(test_bow)
    
    train_d2v = bsr_matrix(train_data_features_d2v)
    test_d2v = bsr_matrix(test_data_features_d2v)
    
    lr_bow = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
    lr_bow = lr_bow.fit(train_bow, list(train["sentiment"]))
    
    svm_d2v = SVC(C = 1.0, kernel='rbf', gamma = 0, probability=True)
    svm_d2v = svm_d2v.fit(train_d2v, list(train["sentiment"]))
    
    result_bow = lr_bow.predict_proba(test_bow)[:,1]
    result_d2v = svm_d2v.predict_proba(test_d2v)[:,1]
    
    test_data_features_ens = []
    
    for i in xrange(len(result_bow)):
        vector = []
        vector.append(result_bow[i])
        vector.append(result_d2v[i])
        
        test_data_features_ens.append(vector)
    
    test_data_features_ens = np.asarray(test_data_features_ens)
    
    result_test_ens = lr_ens.predict_proba(test_data_features_ens)[:,1]
    
    for i in xrange(num_reviews_test):
        result[i] += result_test_ens[i]

for i in xrange(num_reviews_test):
    result[i] /= max_iter

print "output..."
out = open("result\\ensemble_final.csv", 'w')
out.write("\"id\"" + "," + "\"sentiment\"")
out.write("\n")
for i, key in enumerate(list(test["id"])):
    out.write(str(key) + "," + str(result[i]) + "\n")
out.close()


