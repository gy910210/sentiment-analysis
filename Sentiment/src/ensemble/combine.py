import pandas as pd

doc2vec_result = pd.read_csv("result\\doc2vec.csv", header = 0)
bow_result = pd.read_csv("result\\BOW_chi_tfidf.csv", header = 0)

num_reviews = bow_result["id"].size

out = open("result\\ensemble_combine.csv", 'w')
out.write("\"id\"" + "," + "\"sentiment\"")
out.write("\n")

for i in xrange(num_reviews):
    score = (doc2vec_result["sentiment"][i] + bow_result["sentiment"][i]) / 2.0
    out.write(doc2vec_result["id"][i] + "," + str(score) + "\n")
out.close()