# sentiment-classification
Kaggle challenge Bag of words meets bags of popcorn. And ranked 61/579, with precision 0.96255.

# Method
My method contains three parts. One is learning a shallow model; the other is learning a deep model. And then I combine the two models to train an ensemble model. 

#### Shallow Model
The method involves a bag-of-words model, which represents the sentence or document by a vector of words. But due to the sentences have lots of noises, so I use a feature selection process. And chi-square statistic is adopted by me. This will result in a feature vector that is more relevant to the classification label. Then I use the TF-IDF score as each dimension of feature vector. Although I have selected the features, the dimension of feature vector is still very high (19000 features we use in our model). So I can use logistic regression to train the classification model. And I use L1 regularization. The process of training a shallow model is as following. And I call the mothed as BOW.

![](https://github.com/gy910210/sentiment-analysis/raw/master/pic/shallow model.PNG)

Why I call this model shallow? MaiInly because it adopts a bag-of-words based model, which only extracts the shallow words frequency of the sentence. But it will not involve the syntactic and semantic of the sentence. So I call it a shallow model. And I will introduce a deep model which can capture more meanings of sentence.

#### Deep Model
Recently, Le & Mikolov proposed an unsupervised method to learn distributed representations of words and paragraphs. The key idea is to learn a compact representation of a word or paragraph by predicting nearby words in a fixed context window. This captures co-occurrence statistics and it learns embedding of words and paragraphs that capture rich semantics. Synonym words and similar paragraphs often are surrounded by similar context, and therefore, they will be mapped into nearby feature vectors (and vice versa). We call the method as Doc2Vec. Doc2Vec is a neural network like method, but it contains no hidden layers. And Softmax layer is the output. To avoid the high time complexity of Softmax output layer, they propose hierarchical softmax based on Huffman tree. The architecture of the model is as follows.

![](https://github.com/gy910210/sentiment-analysis/raw/master/pic/doc2vec.PNG)

Such embeddings can then be used to represent sentences or paragraphs. And can be used as an input for a classifier. In my method, we first train a 200 dimensions paragraph vector. And then I adopt a SVM classifier with RBF kernel.
The process of training a shallow model is as following.

![](https://github.com/gy910210/sentiment-analysis/raw/master/pic/deep model.PNG)

#### Ensemble Model
The ensemble model will involve the above two method (BOW and Doc2Vec). In practice, ensemble method can always result in high precise than single model. And the more diversity of the base models, the better performance the ensemble method can get. So combining the shallow model and the deep model is reasonable. Not just averaging the outputs of the two base models, I use the outputs of base models as input to another classifier. The architecture of the ensemble model is as follows.

![](https://github.com/gy910210/sentiment-analysis/raw/master/pic/ensemble.PNG)

And in L2 level learning, we use logistic regression.

