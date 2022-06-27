# Topic-Labeling-with-Ukparl-Dataset

<br />

## Introduction

Companies across a wide spectrum of industries are beginning to embrace data science as a
means of gathering and leveraging smarter business intelligence. Those organizations that fail
to keep up won’t be able to compete in the future. This is an exciting time to be a data
scientist, or aspiring data scientist, as dynamic and lucrative new opportunities continue to
open up at a rapid clip.
In order to achieve my goal of becoming a member of the data world, I wanted to continue
my path of learning, and that is what led me to apply for the role at Pivony.
It is logical that in order to get the offer, Pivony had to test my technical skills and my problemsolving ability, and even if I am not the most suitable candidate in terms of pure knowledge,
I am sure that I will compensate for everything with my hard work and dedication.

## Problem

In this Data Science Case Study, I will try to Implement a Topic Labeling algorithm. First, I will
gather and preprocess the needed data for training. I will then present the model that I chose
and detail the algorithm. Finally, after the training, I will interpret and explain the results with
visualizations.

## Data

- The data in this case is the UKParl dataset. It is a Data Set consisting of 354,400 tokens
for Topic Detection with Semantically Annotated Text. It contains British
parliamentary debates of the House of Commons from 2013 to 2016.
- Due to limited time for the case study, I limited the training data to the 2013-2014
session which is already composed of nearly 120,000 tokens and took a considerably
long training time.
- The data is divided into multiple txt files where each file contains a topic so I gathered
all of it into one csv file (≈ 85 mb) to facilitate the manipulation.
- After that, I removed some insignificant add-ons (like some ‘\n’ and resetting the
index) to get a clean DataFrame.
- Then, I did some classic text preprocessing (removing step words, tokenizing the data)
to get good results with the model.

## Model

From my previous knowledge on NLP, I know that the transformers are the most suitable and
recent approach to train huge data (even though it takes a considerable time, but we will
solve this problem later on in the training section). And even if LDA with a good metric can
achieve great results, I really wanted to pursue my intuition on transformers and test one of
its architectures and I could eventually switch the model if I wasn’t able to get clear results
(which is not the case).

This thinking led to my search on finding a pre-trained transformer on topic labeling and
sharpening it with the data I gathered. I first went to HuggingFace website but unfortunately,
I couldn’t really find what I wanted. With further research, I got my hand on a BERT model
(Bidirectional Encoder Representations from Transformers) for Topic Labeling from the library
BertTopic which fits perfectly my expectations

## BertTopic

BertTopic is a topic modeling technique that leverages HuggingFace transformers and c-TFIDF to create dense clusters allowing for easily interpretable topics whilst keeping important
words in the topic descriptions.
The following image shows the BertTopic pipeline: 

![alt text](https://github.com/rasta-nitzsche/Topic-Labeling-with-Ukparl-Dataset/blob/main/BERTopic%20pipeline.JPG)

- First, we convert the documents to numerical data through embeddings. We use BERT
for this purpose as it extracts different embeddings based on the context of the word.
- Then we cluster the data by using dimensionality reduction (UMAP) and then using a
cluster algorithm (HDBSCAN). This is a quite good combination (UMAP + HDBSCAN)
since UMAP maintains a lot of local structure even in lower-dimensional space and
HDBSCAN does not force data points to clusters as it considers them outliers.
- c-TF-IDF is a class-based variant of TF-IDF that allows the extraction of what makes
each set of documents unique compared to the other (main topic). Here we can see
the importance of our preprocessing as TF-IDF penalizes frequent step words.
- In order to create a topic representation, the top 20 words per topic are taken based
on their c-TF-IDF scores. The higher the score, the more representative it should be of
its topic as the score is a proxy of information density.
- Finally, we go through a topic reduction to avoid too many topic detection and keep
what is essential and we got our topics ready.

## Training

Now that we saw the intuition behind our chosen algorithm, we can start training our data
and advance in our project.

Explaining the parameters:

topic_model = BERTopic(verbose=True, embedding_model= "all-mpnet-base-v2" ,min_topic_size=1000, calculate_probabilities=True)

- Verbose = True: to produce the logging output.
- Embedding_model: I chose all-mpnet-base-v2 because he is most accurate among all the models even though it requires way more computation resources.
- min_topic_size=1000: since we have very large data (120,000) a topic would be relevant if it contained at least a certain amount of recurrence.
- calculate_probabilities=True: we calculate the probability of a document belonging to any topic. That way, we can select, for each document, the topic with the highest probability. (It is also very expensive in computation resources)

The huge size of the dataset in addition to the rich set of parameters selected led us to huge
training time. Indeed, while running the configuration on my personal computer (which is
actually not that bad with a GTX 1060 GPU), the estimated time was about 40 hours training
time, which is actually quite huge regarding the deadline.

To solve this problem, I oriented my thinking into cloud solutions so I wanted to try Google
Collab with their Tesla K80 Gpu environment with Cuda technology. A wise choice since
actually reduced the training time from 38 hours to only an hour and half.

So finally, we got our model ready. The next step is to visualize results.

## Results and interpretation

Here is the Google Collab link where I trained my model :

[https://colab.research.google.com/drive/1ihhd6V4QrEeUFu2IgG6CuX5_k8VfNbFQ?us](https://colab.research.google.com/drive/1ihhd6V4QrEeUFu2IgG6CuX5_k8VfNbFQ?usp=sharing)

![alt text](https://github.com/rasta-nitzsche/Topic-Labeling-with-Ukparl-Dataset/blob/main/result1.JPG)

- The map is an Intertopic Distance Map. It is a visualization of the topics in a twodimensional space. The area of these topic circles is proportional to the amount of words that belong to each topic across the dictionary. The circles are plotted using a
multidimensional scaling algorithm based on the words they comprise, so topics that
are closer together have more words in common.
- Topic 0 (the big one in the center) represents the irrelevant or common data structure
between the documents (No clear Topic), we found words like friend, people… The
interactive visualization is available on the Collab Notebook.
- The Topics 3,4,7 (top left) are close to each other and are subjects about conflictual
situations (force, arm, defense for topic 4 – Syria, Iran, Weapons for topic 3 – Ukraine,
Russia, foreign for topic 7)
- We also have on the top left topic 6 about art and culture and topic 8 about Scotland
UK and currency.
- On the bottom right we have topic 1 (transport), topic 2 (energy & gas) and topic 5
(water & environment) which have more words in common.

The following visualization shows each topic and its main word scores so we can have a global
look about what these topics are talking about. (Topic -1 is for not relevant topics)

![alt text](https://github.com/rasta-nitzsche/Topic-Labeling-with-Ukparl-Dataset/blob/main/result2.JPG)


## Conclusion

After analyzing the UkParl Dataset (2013-2014 only) we can assume that the UK Parliament
mainly talks about those following subjects: war and army, environment, art, energy & gas,
transport, which are usual concerns of a country.
From a personal point of view, I really enjoyed doing this project even though I am not sure if
I did it on the right terms of the company (mainly because of the absence of management
which means I will improve a lot with the right guidance). I am also proud of the results and
the learning I got through the project, and I have to thank Pivony for the opportunity and it
was a pleasure cooperating with you.

## Resources and bibliography

- Link to download the Data Set : https://www.clarin.eu/resourcefamilies/parliamentary-corpora
- BertTopic GitHub project: https://github.com/MaartenGr/BERTopic
- Intuition behind the model: https://towardsdatascience.com/topic-modeling-withbert-779f7db187e6
- Federico, N. (2018) UKParl: A Data Set for Topic Detection with Semantically Annotated
Text http://lrec-conf.org/workshops/lrec2018/W2/pdf/6_W2.pdf
