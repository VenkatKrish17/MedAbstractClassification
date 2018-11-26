# from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import decomposition, ensemble

# import pandas, xgboost, numpy, textblob, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers
import json
from textblob import TextBlob
import nltk
from nltk import word_tokenize, pos_tag
from nltk import bigrams

# load the dataset
file = 'train.json'
with open(file) as train_file:
    dict_train = json.load(train_file)

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
# def check_pos_tag(x, flag):
#     cnt = 0
#     try:
#     	wiki = TextBlob(x)
#     	#print(wiki.tags)
#     	#wiki = pos_tag(word_tokenize(x))
#     	for tup in wiki.tags:
#     		ppo = list(tup)[1]
#     		if ppo in pos_family[flag]:
#     			cnt += 1
#     except:
#         pass
#     return cnt

words = []
def count_pos_tag(label, flag):
	cnt = 0
	for content in dict_train:
		if(content['label'] == label):
			try:
				wiki = TextBlob(content['text'])
				#print(wiki.tags)
				for tup in wiki.tags:
					ppo = list(tup)[1]
					if ppo in pos_family[flag]:
						#print(list(tup)[0])
						words.append(list(tup)[0])
						cnt += 1
                        
			except:
				pass
	return cnt

count1 = count_pos_tag("BACKGROUND", 'noun')
# count2 = count_pos_tag("OBJECTIVE", 'noun')
# count3 = count_pos_tag("METHODS", 'noun')
# count4 = count_pos_tag("RESULTS", 'noun')
# count5 = count_pos_tag("CONCLUSIONS", 'noun')

print('BACKGROUND, Noun: ',count1)
# print('OBJECTIVE, Noun: ',count2)
# print('METHODS, Noun: ',count3)
# print('RESULTS, Noun: ',count4)
# print('CONCLUSIONS, Noun: ',count5)

print(words)
