
import json
from textblob import TextBlob
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

stopwords_time = ['year','mins','min','minute','hour','hours','seconds','second','minutes','years','week','weeks','months','month','day','days','milliseconds','sec','secs']

data_dict=None
cnt=0
time=[]
exp_setup=[]
pat=[]

def read_file(file):
    global data_dict
    with open(file) as train_file:
        data_dict = json.load(train_file)

# PATIENTS: {<CD> <NNS>} 
def pos_tag(label):
    classified=[]
    file=open("tagged_complete_"+label+".csv","a")
    res_grammer=r"""
      TIME: {<IN> <CD> <NNS>} 
      SETUP: {<NN|NNP>* <NNP> <CD> <I.*|C.*|N.*|P.*|W.*>*} 
      """
    count = 0
    cp = nltk.RegexpParser(res_grammer)
    for content in data_dict:
        #print(content)
        if(content['label']==label):
            try:
                wiki = TextBlob(content['text'])
                #print(wiki.tags)
                tree=cp.parse(wiki.tags)
                # file.write(str(tree))
                for node in tree.subtrees():
                    if(node.label()=="TIME"):
                        words=node.leaves()
                        sent=""
                        for leaf in words[1:]:
                            sent=sent+" "+str(leaf[0])                       
                        for s in stopwords_time:
                            # print(sent)
                            if s in sent.lower():
                                time.append(sent)
                            else:        
                                continue

                    if(node.label()=="SETUP"):
                        words=node.leaves()
                        # print(words[1:])
                        sent=""
                        for leaf in words:
                            sent=sent+" "+str(leaf[0])   
                            # print(sent)                   
                        exp_setup.append(wiki)

                    # if(node.label()=="PATIENTS"):
                    #     words=node.leaves()
                    #     # print(words[1:])
                    #     sent=""
                    #     for leaf in words:
                    #         sent=sent+" "+str(leaf[0])  
                    #         # print(sent)                   
                    #     pat.append(sent)

            except Exception as e:
                print(e)
                print("Exception")
                pass

read_file('test.json')
pos_tag('RESULTS')

print("Time")
print(len(time))
for i in time:
    print(i)

print("Experiment Setup")
print(len(exp_setup))
for i in exp_setup:
    print(i)

# print("Patients")
# print(len(pat))
# for i in pat:
#     print(i)
