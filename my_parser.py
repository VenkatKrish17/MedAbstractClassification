
import json
from textblob import TextBlob
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import csv
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}
data_dict=None
stopwords_time = ['year','mins','min','minute','hour','hours','seconds','second','minutes','years','week','weeks','months','month','day','days','milliseconds','sec','secs']
def read_file(file):
    global data_dict
    with open(file) as train_file:
        data_dict = json.load(train_file)
    return data_dict

conclusion_grammar=r"""
  FUT:{^<JJ> <NN|NNS> <VB.*|MD> }
  RES:{^<RB>?<DT|PRP.*>?<NNS><VB.*|IN.*>}
  NOV_2:{^<DT><VBZ><DT><JJ>}
"""
result_grammar=r"""
  TIME: {<IN> <CD> <NNS>}
  SETUP: {<NN|NNP>* <NNP> <CD> <I.*|C.*|N.*|P.*|W.*>*}
  """
method_grammar=r"""
NOP: {<CD> <NNS> <IN> <JJ> <NN.*>}
"""
objective_grammar=r"""
PUR:{<DT> <VBN> <JJ|VBN> <NN>}
PA:{<DT> <NN|NNS>}
PUR:{<NN> <IN> <DT> <NN>}
SA:{<JJ> <NN|NNS>}
"""
# DT VBN JJ/VBN NN
# primary objective
# DT NN/NNS
# Aim of the/this study
# NN IN DT NN
# A second/secondary goal/objective/aim
# JJ NN/NNS
cp1 = nltk.RegexpParser(conclusion_grammar)
cp2 = nltk.RegexpParser(result_grammar)
cp3 = nltk.RegexpParser(method_grammar)
cp4 = nltk.RegexpParser(objective_grammar)
def pos_tag(label,text):
    # for content in data_dict[:500]:
    #     #print(content)
    #     cp = nltk.RegexpParser(conclusion_grammar)
    #     if(content['label']==label):
    #         try:
    #             wiki = TextBlob(content['text'])
    #             #print(wiki.tags)
    #             # if(wiki.lower().startswith('addition')):
    #             #     print(wiki.tags)
    #             tree=cp.parse(wiki.tags)
    #             for node in tree.subtrees():
    #                 if(node.label() == "RES"):
    #                     print(wiki)
    #                     #print(wiki.tags)
    #                     # print(node.leaves()[1][1])
    #                     # if(node.leaves()[0][1]=='JJ'):
    #                     #     print(wiki)
    #                     #     print(wiki.tags)
    #                     # if(node.leaves[1][1]=='JJ'):
    #                     #     print(wiki)
    #                     #     print(wiki.tags)
    #
    #         except Exception as e:
    #             print(e)
    #             print("Exception")
    #             pass
    #COMB: {<NN> <VBN> <IN> <NN> <I.*|C.*|V.*|N.*|R.*|P.*|W.*>* <JJ> } ,
    #NOV_1:{^<DT><JJ><NN>}
    if(label=="CONCLUSIONS"):
        out={'CONCLUSIONS':text}
        wiki=TextBlob(text)
        tree=cp1.parse(wiki.tags)
        #print(wiki.tags)
        for node in tree.subtrees():
            #print(node.label())
            if(node.label()=="RES"):
                out= {'OUTCOME':text}
            elif(node.label()=="FUT"):
                out= {'FUTURE ':text}
            if(node.label()=="NOV_2"):
                out= {'NOVELTY':text}
            else:
                out= {'CONCLUSION':text}
        return out
    elif(label=="RESULTS"):
        wiki=TextBlob(text)
        res_output={'time':'NA','experiment':'NA'}
        tree=cp2.parse(wiki.tags)
        for node in tree.subtrees():
            if(node.label()=="TIME"):
                # print("time found")
                words=node.leaves()
                sent=""
                for leaf in words[1:]:
                    sent=sent+" "+str(leaf[0])
                # print("printing sentence")
                # print(sent)
                for s in stopwords_time:
                    if s in sent.lower():
                        res_output['time']=sent
                    else:
                        continue
            if(node.label()=="SETUP"):
                res_output['experiment']=text
        return res_output
    elif(label=="METHODS"):
        wiki=TextBlob(text)
        met_output={'TEST SCENARIO':'NA'}
        tree=cp3.parse(wiki.tags)
        for node in tree.subtrees():
            if(node.label()=="NOP"):
                words=node.leaves()
                sent=""
                for leaf in words:
                    sent=sent+" "+str(leaf[0])
                for s in stopwords_time:
                    if s not in sent.lower():
                        met_output['TEST SCENARIO']=sent
                    else:
                        continue
        return met_output
    elif(label=="BACKGROUND"):
        return {'BACKGROUND':text}
    elif(label=="OBJECTIVE"):
        out={'OBJECTIVE':text}
        wiki=TextBlob(text)
        tree=cp4.parse(wiki.tags)
        #print(wiki.tags)
        for node in tree.subtrees():
            #print(node.label())
            if(node.label()=="PUR"):
                out= {'PURPOSE':text}
            elif(node.label()=="PA"):
                out= {'PRIMARY OBJECTIVE':text}
            if(node.label()=="SA"):
                out= {'SECONDARY OBJECTIVE':text}
            else:
                out= {'OBJECTIVE':text}
        return out

data_dict=read_file('with_id_test.json')
output_list=[]
for k in data_dict:
    output_dict=dict({'id':k['id']})
    for j in k['sentences']:
        output_dict.update(pos_tag(j['label'],j['text']))
    output_list.append(output_dict)

#
# with open('results.csv','w') as file:
#     writer = csv.writer(file)
#     for dict in output_list:
#         for key, value in dict.items():
#             writer.writerow([key,value])

with open('results.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file,['id','BACKGROUND', 'time', 'CONCLUSION','OUTCOME','OBJECTIVE','FUTURE','NOVELTY', 'experiment','TEST SCENARIO','PURPOSE','PRIMARY OBJECTIVE','SECONDARY OBJECTIVE',])
    dict_writer.writeheader()
    dict_writer.writerows(output_list)

# for i in ('train','test','dev'):
#     print(i)
#     read_file('../modified_data/20K/'+i+'.json')
#     pos_tag('OBJECTIVE')
#     read_file('../modified_data/20K/'+i+'.json')
#     pos_tag('RESULTS')
#     read_file('../modified_data/20K/'+i+'.json')
#     pos_tag('METHODS')
#     read_file('../modified_data/20K/'+i+'.json')
#     pos_tag('BACKGROUND')
#     read_file('../modified_data/20K/'+i+'.json')
#     pos_tag('CONCLUSIONS')
