
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:12:45 2020

@author: shawn
"""
from tkinter import *
import time

#code is following the tutorial from tech with tim
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()
import pickle
import numpy
import tensorflow

import tflearn #Tensorflow 2.0 remove a document needed by tflearn, so we are using tensorflow version 1.13.1


import socket

import random
import json
#----------------------------------------------------------------------------------------------------------------------------
from textblob import TextBlob


def sentiment(msg):
     text = msg
     blob = TextBlob(text)
     pol = float(blob.sentiment.polarity)
     return pol






#https://blog.csdn.net/Hallywood/article/details/80154146 Coreference Resolution Stanford
#https://github.com/aleenaraj/Coreference_Resolution
#https://github.com/dasmith/stanford-corenlp-python
#https://github.com/Lynten/stanford-corenlp
#https://blog.csdn.net/guolindonggld/article/details/72795022?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3
#https://stackoverflow.com/questions/50004797/anaphora-resolution-in-stanford-nlp-using-python
from stanfordcorenlp import StanfordCoreNLP
import re
from collections import Counter




#这里的路径是下载并解压stanford-corenlp-full的路径
nlp = StanfordCoreNLP(r'D:\chatbot3\stanford-corenlp-full-2016-10-31')#File Location to change
sentence="I broke up with my boyfriend. But I still love him. I have no idea why he did this to me."
#print ('Tokenize:', nlp.word_tokenize(sentence))
#print (nlp.pos_tag(sentence))
#print (nlp.ner(sentence))
#print (nlp.parse(sentence))
#print (nlp.dependency_parse(sentence))

#props={'annotators': 'tokenize,ssplit,pos','pipelineLanguage':'en','outputFormat':'xml'}
#print (nlp.annotate(sentence, properties=props))

#print(nlp.coref(sentence))

def combineDoubleToken(Tok):
    msgCombine=""
    for a in Tok:
        for b in a:
            msgCombine=msgCombine+b+" "
        msgCombine=msgCombine+". "
    return msgCombine

def coreferenceResolution(msg):
    colist=nlp.coref(msg) 
    comsg=msg.split(".") #This split the msg by ".", which is the same as coref's split way.
    tokenmsg=[]
    for cg in comsg:
        tokenmsg.append(nlp.word_tokenize(cg))
    for ai in range(0,len(colist)): #Each signle list is about one same coreference, such as: boyfriend he him....
        singlelist=colist[ai]
        if(len(singlelist)>1):
            referencePlace=singlelist[0][0]
            referenceStart=singlelist[0][1]
            referenceEnd=singlelist[0][2]
            referenceWord=singlelist[0][3]
            for j in range(0,len(singlelist)):
                core=singlelist[j]
                if referenceWord!=core[3]:
                    tokenReferWord=nlp.word_tokenize(referenceWord)
   
                    tempPlace=core[0]
                    tempStart=core[1]-1
                    tempEnd=core[2]-1
                    tempToken=tokenmsg[tempPlace-1]
               
                    tempToken[tempStart:tempEnd]=""
                    for i in reversed(tokenReferWord):
                        tempToken.insert(tempStart,i)
                    tokenmsg[tempPlace-1]=tempToken
             

                    msg=combineDoubleToken(tokenmsg)
                  
                    coreferenceResolution(msg)
                    break
    return msg
                
                
                
                
          #  print(keyWord)
   
           
          
    


coreferenceResolution(sentence)

#nlp.close() 
#记得关闭，因为这个对内存的消耗是非常大的。
#----------------------------------------------------------------------------------------------------------------------------
"""Reference： Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html
"""

################ Spelling Corrector 



def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open(r"D:\chatbot3\big.txt",'r').read())) #File Location to change


def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


#End of spelling check function


#----------------------------------------------------------------------------------------------------------------------------

with open("qa.json") as file: #qa.json saves questions and answers
    data=json.load(file)


words=[]
labels=[]
docsx=[]
docsy=[]

labels=sorted(labels)
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words.extend(nltk.word_tokenize(coreferenceResolution(pattern)))#coreferenceResolution part!!!!!!!!!!!!!
        docsx.append(nltk.word_tokenize(pattern))
        docsy.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
words=[stemmer.stem(w.lower()) for w in words]
words=sorted(list(set(words)))
inputF=[]
output=[]
outputF=[0 for _ in range(len(labels))]

#The general idea is to see how similar the input is compared to exists questions.
#This uses some techique of machine learning
for x,doc in enumerate(docsx):
    exist=[]

    temp=[stemmer.stem(w) for w in doc]
    for w in words:
        if w in temp:
            exist.append(1)
        else:
            exist.append(0)
    outputF2=outputF[:]
    outputF2[labels.index(docsy[x])]=1
    inputF.append(exist)
    output.append(outputF2)

inputF=numpy.array(inputF) #Change list to array
output=numpy.array(output)
with open("data.pickle","wb") as f:
    pickle.dump((words,labels,inputF,output),f)

#Begin using tflearn
tensorflow.reset_default_graph()

net=tflearn.input_data(shape=[None,len(inputF[0])])
for x in range(0,3,1):
    net=tflearn.fully_connected(net,8)

net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)
model = tflearn.DNN(net)



#Part may change to run!!!!!!! (below)

#model.load("model.tflearn") #If already setted up, can uncomment this part, and comment the below part

model.fit(inputF, output, n_epoch=2000,batch_size=6,show_metric=True) #Only for set up #n_epoch decides how many time to train
model.save("model.tflearn")    #Only for set up

#Part may change to run!!!!!!! (below)










def sentences(s,words):
    s=coreferenceResolution(s) #coreferenceResolution Part!!!!! 
    
    st=[0 for _ in range(len(words))]
    sw=nltk.word_tokenize(s)
    sw=[stemmer.stem(w.lower()) for w in sw] #to lower make sure case is not sensitive
    #print(sw)
    #sw is tokenized list of the input sentence by user
    
    for t0 in sw:
        for i,t2 in enumerate(words):
            if correction(t2)==correction(t0): #Spelling part!!!!!!!!
                st[i]=1
    return numpy.array(st)
 
def ask(inp):
        ise=sentiment(inp)#sentiment Part!!!
        print (nlp.pos_tag(inp))
        probability=model.predict([sentences(inp,words)])
        #print(probability)
        match=False  #Used to check if the match is big enough

   #     print(probability)
        

        for f1 in probability:
            for f2 in f1:
                #print(f2)
                if(round(f2,2)>0.8): #if there is a match big enough
                    match=True;



        ctag=numpy.argmax(probability) #Save the index of the most match place in the array
       
        if(match):#If match an option
            print("---")
            print(ise)
            tag=labels[ctag]     
            tempmode=""#sentiment Part!!!
            if(ise>0):#sentiment Part!!!
                tempmode="Glad to hear that."#sentiment Part!!!
            if(ise<0):#sentiment Part!!!
                tempmode="Sad to hear that."#sentiment Part!!!

            for fch in data["intents"]:
                if fch['tag'] ==tag:
                  responses=fch['responses']
            re=tempmode+random.choice(responses)
            chatbotsend(re)#sentiment Part!!!
        else:#If no option matched
            response2=["Sorry, but i cannot understand","I don't get it","hmmmmm....what?","Ahhh wait, chatbot language please","Whaat I don't get it"]
            chatbotsend(random.choice(response2))
             







              


root = Tk()
root.title("yuki")
#发送按钮事件
def usersend():
  #在聊天内容上方加一行 显示发送人及发送时间
  msgcontent = "user " + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n '
  text_msglist.insert(END, msgcontent, 'green')
  text_msglist.insert(END, text_msg.get('0.0', END))
  ask(text_msg.get('0.0', END))
  text_msg.delete('0.0',END)

def chatbotsend(mssg):
  mssg=mssg+"\n"
  msgcontent = "Chatbot " + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n '
  text_msglist.insert(END, msgcontent, 'green')
  text_msglist.insert(END, mssg)
  #ask(mssg) #Chat with Yukiself
 






frame_left_top   = Frame(width=380, height=270, bg='white')
frame_left_center  = Frame(width=380, height=30, bg='white')
frame_left_bottom  = Frame(width=380, height=20)

text_msglist    = Text(frame_left_top)
text_msg      = Text(frame_left_center);
button_sendmsg   = Button(frame_left_bottom, text="send", command=usersend)

text_msglist.tag_config('green', foreground='#008B00')

frame_left_top.grid(row=0, column=0, padx=2, pady=5)
frame_left_center.grid(row=1, column=0, padx=2, pady=5)
frame_left_bottom.grid(row=2, column=0,pady=10)
frame_left_top.grid_propagate(0)
frame_left_center.grid_propagate(0)
frame_left_bottom.grid_propagate(0)

text_msglist.grid()
text_msg.grid()
button_sendmsg.grid(sticky=E)

root.mainloop()
nlp.close() 









