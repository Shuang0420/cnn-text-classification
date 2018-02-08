# -*- coding: utf-8 -*-
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def getCheckFile(inputfile):
    officialFAQ = []
    fx = open('371FAQ.txt', 'r')  # standardFAQ
    for line in fx:
        line = line.strip()
        officialFAQ.append(line.split()[0])
    fx2 = open(inputfile, 'r')
    fx3 = open('checkFile.txt', 'w')
    i = 1
    for line in fx2:
        #line = re.sub(r'[\x00-\x0F]+', ' ', line)
        FAQ = ""
        line = line.strip()
        ORI = line.split('\t')[0]
        if (len(line.split('\t')) == 2):
            FAQ = line.split('\t')[1]
        else:
            FAQ = line.split('\t')[-1]
        fx3.write(str(i))
        fx3.write(":")
        for x in range(0, len(officialFAQ)):
            if (FAQ == officialFAQ[x]):
                fx3.write(str(x + 1))
        fx3.write("\n")
        i += 1

def newFile1(file,outputfile):
    f=open(file,'r')
    ques=[]
    for line in f:
        #line = re.sub(r'[\x00-\x0F]+', ' ', line)
        ques.append(line.split('\t')[0])
    f=open('checkFile.txt','r')
    docid=[]
    for line in f:
        docid.append(line.split(':')[1])
    f=open(outputfile,'w')
    for line in range(0,len(docid)):
        f.write(ques[line].strip()+'##'+docid[line].strip()+'\n')

'''
def changeTo371FAQ():
    mapping = []
    f=open('faq_mapping.txt','r')
    for line in f:
        mapping.append((line.split('\t')[0],line.split('\t')[1]))
    f=open('newtrain.txt','r')
    fw=open('goodtrain.txt','w')
    for line in f:
        if (len(line.split())<2):
            print line
            continue
        oriQues=line.split('\t')[0]
        sQues=line.split('\t')[1]
        for m in mapping:
            if (sQues==m[1]):
                newQues=m[0]
                if (newQues != '删除'):
                    fw.write(oriQues+'\t'+newQues+'\n')
                    '''

#changeTo371FAQ()
inputfile = sys.argv[1]
outputfile = sys.argv[2]
getCheckFile(inputfile)
newFile1('out.seg',outputfile)
