import os
from Parser.PDFreader import PDFreader
from Parser.DbConnection import DbConnection


class Finder:
    pdfreader = PDFreader()

    def __int__(self):
        #self.__init__("/home/hadoop/LegalTags.txt","/home/hadoop/ML/resume")
        print("")

    def __init__(self,keyword_path,resume_path):
        global key_words,tag  # list of legal terms
        global filenames  # list of filename
        global pdfreader
        global database

        tag=[]
        #getting file names
        key_words = self.SplitKeywords(keyword_path)
        self.Find_files(resume_path)


    def SplitKeywords(self,keyword_path):
        with open(keyword_path) as f:
            keywords = f.readlines()
        keywords = [x.strip('\ufeff') for x in keywords]
        keywords = [x.strip('\n') for x in keywords]
        return keywords

    def Find_files(self,resume_path):
        extracted_text=""
        filenames = os.listdir(resume_path)
        filenames = [x.split() for x in filenames]
        #print(filenames[6])

        for i in range(filenames.__len__()):
            complete_path = resume_path + "/" + str(filenames[i]).strip("['']")
            if(str(filenames[i]).find("pdf")>0):
                extracted_text = self.pdfreader.Extract_text(complete_path)
                self.insert(complete_path,self.FindTag(extracted_text))


    def insert(self,path,tag):
        db = DbConnection()
        temp=""
        for i in range(tag.__len__()):
            temp=temp+", "+tag[i]
        print(temp)
        print(path)
        db.Insert_Table(path,temp)

    def FindTag(self,text):
        for i in  range(key_words.__len__()):
            if(text.find(key_words[i].lower())>0):
                tag.append(key_words[i])
        return tag

#f = Finder("/home/hadoop/LegalTags.txt","/home/hadoop/ML/resume")