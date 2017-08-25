import docx2txt
import PyPDF2
import os

class split:

    global key_words
    global tag
    key_words=[]
    tag={}

    def __int__(self):
        self.parseText()
        self.SplitKeywords()
        self.check_words()

    def parseDOCX(self,path):
        text = docx2txt.process("/home/hadoop/ML/resume/KamalSaboo[15_0].docx")
        #print("After converting text is ", text)
        return text

    def parsePDF(self, path):
        pdfFileObj = open(path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pages = pdfReader.numPages
        # print(pages)
        extracted_text = ""
        for x in range(pages):
            pageObj = pdfReader.getPage(x)
            extracted_text = extracted_text + pageObj.extractText()
        # print(extracted_text)
        pdfFileObj.close()
        return extracted_text

    def SplitKeywords(self,path):
        keyword_path= path #"/home/hadoop/LegalTags.txt"
        with open(keyword_path) as f:
            keywords = f.readlines()
        keywords = [x.strip('\ufeff') for x in keywords]
      #print(keywords)
        return keywords

    def Find_files(self, resume_path):
        extracted_text = ""
        filenames = os.listdir(resume_path)
        filenames = [x.split() for x in filenames]
        #for i in range(filenames.__len__()):
         #   print((str(filenames[i])))
        for i in range(filenames.__len__()):
            complete_path = resume_path + str(filenames[i]).strip("['']")
            #print(complete_path)
            if (str(filenames[i]).find("pdf") > 0):
                extracted_text= self.parsePDF(complete_path)
                print(complete_path)
                self.findTag(extracted_text)

            elif((str(filenames[i]).find("docx") > 0)):
                #print(complete_path)
                extracted_text=self.parseDOCX(complete_path)
                print(complete_path)
                self.findTag(extracted_text)

            #elif((str(filenames[i]).find("doc") > 0)):
             #  print(complete_path)


    def findTag(self,text):
        tag=""
        #if(text.__eq__("/home/hadoop/ML/resume/AsawariShirodkar[12_0].pdf")):
       # print(text)
        keywords= self.SplitKeywords("/home/hadoop/LegalTags.txt")
        for i in range(keywords.__len__()):
            if((text.lower()).find(str(keywords[i]).lower())>0):
                tag=str(keywords[i])+" "+tag
        print(tag)


split1 = split()
#split1.SplitKeywords()
split1.Find_files("/home/hadoop/ML/resume/")