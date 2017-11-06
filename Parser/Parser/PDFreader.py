import PyPDF2

class PDFreader:
    variable = "blah"

    def Extract_text(self,path):
        pdfFileObj = open(path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pages = pdfReader.numPages
        #print(pages)
        extracted_text=""
        for x in range(pages):
            pageObj = pdfReader.getPage(x)
            extracted_text=extracted_text+pageObj.extractText()
        #print(extracted_text)
        pdfFileObj.close()
        return  extracted_text