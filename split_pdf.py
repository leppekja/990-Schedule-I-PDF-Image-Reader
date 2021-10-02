import sys
from PyPDF2 import PdfFileWriter, PdfFileReader


def split_pdf(file, start_page, end_page):
    '''
    Adapted from https://pyshine.com/Make-a-pdf-cutter/
    Start page and end page are inclusive. 
    '''
    try:
        doc_name = file.rsplit('\\')[-1].split(
            '.')[0] + '_'.join([str(start_page), str(end_page)]) + '.pdf'
        input_file = PdfFileReader(open(file, 'rb'))
        output_file = PdfFileWriter()
        for i in range(int(start_page) - 1, int(end_page)):
            output_file.addPage(input_file.getPage(i))

        with open(doc_name, 'wb') as outputStream:
            output_file.write(outputStream)

        print("File Successfully Split")
        print("File Name: {}".format(doc_name))
    except Exception as e:
        print("Splitting Unsuccessful.")
        print(e)
    return None


if __name__ == '__main__':
    file_name = sys.argv[1]
    start_page = sys.argv[2]
    end_page = sys.argv[3]
    split_pdf(file_name, start_page, end_page)
