import sys
import os
from PyPDF2 import PdfFileWriter, PdfFileReader


def split_pdf(file, start_page, end_page, output_folder_path):
    '''
    Adapted from https://pyshine.com/Make-a-pdf-cutter/
    Start page and end page are inclusive. 
    '''
    try:
        doc_name = file.rsplit('\\')[-1].split(
            '.')[0] + '_' + '_'.join([str(start_page), str(end_page)]) + '.pdf'

        if output_folder_path:
            doc_name = os.path.join(output_folder_path, doc_name)

        input_file = PdfFileReader(open(file, 'rb'))
        output_file = PdfFileWriter()
        for i in range(int(start_page) - 1, int(end_page)):
            output_file.addPage(input_file.getPage(i))

        with open(doc_name, 'wb') as outputStream:
            output_file.write(outputStream)

        print("File Successfully Split")
        print("File Name: {}".format(doc_name))
        print("-" * 20)
    except Exception as e:
        print("Splitting Unsuccessful.")
        print(e)
    return None


def split_folder(folder, output_folder_path):

    def user_input(file_name):
        print(file_name)
        start_page = input("Start Page of File: ")
        end_page = input("End Page of File: ")
        print("-" * 25)
        return (start_page, end_page)

    for file_name in os.listdir(folder):
        f = os.path.join(folder, file_name)
        if os.path.isfile(f):
            start_page, end_page = user_input(file_name)
            split_pdf(f, start_page, end_page, output_folder_path)


if __name__ == '__main__':
    location = sys.argv[1]

    if os.path.isdir(location):
        output_folder = sys.argv[2]
        split_folder(location, output_folder)
    else:
        start_page = sys.argv[2]
        end_page = sys.argv[3]
        output_folder = sys.argv[4]
        split_pdf(location, start_page, end_page, output_folder)
