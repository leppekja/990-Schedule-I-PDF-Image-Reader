from tabula import read_pdf
from PyPDF2 import PdfFileWriter, PdfFileReader
import camelot
import pandas as pd

FILE = 'Tides_Foundation_PDF.pdf'


def pdf_to_csv(file, start_page, end_page, output_file_name, table_headers=False, cell_lines=False):
    """
    Function to convert IRS 990 Schedule I tables to .csv format.

    Inputs:
    file: file path to the .pdf to read
    start_page: page which the grant tables begins on
    end_page: page which the grant tables ends on (inclusive)
    table_headers: If the table to read has headers or not
    cell_lines: True if inner table borders; false if not. Enables lattice option if cell lines exist
    """
    headers = (None if table_headers is False else table_headers)

    return read_pdf(file, pages='{}-{}'.format(start_page, end_page), pandas_options={'header': headers}, stream=not cell_lines, lattice=cell_lines)


def split_pdf(file, start_page, end_page):
    '''
    Adapted from https://pyshine.com/Make-a-pdf-cutter/
    '''
    doc_name = file.rsplit('\\')[-1].split(
        '.')[0] + '_'.join([str(start_page), str(end_page)]) + '.pdf'
    input_file = PdfFileReader(open(file, 'rb'))
    output_file = PdfFileWriter()
    for i in range(start_page - 1, end_page):
        output_file.addPage(input_file.getPage(i))

    with open(doc_name, 'wb') as outputStream:
        output_file.write(outputStream)

    return None


def camelot_read_pdf(file, start_page, end_page, flavor, save=True, del_header_indices=None):
    tables = camelot.read_pdf(file, flavor=flavor, pages=str(
        start_page) + '-' + str(end_page))
    dfs = [t.df for t in tables]
    concat_dfs = pd.concat(dfs)

    if del_header_indices:
        concat_dfs.sort_index(inplace=True)
        concat_dfs = concat_dfs.drop(index=[del_header_indices])

    return concat_dfs
