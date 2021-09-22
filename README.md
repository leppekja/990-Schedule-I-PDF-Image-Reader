### IRS-990 Schedule I

##### Image / PDF files to CSV

##### Author: Jacob Leppek

##### Date: 9/2/2021

A script to transform PDFs and scanned PDFs of IRS 990 Schedule I forms to CSVs.

1. Install the necessary packages:

- Python
- tabula-py
- pandas
- poppler

conda create -n ocr python=3.8 pandas
conda install -c conda-forge tabula-py

https://github.com/Belval/pdf2image
conda install -c conda-forge poppler -n ocr (for windows, requires download process)
conda install -c conda-forge pdf2image
conda install -c conda-forge pytesseract
Installing tesseract instructions for windows here: https://github.com/UB-Mannheim/tesseract/wiki

2. Manual Preparation

Identify if the file to be read is PDF readable or an image.

- If the text can be highlighted in Adobe PDF reader, then use the PDF_TO_CSV function
- If the text cannot be highlighted in Adobe PDF reader (file is an image), then use the IMAGE_PDF_TO_CSV function

Identify the pages which need to be read for each file.

Known Issues:

Reading the PDF text frequently fails due to Unicode mapping errors:

    Got stderr: Sep 15, 2021 6:38:24 PM org.apache.pdfbox.pdmodel.font.PDType0Font toUnicode
    WARNING: No Unicode mapping for CID+425 (425) in font BNDHBP+Calibri
    Sep 15, 2021 6:38:25 PM org.apache.pdfbox.pdmodel.font.PDType0Font toUnicode
    WARNING: No Unicode mapping for CID+415 (415) in font BNDHBP+Calibri
    Sep 15, 2021 6:38:25 PM org.apache.pdfbox.pdmodel.font.PDType0Font toUnicode
    WARNING: No Unicode mapping for CID+332 (332) in font BNDHBP+Calibri

Workarounds for this using the tabula package are unavailable. See the [documentation](https://tabula-py.readthedocs.io/en/latest/faq.html#i-got-a-warning-error-message-from-pdfbox-including-org-apache-pdfbox-pdmodel-is-it-the-cause-of-empty-dataframe), and StackOverflow [here](https://stackoverflow.com/questions/58829597/how-to-solve-no-unicode-mapping-error-from-pdfbox). In these cases, use the IMAGE_PDF_TO_CSV function.

Tides Foundation 2016
Page 74 not read
Page 80 - missed top 5 grants
Check for () to indicate negative grant amounts
Remove any Totals rows
Check if

IMAGE PROCESSING ISSUES:

- Some missing grants
- text cut off

## Process Labels:

For each file, marks the process used:

#### camelot-py

- C:\Users\Jacob\Documents\Contracting\DAFs_Bergoff_Pruitt\Files\Tides Foundation 2016 Form 990 PDF PG_59_86.pdf
  Drop indices [0,1,2,3]
- C:\Users\Jacob\Documents\Contracting\DAFs_Bergoff_Pruitt\Files\Tides Foundation 2017 Form 990 PDF PG_53_80.pdf
  Drop indices[0,1,2,3,4]

#### read_990_image

- Files\Bessemer Gift Fund 2018 990 Image PG_31_122.pdf
- Files\Community Foundation of Greater Memphis 2016 Form 990 Image PG_26-721.pdf
- Files\Community Foundation of Greater Memphis 2017 Form 990 Image PG_28-102.pdf
- Files\Tulsa Community Foundation 2018 Form 990 Image PG-50-246.pdf
- Files\Tulsa Community Foundation 2017 Form 990 Image PG-49-221.pdf

Approach informed by

- https://fazlurnu.com/2020/06/23/text-extraction-from-a-table-image-using-pytesseract-and-opencv/
- https://medium.com/analytics-vidhya/table-detection-and-text-extraction-5a2934f61caa
