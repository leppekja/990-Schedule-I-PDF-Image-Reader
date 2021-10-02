from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from pdf2image.exceptions import PDFPageCountError
from pdf2image.exceptions import PDFSyntaxError
import pytesseract
from pytesseract import Output, TesseractError
import re
import pandas as pd
import cv2
import numpy as np
import sys


def create_samples(path, start_page, end_page, rotate_img=True):
    originals = convert_pdf_pages_to_imgs(path, start_page, end_page)
    cleaned = [clean_image(i, rotate_img=rotate_img) for i in originals]
    if rotate_img:
        originals = [rotate(i) for i in originals]

    return originals, cleaned


def read_img(img):
    '''
    Read in a grayscale image.
    '''
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def show(img):
    '''
    Show image on screen; press key to exit.
    '''
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def rotate(img):
    '''
    Rotate image 90 degrees clockwise.
    '''
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated


def clean_image(img, rotate_img=True):
    '''
    Makes an image grayscale and rotates it, when not reading directly from file
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if rotate_img:
        img = rotate(img)
    return img


def blank_image(img):
    '''
    Create a blank canvas for test previewing contours
    https://pythonwife.com/contours-in-opencv/
    '''
    blank = np.zeros(img.shape, dtype='uint8')
    return blank


def draw(img, to_draw):
    '''
    Draw the contours on a copy of the given image to preview
    '''
    img_copy = img.copy()
    drawn = cv2.drawContours(img_copy, to_draw, -1, (255, 0, 0), 3)
    show(drawn)
    return None


def identify_horizontal_borders(img, first_page=False):
    '''
    Identifies the horizontal borders of the table in the given image
    https://medium.com/analytics-vidhya/table-detection-and-text-extraction-5a2934f61caa
    Note that this function is specific to the format of the IRS-990, ignores the top header lines
    Ensure sys is imported
    Returns them sorted, from smallest to largest (top-down)
    '''
    x, y = img.shape
    # connect and highlight horizontal lines
    img_copy = preprocess(img.copy(), kind='h')

    hor_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (np.array(img_copy).shape[1]//150, 1))
    lines = cv2.dilate(img_copy, hor_kernel, iterations=2)
    contours, hierarchy = cv2.findContours(
        lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove any lines on the border of the image
    # We only want the center table
    # Remove the top two title lines and the box around the entire image
    contours = [c for c in contours if
                50 < np.mean(c[:, 0][:, 1]) < (x - 50)][1:-2]

    # sort the contours from top to bottom
    contours.sort(key=lambda h: np.mean(h[:, 0][:, 1]))

    # Need to avoid false positives created by hole punch circles
    # 2000 chosen arbitrarily
    contours = [c for c in contours if cv2.contourArea(c) > 2000]

    # If first page, delete additional lines for header and footer
    if first_page:
        table_lines = []
        # iterate from the bottom line up to the top
        contours.reverse()
        # check spacing between bottom lines to decide how many to take out.
        for idx, line in enumerate(contours[:-1]):
            # if the line above the current line is greater than 50 px away, include it
            if np.mean(contours[idx + 1][:, 0][:, 1]) - np.mean(line[:, 0][:, 1]) < -75:
                table_lines.append(line)
        table_lines.reverse()

        # Delete headers, measure from the bottom of the document
        contours = table_lines[-7:]

    return contours


def identify_vertical_borders(img, first_page=False):
    '''
    Identifies the vertical borders of the table in the given image
    https://medium.com/analytics-vidhya/table-detection-and-text-extraction-5a2934f61caa
    Ensure sys is imported
    Returns them sorted, from smallest to largest (left-right)
    '''
    x, y = img.shape
    # connect and higlight vertical lines
    img = preprocess(img.copy(), kind='v')

    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, np.array(img).shape[0]//150))
    lines = cv2.dilate(img, vert_kernel, iterations=4)
    contours, hierarchy = cv2.findContours(
        lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove any lines on the border of the image
    # We only want the center table
    contours = [c for c in contours if
                50 < np.mean(c[:, 0][:, 0]) < (y - 50)][1:]

    contours.sort(key=lambda v: np.mean(v[:, 0][:, 0]))

    # Cover any issues again with single lines read as multiple lines
    # Join borders if they are generally on the same vertical line
    contours = join_borders(contours, 'v')

    # If first page, delete additional lines for header and footer
    if first_page:
        contours = contours[1:7] + contours[8:9]

    # Need to avoid false positives created by hole punch circles
    # 2000 chosen arbitrarily
    contours = [c for c in contours if cv2.contourArea(c) > 2000]

    return contours


def join_borders(borders, kind):
    '''
    In cases where lines are too thin / break, need to remove extra borders. 
    Checks if vertical or horizontal borders are similar, then removes whatever additional ones exist
    Accepts 'h' or 'v' for kind argument
    '''
    if kind == 'h':
        orient_idx = 1
    elif kind == 'v':
        orient_idx = 0
    else:
        raise 'Invalid kind given as argument; accepts h or v'

    joined_borders = []

    for idx, border in enumerate(borders):
        line_mean = np.mean(border[:, 0][:, orient_idx])

        if idx == len(borders) - 1:
            joined_borders.append(border)
        else:
            next_line = np.mean(borders[idx + 1][:, 0][:, orient_idx])

            if line_mean - 10 < next_line < line_mean + 10:
                # The next line is within the range of error for the existing line
                # so skip adding it.
                pass
            else:
                joined_borders.append(border)
    # Add the very last border back in, since that isn't caught by the check
    return joined_borders


def calculate_boxes(img, horizontal_borders, vertical_borders, debug=False, original_image=None):
    '''
    Calculates the coordinates of the bounding rectangles for each cell
    Returns list of ((left, top),(right, bottom)), along with table dimensions (x, y)
    '''
    x, y = img.shape
    # Has additional bottom line
    num_rows = len(horizontal_borders) - 1
    # Missing two far left/right borders
    num_cols = len(vertical_borders)
    # Reverse lists to iterate top-bottom, left-right.
    # horizontal_borders.reverse()
    # vertical_borders.reverse()

    coordinates = []

    # Get intersections and average lines
    # [1:] Stop prior to the bottom border
    for h_idx, h_line in enumerate(horizontal_borders[:-1]):
        top_cell_border = np.mean(h_line[:, 0][:, 1])
        bottom_cell_border = np.mean(horizontal_borders[h_idx + 1][:, 0][:, 1])

        # left_cell_border = 100
        # right_cell_border = y - 100
        for v_idx, v_line in enumerate(vertical_borders):
            # No far right or left gridlines, so need to correct for those
            # just stretch to end of page

            # Far left (start box)
            if v_idx == 0:
                left_cell_border = 5
                right_cell_border = np.mean(v_line[:, 0][:, 0])
            else:
                left_cell_border = np.mean(
                    vertical_borders[v_idx - 1][:, 0][:, 0])

            # Far right (end box)
            if v_idx == len(vertical_borders) - 1:
                left_cell_border = np.mean(
                    vertical_borders[v_idx - 1][:, 0][:, 0])
                right_cell_border = y - 90
            else:
                right_cell_border = np.mean(
                    vertical_borders[v_idx][:, 0][:, 0])

            cell_coordinates = ((left_cell_border, top_cell_border),
                                (right_cell_border, bottom_cell_border))

            if debug:
                draw_cell_borders(original_image, [cell_coordinates])

            coordinates.append(
                cell_coordinates)

    return coordinates, (num_rows, num_cols)


def borders_and_boxes(img, debug=False, original_image=None, first_page=False):
    h_lines = identify_horizontal_borders(img, first_page=first_page)
    v_lines = identify_vertical_borders(img, first_page=first_page)
    boxes, dim = calculate_boxes(
        img, h_lines, v_lines, debug=debug, original_image=original_image)
    return h_lines, v_lines, boxes, dim


def validate_boxes(img, cells, replace=True):
    '''
    Occasionally receive the tile cannot extend outside image error,
    which seems to be a problem with the shape of the box. Confirm
    that coords are within the img and not an impossible rectangle shape.
    If replace, provides blank box to not mess up dimensions.
    '''
    x, y = img.shape
    valid_boxes = []
    invalid_boxes = []
    # confirm that the coordinate is within image boundaries
    def boundary_check(c, b): return 0 < c < b
    # confirm that the box shape is correct
    # top left coords need to be smaller than bottom right coords
    def shape_check(
        cell): return cell[0][0] < cell[1][0] and cell[0][1] < cell[1][1]

    for cell in cells:
        valid = True
        if not shape_check(cell):
            valid = False
        for coordinate_pair in cell:
            for coord in coordinate_pair:
                # If either is not true, return false
                if not boundary_check(coord, x) and not boundary_check(coord, y):
                    valid = False

        if valid:
            valid_boxes.append(cell)
        else:
            if replace:
                valid_boxes.append(((0, 0), (1, 1)))
            invalid_boxes.append(cell)

    print("{} invalid boxes out of {}".format(len(invalid_boxes), len(cells)))

    return valid_boxes, invalid_boxes


def draw_cell_borders(img, coords, end_after=None, padding=5):
    '''
    Preview the drawn boxes on an image with padding and different colors
    '''
    img_copy = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    idx = 0

    if end_after:
        coords = coords[:end_after]

    for start, end in coords:
        x1, y1 = start
        x2, y2 = end
        # Move the top-left coordinate out, and the bottom-right coordinate in
        cv2.rectangle(img_copy, (round(x1) + padding, round(y1) + padding),
                      (round(x2) - padding, round(y2) - padding), colors[idx], 2)
        # Cycle through the colors
        if idx == 2:
            idx = 0
        else:
            idx += 1

    show(img_copy)
    return None


def crop_cells(img, coords):
    '''
    Crops the cells from the given list of rectangles
    Returns: list of cell coordinates
    '''
    cells = []

    for start, end in coords:
        x1, y1 = start
        x2, y2 = end
        cells.append(img[round(y1):round(y2), round(x1):round(x2)].copy())

    return cells


def ocr_cell(cell, psm=None):
    '''
    OCR a given cell and return the text
    '''
    try:
        if psm:
            ocr_dict = pytesseract.image_to_data(
                cell, lang='eng', output_type=Output.DICT, config=psm)
        else:
            ocr_dict = pytesseract.image_to_data(
                cell, lang='eng', output_type=Output.DICT)
        return ocr_dict['text']
    except SystemError as e:
        return ['READ_ERROR']


def process_text(text):
    '''
    Joins the list of text returned from the OCR_DICT.
    Existing Issue: Names and Addresses are in the same cell.
    Can likely implement regex based on first number to ID address(?)
    May be a number in an organization name, however.
    '''
    joined_text = ' '.join(text)
    return joined_text.strip()


def read_cells_on_page(cells, dim):
    '''
    Reads the cropped cells and returns a dataframe. Uses given dim to calculate
    rows and columns.
    '''
    row_text = []

    for cell in cells:
        text = ocr_cell(cell, psm='--psm 6')
        cleaned_text = process_text(text)
        row_text.append(cleaned_text)

    array_to_df = np.array(row_text).reshape(dim)

    return pd.DataFrame(array_to_df)


def read_document(document_path, start_page, end_page, validate_shapes=True, save_file=True, headers_on_first_page=True, rotate_img=True, reset_index=False):
    '''
    Note that start page must be the opening page of the Schedule I, e.g., have the header
    https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
    Returns dataframe of all pages. 
    header_on_first_page indicates if the start page uses the IRS 990 schedule i start page,
    where we need to ignore all of the headers and footers. If all pages are continuation pages,
    this should be false. 
    '''
    pytesseract.pytesseract.tesseract_cmd = "C:/Users/Jacob/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    page_dataframes = []
    doc_name = document_path.rsplit("\\")[-1].split('.')[0]
    # Convert page by page
    for page in range(start_page, end_page):
        try:
            first_page_ind = (
                True if page == start_page and headers_on_first_page else False)
            # Index into returned list
            original_image = convert_pdf_pages_to_imgs(
                document_path, start_page=page, end_page=page + 1)[0]

            # Rotate and make grayscale
            cleaned_image = clean_image(original_image, rotate_img=rotate_img)

            horizontal_lines = identify_horizontal_borders(
                cleaned_image, first_page=first_page_ind)
            vertical_lines = identify_vertical_borders(
                cleaned_image, first_page=first_page_ind)

            boxes, dim = calculate_boxes(
                cleaned_image, horizontal_lines, vertical_lines)

            if validate_shapes:
                boxes, invalid_boxes = validate_boxes(cleaned_image, boxes)

            page_cells = crop_cells(cleaned_image, boxes)

            print("Page: {}, {} h_lines, {} v_lines, {} page cells".format(
                page, len(horizontal_lines), len(vertical_lines), len(page_cells)))
            # try:
            page_df = read_cells_on_page(page_cells, dim)

            # Add additional human-readable checking information
            num_cols = page_df.shape[1]
            column_names = ['Name and Address', 'EIN',
                            'Designation', 'Amount', 'X', 'X2', 'Description']

            if num_cols > len(column_names):
                column_names.extend(['Col'] * (num_cols - len(column_names)))
            elif num_cols < len(column_names):
                page_df['Temp'] = None

            page_df.columns = column_names

            page_df['page_num'] = page
            page_df['file_name'] = doc_name

            page_dataframes.append(page_df)

        except Exception as e:
            print("Page {} failed to read. Passing.".format(page))
            print(e)
            pass

    if reset_index:
        [print(df.columns) for df in page_dataframes]
        all_dfs = pd.concat(page_dataframes, ignore_index=True)
    else:
        all_dfs = pd.concat(page_dataframes)

    if save_file:
        all_dfs.to_csv(doc_name + '.csv')

    return pd.concat(page_dataframes)


def convert_pdf_pages_to_imgs(file, start_page, end_page, save=False):
    '''
    Converts a PDF page to an image and saves it.
    Returns list of images.
    Convert to OpenCv format with [np.array(x) for x in image_list], for example
    '''
    image_list = convert_from_path(
        file, first_page=start_page, last_page=end_page)
    if save:
        for idx, img in enumerate(image_list):
            img.save('page' + str(idx) + '.png')
    else:
        return [np.array(i) for i in image_list]


def preprocess(img, kind):
    '''
    Pre processes the image slightly to correct for breaks in the cell borders.
    '''
    img = img.copy()
    if kind == 'v':
        kernel_shape = (3, 1)
    elif kind == 'h':
        kernel_shape = (1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    erode = cv2.erode(img, kernel, iterations=1)
    return erode


def process_cell(img):
    thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh
    return invert
