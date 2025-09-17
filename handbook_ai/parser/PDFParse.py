from unstructured.partition.pdf import partition_pdf
import os
import fitz


def unstructured_pdf(file,images_output_dir):

    # Extract images, tables from a PDF file.
    file_name = file.split('/')[-1].split('.')[0]

    if not os.path.exists(images_output_dir + '/' + file_name):
        os.makedirs(images_output_dir + '/' + file_name)

    raw_pdf_elements = partition_pdf(
        filename=file,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=images_output_dir + '/' + file_name,
    )

    # Categorize extracted elements from a PDF into tables and texts.
    # raw_pdf_elements: List of unstructured.documents.elements
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return texts, tables


def PyMuPDF_pdf(file,images_output_dir):

    # Extract images, tables from a PDF file.
    file_name = file.split('/')[-1].split('.')[0]
    imgs_path = images_output_dir + '/' + file_name

    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    doc = fitz.open(file)

    contents = ''
    tables = []
    cnt_dict = {}
    imgs_dict = {}

    for page in doc:
        contents = contents + page.get_text('text')

        img_list = page.get_images()
        for img in img_list:
            loc_key = '_'.join(str(x) for x in list(fitz.Rect(img[:4])))
            if loc_key not in cnt_dict.keys():
                cnt_dict[loc_key] = 1
                imgs_dict[loc_key] = [img]
            else:
                cnt_dict[loc_key] += 1
                imgs_dict[loc_key].append(img)

        tabs_list = page.find_tables()
        for table in tabs_list.tables:
            tables.append(table.to_markdown())

    dup = 3
    i = 0
    for key in imgs_dict.keys():
        if cnt_dict[key] < dup:
            for img in imgs_dict[key]:
                pix = fitz.Pixmap(doc,img[0])
                save_name = imgs_path + "{}.png".format(i)
                pix.save(save_name)
                i += 1

    return [contents], tables