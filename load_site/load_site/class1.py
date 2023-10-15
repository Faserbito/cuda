from flask import Flask, jsonify

def toggle_case_in_file(file):
    # Для считывания PDF
    import PyPDF2
    # Для анализа структуры PDF и извлечения текста
    from pdfminer.high_level import extract_pages, extract_text
    from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
    # Для извлечения текста из таблиц в PDF
    import pdfplumber
    # Для извлечения изображений из PDF
    from PIL import Image
    from pdf2image import convert_from_path
    # Для выполнения OCR, чтобы извлекать тексты из изображений 
    import pytesseract 
    # Для удаления дополнительно созданных файлов
    import os
    import glob

    # Создаём функцию для вырезания элементов изображений из PDF
    def crop_image(element, pageObj):
        # Получаем координаты для вырезания изображения из PDF
        [image_left, image_top, image_right, image_bottom] = [element.x0,element.y0,element.x1,element.y1] 
        # Обрезаем страницу по координатам (left, bottom, right, top)
        pageObj.mediabox.lower_left = (image_left, image_bottom)
        pageObj.mediabox.upper_right = (image_right, image_top)
        # Сохраняем обрезанную страницу в новый PDF
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_writer.add_page(pageObj)
        # Сохраняем обрезанный PDF в новый файл
        with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
            cropped_pdf_writer.write(cropped_pdf_file)

    # Создаём функцию для преобразования PDF в изображения
    def convert_to_images(input_file,):
        images = convert_from_path(input_file)
        image = images[0]
        output_file = "PDF_image.png"
        image.save(output_file, "PNG")

    # Создаём функцию для считывания текста из изображений
    def image_to_text(image_path):
        # Считываем изображение
        img = Image.open(image_path)
        # Извлекаем текст из изображения
        text = pytesseract.image_to_string(img)
        return text

    # Преобразуем таблицу в соответствующий формат
    def table_converter(table):
        table_string = ''
        # Итеративно обходим каждую строку в таблице
        for row_num in range(len(table)):
            row = table[row_num]
            # Удаляем разрыв строки из текста с переносом
            cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
            # Преобразуем таблицу в строку
            table_string += ('|'+'|'.join(cleaned_row)+'|'+'\n')
        # Удаляем последний разрыв строки
        table_string = table_string[:-1]
        return table_string

    # Укажите путь к папке с PDF файлами
    # pdf_folder = 'C:/Users/faser/Downloads/train_dataset_dataset/dataset/ФМУ-76/'
    # pdf_files = glob.glob(pdf_folder + '/*.pdf')

    pdf_files = ['C:/Users/faser/Downloads/train_dataset_dataset/dataset/ФМУ-76/' + file.filename]
    print(pdf_files)
    
    #uploaded_file = request.files['file']

    #--------------------------------------------------------------------------------------------------------------------------------
    #pdf_files = ['C:/Users/faser/Downloads/train_dataset_dataset/dataset/ФМУ-76/ФМУ76_292_27.03.2023.pdf']

    # Создаём функцию для извлечения текста
    def text_extraction(element):
        # Извлекаем текст из вложенного текстового элемента
        line_text = element.get_text()
    
        # Находим форматы текста
        # Инициализируем список со всеми форматами, встречающимися в строке текста
        line_formats = []
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                # Итеративно обходим каждый символ в строке текста
                for character in text_line:
                    if isinstance(character, LTChar):
                        # Добавляем к символу название шрифта
                        line_formats.append(character.fontname)
                        # Добавляем к символу размер шрифта
                        line_formats.append(character.size)
        # Находим уникальные размеры и названия шрифтов в строке
        format_per_line = list(set(line_formats))
    
        # Возвращаем кортеж с текстом в каждой строке вместе с его форматом
        return (line_text, format_per_line)
    
    # Извлечение таблиц из страницы
    def extract_table(pdf_path, page_num, table_num):
        # Открываем файл pdf
        pdf = pdfplumber.open(pdf_path)
        # Находим исследуемую страницу
        table_page = pdf.pages[page_num]
        # Извлекаем соответствующую таблицу
        table = table_page.extract_tables()[table_num]
        return table

    # создаём объект файла PDF
    for pdf_path in pdf_files:
        pdfFileObj = open(pdf_path, 'rb')
        # создаём объект считывателя PDF
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)

        # Создаём словарь для извлечения текста из каждого изображения
        text_per_page = {}
        # Извлекаем страницы из PDF
        for pagenum, page in enumerate(extract_pages(pdf_path)):
    
            # Инициализируем переменные, необходимые для извлечения текста со страницы
            pageObj = pdfReaded.pages[pagenum]
            page_text = []
            line_format = []
            text_from_images = []
            text_from_tables = []
            page_content = []
            # Инициализируем количество исследованных таблиц
            table_num = 0
            first_element = True
            table_extraction_flag = False
            # Открываем файл pdf
            pdf = pdfplumber.open(pdf_path)
            # Находим исследуемую страницу
            page_tables = pdf.pages[pagenum]
            # Находим количество таблиц на странице
            tables = page_tables.find_tables()
            
            #Проыерка на валидность 
            cancellation = False
            first = True
            if "ФМУ76" in file.filename:
                # Работа с документом типа ФМУ-76
                for i in range(len(tables)):
                    text = tables[i].extract()
                    if text[0][0] == "Код":
                        if text[2][0] == "-" or text[2][0] == " " or text[3][0] == "-" or text[3][0] == " ":
                            cancellation = True
                    if text[0][0] == "Номер":
                        if text[1][0] == "-" or text[1][0] == " " or text[1][1] == "-" or text[1][1] == " ":
                            cancellation = True
                    if text[0][0].replace("\n", " ") == "Структурное подразделение (цех, участок и др.)":
                        if text[2][0] == "-" or text[2][0] == " " or text[2][1] == "-" or text[2][1] == " " or text[2][2] == "-" or text[2][2] == " " or text[2][3] == "-" or text[2][3] == " ":
                            cancellation = True
                    if text[0][0].replace("\n", " ") == 'Технический счет 32 "Затраты"':
                        for j in range(3, len(text)):
                            if text[j][0] == "-" or text[j][0] == " " or text[j][1] == "-" or text[j][1] == " " or text[j][2] == "-" or text[j][2] == " " or text[j][4] == "-" or text[j][4] == " " or text[j][5] == "-" or text[j][5] == " " or text[j][6] == "-" or text[j][6] == " " or text[j][7] == "-" or text[j][7] == " " or text[j][8] == "-" or text[j][8] == " " or text[j][9] == "-" or text[j][9] == " " or text[j][10] == "-" or text[j][10] == " " or text[j][15] == "-" or text[j][15] == " " or text[j][16] == "-" or text[j][16] == " " or text[j][17] == "-" or text[j][17] == " ":
                                cancellation = True                                
            elif "М11" in file.filename:
                # Работа с документом типа M-11
                first = False
                for i in range(len(tables)):
                    text = tables[i].extract()
                    if text[0][0] == "Коды":
                        if text[2][0] == "-" or text[2][0] == " " or text[3][0] == "-" or text[3][0] == " ":
                            cancellation = True
                    if text[0][0].replace("\n", " ").replace("- ", "") == "Дата составления":
                        dictionary[4] = (f"{text[0][0]}".replace("\n", " ").replace("- ", ""), f"{text[2][0]}")
                        dictionary[5] = (f"{text[0][1]}".replace("\n", " ").replace("- ", ""), f"{text[2][1]}")
                        dictionary[6] = (f"{text[1][2]}".replace("\n", " ").replace("- ", ""), f"{text[2][2]}")
                        dictionary[7] = (f"{text[1][3]}".replace("\n", " ").replace("- ", ""), f"{text[2][3]}")
                        dictionary[8] = (f"{text[1][4]}".replace("\n", " ").replace("- ", ""), f"{text[2][4]}")
                        dictionary[9] = (f"{text[1][5]}".replace("\n", " ").replace("- ", ""), f"{text[2][5]}")
                        dictionary[28] = (f"{text[1][6]}".replace("\n", " ").replace("- ", ""), f"{text[2][6]}")
                        dictionary[25] = (f"{text[1][7]}".replace("\n", " ").replace("- ", ""), f"{text[2][7]}")
                        dictionary[10] = (f"{text[0][8]}".replace("\n", " ").replace("- ", ""), f"{text[2][8]}")
                    if text[0][0].replace("\n", " ").replace("- ", "") == 'Корреспондирующий счет':
                        for j in range(3, len(text)):
                            dictionary[29] = (f"{text[0][0]}".replace("\n", " ").replace("- ", ""), f"{text[j][0]}")
                            dictionary[14] = (f"{text[1][1]}".replace("\n", " ").replace("- ", ""), f"{text[j][1]}")
                            dictionary[15] = (f"{text[1][2]}".replace("\n", " ").replace("- ", ""), f"{text[j][2]}")
                            dictionary[16] = (f"{text[0][3]}".replace("\n", " ").replace("- ", ""), f"{text[j][3]}")
                            dictionary[17] = (f"{text[0][4]}".replace("\n", " ").replace("- ", ""), f"{text[j][4]}")
                            dictionary[30] = (f"{text[0][5]}".replace("\n", " ").replace("- ", ""), f"{text[j][5]}")
                            dictionary[31] = (f"{text[0][6]}".replace("\n", " ").replace("- ", ""), f"{text[j][6]}")
                            dictionary[18] = (f"{text[1][7]}".replace("\n", " ").replace("- ", ""), f"{text[j][7]}")
                            dictionary[19] = (f"{text[1][8]}".replace("\n", " ").replace("- ", ""), f"{text[j][8]}")
                            dictionary[20] = (f"{text[1][9]}".replace("\n", " ").replace("- ", ""), f"{text[j][9]}")
                            dictionary[21] = (f"{text[1][10]}".replace("\n", " ").replace("- ", ""), f"{text[j][10]}")
                            dictionary[26] = (f"{text[0][11]}".replace("\n", " ").replace("- ", ""), f"{text[j][11]}")
                            dictionary[27] = (f"{text[0][12]}".replace("\n", " ").replace("- ", ""), f"{text[j][12]}")
                            dictionary[22] = (f"{text[0][13]}".replace("\n", " ").replace("- ", ""), f"{text[j][13]}")
                            dictionary[32] = (f"{text[0][14]}".replace("\n", " ").replace("- " , ""), f"{text[j][14]}")
                            dictionary[33] = (f"{text[0][15]}".replace("\n", " ").replace("- ", ""), f"{text[j][15]}")

            dictionary = {}
            
            # Находим все элементы
            page_elements = [(element.y1, element) for element in page._objs]
            # Сортируем все элементы по порядку нахождения на странице
            #page_elements.sort(key=lambda a: a[0], reverse=True)

            # Находим элементы, составляющие страницу
            for i,component in enumerate(page_elements):
                # Извлекаем положение верхнего края элемента в PDF
                pos = component[0]
                # Извлекаем элемент структуры страницы
                element = component[1]

            # Создаём ключ для словаря
            dctkey = 'Page_' + str(pagenum)
            # Добавляем список списков как значение ключа страницы
            text_per_page[dctkey] = [page_text, line_format, text_from_images,text_from_tables, page_content]

        # Закрываем объект файла pdf
        pdfFileObj.close()

        result = ''.join(text_per_page['Page_0'][4])
        #print(result)
        if cancellation:
            if first:
                return "Аннулирован; Несоответствие документу ФУЭ76"
            else:
                return "Аннулирован; Несоответствие документу М11"
        else:
            return "Документ соответстует требованиям"  
        
















# # Словарь для ФМУ-76
            # for i in range(len(tables)):
            #     text = tables[i].extract()
            #     if text[0][0] == "Код":
            #         dictionary[3] = ("Организация", f"{text[2][0]}")
            #         dictionary[4] = ("Структурное подразделение", f"{text[3][0]}")
            #     if text[0][0] == "Номер":
            #         dictionary[1] = ("Номер документа", f"{text[1][0]}")
            #         dictionary[2] = ("Дата составления", f"{text[1][1]}")
            #     if text[0][0].replace("\n", " ") == "Структурное подразделение (цех, участок и др.)":
            #         dictionary[5] = (f"{text[0][0]}".replace("\n", " "), f"{text[2][0]}")
            #         dictionary[6] = (f"{text[0][1]}".replace("\n", " "), f"{text[2][1]}")
            #         dictionary[28] = (f"{text[1][2]}".replace("\n", " "), f"{text[2][2]}")
            #         dictionary[27] = (f"{text[1][3]}".replace("\n", " "), f"{text[2][3]}")
            #     if text[0][0].replace("\n", " ") == 'Технический счет 32 "Затраты"':
            #         for j in range(3, len(text)):
            #             dictionary[31] = (f"{text[0][0]}".replace("\n", " "), f"{text[j][0]}")
            #             dictionary[32] = (f"{text[0][1]}".replace("\n", " "), f"{text[j][1]}")
            #             dictionary[29] = (f"{text[1][2]}".replace("\n", " "), f"{text[j][2]}")
            #             dictionary[30] = (f"{text[1][3]}".replace("\n", " "), f"{text[j][3]}")
            #             dictionary[11] = (f"{text[1][4]}".replace("\n", " "), f"{text[j][4]}")
            #             dictionary[12] = (f"{text[1][5]}".replace("\n", " "), f"{text[j][5]}")
            #             dictionary[33] = ("Характеристика", "-")
            #             dictionary[34] = (f"{text[0][6]}".replace("\n", " "), f"{text[j][6]}")
            #             dictionary[13] = (f"{text[1][7]}".replace("\n", " "), f"{text[j][7]}")
            #             dictionary[14] = (f"{text[1][8]}".replace("\n", " "), f"{text[j][8]}")
            #             dictionary[15] = (f"{text[0][9]}".replace("\n", " "), f"{text[j][9]}")
            #             dictionary[16] = (f"{text[1][10]}".replace("\n", " "), f"{text[j][10]}")
            #             dictionary[17] = (f"{text[1][11]}".replace("\n", " "), f"{text[j][11]}")
            #             dictionary[18] = (f"{text[1][12]}".replace("\n", " "), f"{text[j][12]}")
            #             dictionary[19] = (f"{text[0][13]}".replace("\n", " "), f"{text[j][13]}")
            #             dictionary[20] = (f"{text[0][14]}".replace("\n", " "), f"{text[j][14]}")
            #             dictionary[21] = (f"{text[0][15]}".replace("\n", " "), f"{text[j][15]}")
            #             dictionary[35] = (f"{text[0][16]}".replace("\n", " "), f"{text[j][16]}")
            #             from work_with_excel import work

            #             work('FMU-76', dictionary)            

            # Словарь для М-11
            # for i in range(len(tables)):
            #     text = tables[i].extract()
            #     if text[0][0] == "Коды":
            #         dictionary[2] = ("Организация", f"{text[2][0]}")
            #         dictionary[3] = ("Структурное подразделение", f"{text[3][0]}")
            #     if text[0][0].replace("\n", " ").replace("- ", "") == "Дата составления":
            #         dictionary[4] = (f"{text[0][0]}".replace("\n", " ").replace("- ", ""), f"{text[2][0]}")
            #         dictionary[5] = (f"{text[0][1]}".replace("\n", " ").replace("- ", ""), f"{text[2][1]}")
            #         dictionary[6] = (f"{text[1][2]}".replace("\n", " ").replace("- ", ""), f"{text[2][2]}")
            #         dictionary[7] = (f"{text[1][3]}".replace("\n", " ").replace("- ", ""), f"{text[2][3]}")
            #         dictionary[8] = (f"{text[1][4]}".replace("\n", " ").replace("- ", ""), f"{text[2][4]}")
            #         dictionary[9] = (f"{text[1][5]}".replace("\n", " ").replace("- ", ""), f"{text[2][5]}")
            #         dictionary[28] = (f"{text[1][6]}".replace("\n", " ").replace("- ", ""), f"{text[2][6]}")
            #         dictionary[25] = (f"{text[1][7]}".replace("\n", " ").replace("- ", ""), f"{text[2][7]}")
            #         dictionary[10] = (f"{text[0][8]}".replace("\n", " ").replace("- ", ""), f"{text[2][8]}")
            #     if text[0][0].replace("\n", " ").replace("- ", "") == 'Корреспондирующий счет':
            #         for j in range(3, len(text)):
            #             dictionary[29] = (f"{text[0][0]}".replace("\n", " ").replace("- ", ""), f"{text[j][0]}")
            #             dictionary[14] = (f"{text[1][1]}".replace("\n", " ").replace("- ", ""), f"{text[j][1]}")
            #             dictionary[15] = (f"{text[1][2]}".replace("\n", " ").replace("- ", ""), f"{text[j][2]}")
            #             dictionary[16] = (f"{text[0][3]}".replace("\n", " ").replace("- ", ""), f"{text[j][3]}")
            #             dictionary[17] = (f"{text[0][4]}".replace("\n", " ").replace("- ", ""), f"{text[j][4]}")
            #             dictionary[30] = (f"{text[0][5]}".replace("\n", " ").replace("- ", ""), f"{text[j][5]}")
            #             dictionary[31] = (f"{text[0][6]}".replace("\n", " ").replace("- ", ""), f"{text[j][6]}")
            #             dictionary[18] = (f"{text[1][7]}".replace("\n", " ").replace("- ", ""), f"{text[j][7]}")
            #             dictionary[19] = (f"{text[1][8]}".replace("\n", " ").replace("- ", ""), f"{text[j][8]}")
            #             dictionary[20] = (f"{text[1][9]}".replace("\n", " ").replace("- ", ""), f"{text[j][9]}")
            #             dictionary[21] = (f"{text[1][10]}".replace("\n", " ").replace("- ", ""), f"{text[j][10]}")
            #             dictionary[26] = (f"{text[0][11]}".replace("\n", " ").replace("- ", ""), f"{text[j][11]}")
            #             dictionary[27] = (f"{text[0][12]}".replace("\n", " ").replace("- ", ""), f"{text[j][12]}")
            #             dictionary[22] = (f"{text[0][13]}".replace("\n", " ").replace("- ", ""), f"{text[j][13]}")
            #             dictionary[32] = (f"{text[0][14]}".replace("\n", " ").replace("- " , ""), f"{text[j][14]}")
            #             dictionary[33] = (f"{text[0][15]}".replace("\n", " ").replace("- ", ""), f"{text[j][15]}")
            #             from work_with_excel import work

            #             work('M-11', dictionary)