def toggle_case_in_file(file):
    try:
        content = file.read()  # Чтение содержимого файла
        toggled_content = content.swapcase()  # Меняем регистр букв в тексте

        # Записываем измененный текст обратно в файл
        with open(file.filename, 'w') as updated_file:
            updated_file.write(toggled_content)

        return "everything is ok"
    except Exception as e:
        return str(e) 