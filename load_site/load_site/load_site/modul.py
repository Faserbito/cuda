def toggle_case_in_file(file):
    try:
        content = file.read()  # Чтение содержимого файла
        toggled_content = content.swapcase()  # Меняем регистр букв в тексте

        # Записываем измененный текст обратно в файл
        with open(file.filename, 'w') as updated_file:
            updated_file.write(toggled_content)

        return "Регистр букв в файле был изменен"
    except Exception as e:
        return str(e)  # Возвращаем сообщение об ошибке, если что-то пошло не так

