def toggle_case_in_file(file):
    try:
        content = file.read()  # ������ ����������� �����
        toggled_content = content.swapcase()  # ������ ������� ���� � ������

        # ���������� ���������� ����� ������� � ����
        with open(file.filename, 'w') as updated_file:
            updated_file.write(toggled_content)

        return "everything is ok"
    except Exception as e:
        return str(e) 