def toggle_case_in_file(file):
    #try:
    content = file.read()  # ������ ����������� �����
    print(content)
    toggled_content = content.decode("utf-8").swapcase()  # ������ ������� ���� � ������ � ����������� � ������
    print(toggled_content)

    # ���������� ���������� ����� ������� � ����
    with open(file.filename, 'w') as updated_file:
        updated_file.write(toggled_content)

    return "������� ���� � ����� ��� �������"
    # except Exception as e:
    #     return str(e)  # ���������� ��������� �� ������, ���� ���-�� ����� �� ���
