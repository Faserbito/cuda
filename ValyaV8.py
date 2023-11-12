from os import makedirs
import cv2
import numpy as np
import cvzone
import math
from datetime import datetime, timedelta
import openpyxl

from ultralytics import YOLO

from tkinter import *
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename, asksaveasfilename
from work_with_excel import start, work, diagram

def line_diag(path_save):
    import openpyxl
    from openpyxl.chart import BarChart, Reference
    import matplotlib.pyplot as plt
    import numpy as np

    # Создаем файл Excel
    wb = openpyxl.load_workbook(path_save)
    ws = wb.active

    # Создаем объект графика
    chart = LineChart()

    # Указываем диапазон данных для графика
    data_range = Reference(ws, min_col=7, min_row=2, max_col=11, max_row=3)
    chart.add_data(data_range, titles_from_data=True)

    # Добавляем график в Excel файл
    ws.add_chart(chart, "F7")

    # Сохраняем файл Excel
    wb.save(path_save)

def send_email(message):
    server = 'smtp.mail.ru'
    user = 'example_api@mail.ru'
    password = 'LkybB53PjruzpkPAjnFi'

    sender = 'example_api@mail.ru'
    to_address = 'ne4kin.zh@yandex.ru'
    subject = 'Вторжение в wi-fi сеть неизвестных устройств' 

    body = "\r\n".join((f"From: {user}", f"To: {to_address}", 
           f"Subject: {subject}", message))

    mail = smtplib.SMTP_SSL(server)
    mail.login(user, password)
    mail.sendmail(sender, to_address, body.encode('utf8'))
    mail.quit()

root = Tk()
root.withdraw()
filepath = askopenfilename(title = 'Выберите видео для загрузки', defaultextension='.mp4',
                  filetypes=[('MP4 Videos','*.mp4'),
                             ('All files','*.*')])
if filepath:
    print(f"Выбранный файл: {filepath}")
else: 
    print("Ничего не выбрано")
    exit()

path_save = asksaveasfilename(title = 'Сохранение в excel-файл', defaultextension='.xlsx',
                  filetypes=[('Excel files','*.xlsx')])
if not path_save:
    print("Ничего не выбрано")
    exit()
else:
    start(path_save)

model = YOLO("C:/Lf/ValyaV8/ValyaV8/runs/detect/train5/weights/best.pt")  # загрузите предварительно обученную модель YOLOv8n

cap = cv2.VideoCapture(filepath)

fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

classNames = ["person", "svarka", "vinuzhdenaya"]

while True:
    success, img = cap.read()
    
    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    ms = calc_timestamps[-1] + 1000/fps
    calc_timestamps.append(ms)
    
    # Создаем объект timedelta с миллисекундами
    delta = timedelta(milliseconds=ms)

    # Создаем временный объект datetime с нулевой датой и прибавляем delta
    base_time = datetime(1, 1, 1, 0, 0, 0)
    result_time = base_time + delta

    # Форматируем результат в строку "чч:мм:сс"
    result_time = result_time.strftime("%H:%M:%S")

    start = (10, 50)
    font_size = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 1
    text = "TIME: " + result_time
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(img, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    results = model(img, stream=True)
    
    for r in results:
        finded_objects = {}
        for box in r.boxes:
            if classNames[int(box.cls[0])] in finded_objects:
                finded_objects[classNames[int(box.cls[0])]] += 1
            else:
                finded_objects[classNames[int(box.cls[0])]] = 1
        boxes = r.boxes     
        for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0]*100))/100

                cls = box.cls[0]
                name = classNames[int(cls)]
                
                cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)
                
        
        status_left_side = ""
        status_right_side = ""

        pred_st_left_side = ""
        pred_st_right_side = ""
        
        if 'vinuzhdenaya' in finded_objects.keys() or 'svarka' in finded_objects.keys():
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = box.cls[0]
                name = classNames[int(cls)]
                        
                average_koord_x = (x1 + x2)/2
                if average_koord_x >= 640/2:
                    side = 'right'
                else:
                    side = 'left'
                
                pred_st_left_side = status_left_side
                pred_st_right_side = status_right_side
            
                data = ['', '', '', '']
                if name == "vinuzhdenaya":
                    if side == 'left':
                        status_left_side = "вынужденная"
                        if status_left_side != pred_st_left_side:
                            data[0] = result_time
                            data[1] = status_left_side
                            work(path_save, data)
                    elif side == 'right':
                        status_right_side = "вынужденная"
                        if status_right_side != pred_st_right_side:
                            data[2] = result_time
                            data[3] = status_right_side
                            work(path_save, data)
                elif name == "svarka":
                    if side == 'left':
                        status_left_side = "сварка"
                        if status_left_side != pred_st_left_side:
                            data[0] = result_time
                            data[1] = status_left_side
                            work(path_save, data)
                    elif side == 'right':
                        status_right_side = "сварка"
                        if status_right_side != pred_st_right_side:
                            data[2] = result_time
                            data[3] = status_right_side
                            work(path_save, data)
        else:
            if len(finded_objects) != 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = box.cls[0]
                    name = classNames[int(cls)]

                    average_koord_x = (x1 + x2)/2
                    if average_koord_x >= 640/2:
                        side = 'right'
                    else:
                        side = 'left'
                
                        pred_st_left_side = status_left_side
                        pred_st_right_side = status_right_side
            
                        data = ['', '', '', '']
                        if name == "person":
                            if side == 'left':
                                status_left_side = "простой"
                                if status_left_side != pred_st_left_side:
                                    data[0] = result_time
                                    data[1] = status_left_side
                                    work(path_save, data)
                            elif side == 'right':
                                status_right_side = "простой"
                                if status_right_side != pred_st_right_side:
                                    data[2] = result_time
                                    data[3] = status_right_side
                                    work(path_save, data)
            else:
                data[0] = result_time
                data[1] = "простой (на рабочих местах не найдено людей)"
                data[2] = result_time
                data[3] = "простой (на рабочих местах не найдено людей)"
                work(path_save, data)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

diagram(path_save)

line_diag()

message = f'Excel файл с данными о видео {filepath}.'
send_email(message)