from flask import Flask, render_template, request, redirect, url_for
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from keras_preprocessing.image import load_img

from flask_mysqldb import MySQL
from datetime import datetime


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'vehicles'

mysql = MySQL(app)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    img = cv2.imread(str(image_path))  # -----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text = str(result[0][-2])
    now = datetime.now()

    if request.method == "POST":
        rnumber = text
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO vehicles_list (rnumber, time) VALUES (%s, %s)", (rnumber, time))
        mysql.connection.commit()

    entranceTime = time
    return render_template('index.html', prediction=text, entranceTime=entranceTime)

# exit gate
@app.route("/forward/", methods=['POST'])
def move_forward():
    imagefile = request.files['imagefile2']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    img = cv2.imread(str(image_path))  # -----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text = str(result[0][-2])
    now = datetime.now()

    if request.method == "POST":
        rnumber = text
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        cur = mysql.connection.cursor()
        # cur.execute("INSERT INTO vehicles_list (rnumber, time) VALUES (%s, %s)", (rnumber, time))
        cur.execute("SELECT * FROM vehicles_list WHERE rnumber=%s", [rnumber])
        data = cur.fetchone()
        entranceTime = data[2]
        entranceTime= datetime.strptime(str(entranceTime), '%Y-%m-%d %H:%M:%S')
        exitTime =datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        duration = exitTime- entranceTime 
        totalSeconds = duration.total_seconds()
        charge = totalSeconds/10
        mysql.connection.commit()

    # writing to history
    if request.method == "POST":
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO vehicles_history (rnumber, entrancetime, exittime, duration, charge) VALUES (%s, %s, %s, %s, %s)", (rnumber, entranceTime, exitTime, duration, charge))
        mysql.connection.commit()

    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM vehicles_list WHERE rnumber=%s", [rnumber])
    mysql.connection.commit()

    return render_template('index.html', viewBill = True, prediction2=rnumber, entranceTime=entranceTime,
                           exitTime=exitTime, duration=duration,
                           charge = charge)
    

@app.route('/history', methods=['GET'])
def history():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM vehicles_history")
    data = cur.fetchall()
    cur.close()
    # mysql.connection.commit()
    return render_template('history.html', vehicleHistory = data)

@app.route('/present', methods=['GET'])
def present():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM vehicles_list")
    data = cur.fetchall()
    cur.close()
    return render_template('present.html', vehicleList = data)


if __name__ == '__main__':
    app.run( debug=True)
