import flask
from flask import request, make_response, jsonify
from flask_cors import CORS
import numpy as np
import io
import os
import json
import base64
from scipy import misc, fft
from PIL import Image
import cv2
from skimage import data, filters, util, exposure
import math

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config["DEBUG"] = True


def readImage(request):
    image = request.files['image']
    fileFormat = image.filename.split('.')[-1].upper()
    if fileFormat.upper() == 'JPG':
        fileFormat = "JPEG"
    data = np.fromstring(image.read(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)
    cv2.imwrite('original.'+fileFormat, img)
    return img, fileFormat


def convertImage(img, fileFormat):
    conv = Image.fromarray(img)
    buff = io.BytesIO()
    conv.save(buff, format=fileFormat)
    newImage = base64.b64encode(buff.getvalue()).decode("utf-8")
    return newImage


def saveArrayImage(name, image):
    filename = name+'.'+"JPEG"
    cv2.imwrite(filename, np.asarray(image))
    newImageFile = Image.open(filename)
    newImageArray = np.asarray(newImageFile)
    newImage = convertImage(newImageArray, "JPEG")
    return newImage


def convertTo256(image):
    image = np.asarray(image)
    imgFinal = []
    for i in range(0, len(image)):
        imgFinal.append([])
        for j in range(0, len(image[i])):
            num = max(1, image[i, j]*255)
            imgFinal[i].append(int(math.floor(num)))
    return imgFinal


def convertTo256Scale(image):
    image = np.asarray(image)
    m = np.max(image)
    imgFinal = []
    for i in range(0, len(image)):
        imgFinal.append([])
        for j in range(0, len(image[i])):
            num = max(0, (image[i, j]/m)*255)
            imgFinal[i].append(int(math.floor(num)))
    return imgFinal


def convertToRealWithScale(image):
    imgScale = []
    imgImag = np.angle(image)
    imgReal = np.abs(image)
    for i in range(0, len(image)):
        imgScale.append([])
        for j in range(0, len(image[i])):
            imgScale[i].append(
                int(math.floor(math.log2(1+abs(image[i, j])))))
    imgScale = np.asarray(imgScale)
    m = np.max(imgScale)
    print(m)
    imgFinal = []
    for i in range(0, len(imgScale)):
        imgFinal.append([])
        for j in range(0, len(imgScale[i])):
            num = math.floor(((int(imgScale[i, j])/m)*256))
            imgFinal[i].append(num)
    return imgFinal, imgImag, imgReal


def getMagAndPhase(image):
    return np.abs(image), np.angle(image)


def inverse(real, imag):
    img = np.multiply(real, np.exp(1j*imag))
    return img


def bandReject(image, nx, ny):
    imageFinal = []
    if nx >= 1 or ny >= 1:
        threshHold = 0
        if nx > 1 or ny > 1:
            threshHold = 1
        if nx > 2 or ny > 2:
            threshHold = 2
        height = len(image)
        width = len(image[0])
        for i in range(0, len(image)):
            imageFinal.append([])
            for j in range(0, len(image[i])):
                distance = math.sqrt((j-(width/2))**2+(i-(height/2))**2)
                actulaDistance = math.sqrt(nx**2+ny**2)
                if distance <= actulaDistance+threshHold and distance >= actulaDistance-threshHold:
                    imageFinal[i].append(0)
                else:
                    imageFinal[i].append(image[i, j])
    else:
        imageFinal = image
    return imageFinal


def notchFilter(image, nx, ny):
    imageFinal = []
    centerHeight = len(image)/2
    centerWidth = len(image[0])/2
    fx1 = centerWidth-nx
    fy1 = centerHeight+ny
    fx2 = centerWidth+nx
    fy2 = centerHeight-ny
    print(ny, nx)
    for i in range(0, len(image)):
        imageFinal.append([])
        for j in range(0, len(image[i])):
            if ((i == fy1 or i == fy2) and ny > 0) or ((j == fx1 or j == fx2) and nx > 0):
                imageFinal[i].append(0)
            else:
                imageFinal[i].append(image[i, j])
    return imageFinal


def medianMask(size):
    mask = []
    for i in range(0, size):
        subMask = []
        for j in range(0, size):
            subMask.append(1)
        mask.append(subMask)
    return mask


@app.route('/', methods=['GET'])
def home():
    return "<h1>Hello...</h1>"


@app.route('/upload', methods=['POST'])
def upload():
    img, fileFormat = readImage(request)
    newImage = convertImage(img, fileFormat)
    hist = np.histogram(img, 256)
    return jsonify({'newImage': newImage, 'hist': hist[0].tolist()})


@app.route('/histEq', methods=['GET'])
def histEq():
    file = Image.open('original.JPEG')
    image = np.asarray(file)
    imgHistEq = exposure.equalize_hist(image)
    imgHistEqFinal = convertTo256(imgHistEq)
    newImage = saveArrayImage('histEq', imgHistEqFinal)
    hist = np.histogram(imgHistEq, 256)
    return jsonify({'img': newImage, 'hist': hist[0].tolist()})


@app.route('/sobel_and_lablace', methods=['POST'])
def sobelAndLablace():
    file = Image.open('original.JPEG')
    image = np.asarray(file)
    mode = request.json['mode']
    print(mode)
    if mode == 'sobel':
        sobel = filters.sobel(image)
        sobelFinal = convertTo256(sobel)
        newImage = saveArrayImage('sobel', sobelFinal)
    elif mode == 'laplace':
        lablace = filters.laplace(image)
        lablaceFinal = convertTo256(lablace)
        newImage = saveArrayImage('laplace', lablaceFinal)
    return jsonify({'img': newImage})


@app.route('/fourier', methods=['GET'])
def fourier():
    file = Image.open('original.JPEG')
    image = np.asarray(file)
    fourier = fft.fftshift(fft.fft2(image))
    fourierReal, imag, real = convertToRealWithScale(fourier)
    newImage = saveArrayImage('fourier', fourierReal)
    return jsonify({'img': newImage})


@app.route('/salt_and_pepper', methods=['POST'])
def saltAndPepper():
    file = Image.open('original.JPEG')
    image = np.asarray(file)
    amount = request.json['amount']
    sAndPImage = util.noise.random_noise(
        image, mode='s&P', amount=float(amount))
    sAndPImageFinal = convertTo256(sAndPImage)
    newImage = saveArrayImage('s&P', sAndPImageFinal)
    size = request.json['size']
    mask = medianMask(int(size))
    median = filters.rank.median(np.asarray(
        sAndPImageFinal), selem=np.asarray(mask))
    newFilterImage = saveArrayImage('median', median)
    return jsonify({'img': newImage, 'filter': newFilterImage})


@app.route('/periodic', methods=['POST'])
def periodic():
    file = Image.open('original.JPEG')
    width, height = file.size
    image = np.asarray(file)
    x, y = np.meshgrid(range(0, width), range(0, height))
    nx = request.json['nx']
    ny = request.json['ny']
    Wx = np.max(x)
    Wy = np.max(y)
    fx = int(nx)/Wx
    fy = int(ny)/Wy
    mode = request.json['mode']
    if mode == 'sin':
        pxy = np.sin(2*math.pi*fx*x + 2*math.pi*fy*y)+1
    elif mode == 'cos':
        pxy = np.cos(2*math.pi*fx*x + 2*math.pi*fy*y)+1
    nois = convertTo256(pxy)
    noisyImage = image+np.asarray(nois)
    noisyImageScale = convertTo256Scale(noisyImage)
    noisyImageFinal = saveArrayImage('periodic', noisyImageScale)
    fourier = fft.fftshift(fft.fft2(noisyImageScale))
    fourierFinal, imag, real = convertToRealWithScale(fourier)
    newImage = saveArrayImage('fourierPeriodic', np.asarray(fourierFinal))
    return jsonify({'img': noisyImageFinal, 'fourier': newImage})


@app.route('/filter', methods=['POST'])
def filter():
    file = Image.open('fourierPeriodic.JPEG')
    image = np.asarray(file)
    per = Image.open('periodic.JPEG')
    imagep = np.asarray(per)
    selectedFilter = request.json['filter']
    nx = int(request.json['nx'])
    ny = int(request.json['ny'])
    newImage = []
    inv = []
    f = fft.fftshift(fft.fft2(imagep))
    imager, imagei = getMagAndPhase(f)
    if selectedFilter == 'band':
        newImage = saveArrayImage(
            'band', np.asarray(bandReject(image, nx, ny)))
        inv = inverse(np.asarray(bandReject(imager, nx, ny)), imagei)
    elif selectedFilter == 'notch':
        newImage = saveArrayImage(
            'notch', np.asarray(notchFilter(image, nx, ny)))
        inv = inverse(np.asarray(notchFilter(imager, nx, ny)), imagei)
    else:
        newImage = saveArrayImage('fourierPeriodic', image)
        inv = inverse(imager, imagei)
    invF = fft.ifft2(inv)
    newImageInv = saveArrayImage('inverse', np.abs(invF))
    return jsonify({'img': newImage, 'inv': newImageInv})


@app.route('/mask', methods=['POST'])
def mask():
    click = request.json['click']
    if click == 0:
        file = Image.open('fourierPeriodic.JPEG')
        fileOrg = Image.open('periodic.JPEG')
    elif click == 1:
        file = Image.open('mask.JPEG')
        fileOrg = Image.open('maskOrg.JPEG')
    width, height = file.size
    x = request.json['x']
    y = request.json['y']
    h = request.json['height']
    w = request.json['width']
    fx = math.floor(x/(w/width))
    fy = math.floor(y/(h/height))
    image = np.asarray(file)
    imageOrg = np.asarray(fileOrg)
    f = fft.fftshift(fft.fft2(imageOrg))
    imager, imagei = getMagAndPhase(f)
    editedImage = image.copy()
    editedImage[fy, fx] = 0
    editedImageOrg = imager.copy()
    editedImageOrg[fy, fx] = 0
    editedImageFinal = saveArrayImage('mask', editedImage)
    inv = inverse(np.asarray(editedImageOrg), imagei)
    invF = fft.ifft2(inv)
    newImageInv = saveArrayImage('maskOrg', np.abs(invF))

    return jsonify({'img': editedImageFinal, 'inv': newImageInv})


@app.route('/clear', methods=['DELETE'])
def clear():
    if os.path.exists('original.JPEG'):
        os.remove('original.JPEG')
    if os.path.exists('histEq.JPEG'):
        os.remove('histEq.JPEG')
    if os.path.exists('fourier.JPEG'):
        os.remove('fourier.JPEG')
    if os.path.exists('band.JPEG'):
        os.remove('band.JPEG')
    if os.path.exists('fourierPeriodic.JPEG'):
        os.remove('fourierPeriodic.JPEG')
    if os.path.exists('inverse.JPEG'):
        os.remove('inverse.JPEG')
    if os.path.exists('mask.JPEG'):
        os.remove('mask.JPEG')
    if os.path.exists('maskOrg.JPEG'):
        os.remove('maskOrg.JPEG')
    if os.path.exists('median.JPEG'):
        os.remove('median.JPEG')
    if os.path.exists('notch.JPEG'):
        os.remove('notch.JPEG')
    if os.path.exists('periodic.JPEG'):
        os.remove('periodic.JPEG')
    if os.path.exists('s&p.JPEG'):
        os.remove('s&p.JPEG')
    if os.path.exists('sobel.JPEG'):
        os.remove('sobel.JPEG')
    if os.path.exists('laplace.JPEG'):
        os.remove('laplace.JPEG')

    return jsonify({'meassage': 'success'})


app.run()
