# Anti-Spoofing Face Recognition

## Definition

Face recognition can be done by extracting face features. However, face recognition can be spoofed by photo or video recording to mimic person. There are some type of attack of spoof:

1. Replay attack
    
    Replay attack in face recognition spoofing occurs when an attacker uses a recorded video or series of images of the victim to deceive the face recognition system.
    
2. Print attack
    
    Print attack in face recognition spoofing occurs when an attacker uses printed photos of the victim to deceive the face recognition system.

Therefore, we need to add face recognition with anti-spoof detection using deep learning model.

## Requirements

The system should handle variations in lighting conditions, facial expressions, and angles, and be robust against different spoofing attacks.

## Dataset

This model was trained using the CelebA Spoof Dataset ([GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | [Kaggle](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing)).

## Data Preprocessing

To achieve the requirements, we need to use image augmentation techniques so that the model can be robust to detect spoof attack.

**Lighting Conditions**

Lighting condition can be obtained by adding the following techniques:

- ISO Noise - Camera has a capability to adjust ISO number based on the environment. ISO number will be higher in the darker condition. However, the higher ISO number, the more noise will come
- Motion Blur - With darker condition, camera will slow the shutter speed so that it can absorb as many light as possible. However, slower shutter speed can give some blurriness to the image

**Angles**

Angles variation can be obtained by adding the following techniques:

- Horizontal and Vertical flip
- Affine
- Shift Scale Rotate

**Others**

Other image augmentation techniques can help model to train variation condition of the image:

- Image Compression - Correlate with the quality of image
- Gaussian Noise - Correlate with noise image
- Normalize - To make the image can be learned easily by the model
- Coarse Dropout - Bad image still can be used to spoof attack, so we need model to learn about it.
- Resizing


MobileNet v2 is used to show the training process in this notebook. MobileNet v2 is light model and can be used for general purpose. I want to use FaceNet since it is specialized for face recognition, but I couldn't find the way to import the pre-trained model.

However, because of the lack of resources, the training process is not completely done. In the real case, the adequate resources should be available for training.

Open the notebook [here](https://github.com/BIMA/anti-spoofing-face-recognition/blob/main/notebook/satnusapersada-technical-test-mobilenet.ipynb).

## Scope of Work

Because of the limit time to complete this project, I narrow down the scope into two steps:

1. Show how I did the data preprocessing by using several image augmentation techniques in Jupyter notebook
2. Show how the pre-trained model is built into API so that I can predict sample video to get the result through the API

Also, this model only tested for recognizing real person and fake person from a photo through webcam.

### Blocker
When I tried to download the smallest zip file of CelebA dataset from this [Google Drive](https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z), I found the file was corrupted. Hence, need to download whole zip to solve the issue.

## Tools

I am using several libraries to finish this project:

1. Python 3.10
2. ONNXRuntime
3. FastAPI

## Demo - Run stream prediction

Run stream prediction will use webcam to stream a video and do the real-time prediction.

https://github.com/BIMA/anti-spoofing-face-recognition/assets/25656575/ae8ef299-1e03-4e75-8249-f4d5df1dcf62


## Demo - Run offline prediction


Run offline prediction will take a video from directory and store the result into directory. *The video is trimmed because prediction takes several minutes to complete.*

https://github.com/BIMA/anti-spoofing-face-recognition/assets/25656575/1ddbd72c-d9db-4c6c-accf-c2ffd440b57d

## Demo - Run prediction through API

Run prediction through API is uploading the video to API (using Postman) and store the result into directory. *The video is trimmed because prediction takes several minutes to complete.*

https://github.com/BIMA/anti-spoofing-face-recognition/assets/25656575/9b9d9a99-10e9-4d9e-a2b7-b669f7a9c8ca

## How to run the prediction script locally?

Install the requirements first by this command:

```bash
pip install -r requirements.txt
```

Run this command to use webcam and do not store any result video:

```bash
python main.py
```

To predict specific video and store the result to another folder, run this command

```bash
python main.py --input /path/to/input_video --output /path/to/output_video
```

For more details about the argument, see below


`--input (-i)` - Path to video for predictions, if not passed webcam stream will be processed

`--output (-o)` - Path to save processed video, if not passed the processed video will not be saved

`--model_path (-m)` - Path to pretrained ONNX model

`--threshold (-t)` - Real face probability threshold above which the prediction is considered true, default 0.5


## How to run and test API locally?

Steps:

1. Install all requirements using this command: `pip install -r requirements.txt`
2. Run this command: `uvicorn main:app --port=4444 --reload`
3. Open Postman
4. Create new request
5. Change method to POST
6. Put the endpoint link: `localhost:4444/video/detect-faces`
7. Go to `Body` tab
8. Select `form-data`
9. Put value in `Key` column with `video_file` and click the dropdown and select File
10. Upload the recorded video
11. Click Send button


## Reference

There are two resources to complete this technical test:

1. Fine tune on CelebA notebook. Visit this [link](https://www.kaggle.com/code/shijunjie07/fine-tune-on-celeba-dataset-for-pc-pad-phone).
2. Face Recognition Anti Spoofing project. Visit this [link](https://github.com/hairymax/Face-AntiSpoofing).

I used some code from those resources with some modification
