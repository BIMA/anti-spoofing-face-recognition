# Anti-Spoofing Face Recognition

## Definition

Face recognition can be done by extracting face features. However, face recognition can be spoofed by photo or video recording to mimic person. There are some type of attack of spoof:

1. Replay attack
    
    Replay attack in face recognition spoofing occurs when an attacker uses a recorded video or series of images of the victim to deceive the face recognition system.
    
2. Print attack
    
    Print attack in face recognition spoofing occurs when an attacker uses printed photos of the victim to deceive the face recognition system.

## Dataset

This model was trained using the CelebA Spoof Dataset ([GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | [Kaggle](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing)).

## Notebook

Data preprocessing is done in the notebook. Data processing includes the following image augmentation:

- Resizing
- Motion Blur
- Horizontal Flip
- Vertical Flip
- Image Compression
- Affine
- Gaussian Noise
- ISO Noise
- Normalize
- Shift Scale Rotate
- Coarse Dropout

MobileNet v2 is used to show the training process in this notebook. However, because of the lack of resources, the training process is not completely done. In the real case, the adequate resources should be available for training.

Open the notebook [here]().

## Scope of Work

Because of the limit time to complete this project, I narrow down the scope into two steps:

1. Show how I did the data preprocessing by using several image augmentation techniques in Jupyter notebook
2. Show how the pre-trained model is built into API so that I, as the client, can predict sample video to get the result using API

Also, this model only tested for recognizing real person and fake person from a photo through webcam.

### Blocker
When I tried to download the smallest zip file from this [Google Drive](https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z), I found the file was corrupted. Hence, need to download whole zip to solve the issue.

## Demo - Run stream prediction

Run stream prediction will use webcam to stream a video and do the real-time prediction.

## Demo - Run offline prediction

Run offline prediction will take a video from directory and store the result into directory.

## Demo - Run prediction through API

Run prediction through API is uploading the video to API (using Postman) and store the result into directory.

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