# Anti-Spoofing Face Recognition

## How to run locally?

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

## How to deploy into Cloud?