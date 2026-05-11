# 👤 Human Detection using YOLOv8

This project demonstrates **real-time human (person) detection** using the **YOLOv8 model** from [Ultralytics](https://github.com/ultralytics/ultralytics).  
It works on both **webcam streams** and **video files**, highlighting detected persons with bounding boxes and a total count.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📖 Project Overview
The goal of this project is to build a **real-time person detection system** using the YOLOv8 deep learning model.  
The system can process input from:
- A **webcam** (live detection)  
- A **video file** (offline detection)  

It identifies all persons in each frame, draws bounding boxes, labels them, and counts the total persons detected.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🌐 Live Demo

Click here to try the live app 👉 [Human/Person Detection App](https://person-detection.streamlit.app/)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## ⚙️ Tools & Technologies
- **Python 3.8+** – Core programming language  
- **OpenCV (cv2)** – Video processing and visualization  
- **YOLOv8 (Ultralytics)** – Deep learning object detection model  
- **PyWin32** – Windows-specific utilities (mutex handling)  

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ✨ Features
- 🔍 Detect humans in **real-time** using webcam  
- 🎥 Detect humans from **video files**  
- 📦 Uses **YOLOv8n / YOLOv8s** (lightweight and accurate)  
- 📊 Displays total number of persons detected in each frame  
- ✅ Easy to customize for different YOLOv8 variants  

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📂 Project Structure

├── README.md 

├── app.py

├── input_video.mp4 

├── output_video.mp4 

├── packages.txt

├── person_video.py

├── person_webcam.py

├── requirements.txt 

├── runtime.txt

└── yolov8n.pt

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Usage

After installing the dependencies, you can run the project in two modes:

▶️ Run Detection on Webcam

Start real-time person detection using your laptop/PC webcam:

```bash
python person_webcam.py 
```

▶️ Run Detection on a Video File

Run detection on an existing video file:

```bash
python person_video.py
```

⏹️ Exit the Program

Press q on your keyboard anytime to stop detection and close the window.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 Results

- Persons are detected with bounding boxes and labels (Person 1, Person 2, etc.)

- The total number of persons per frame is displayed

- Works on both live webcam and offline videos
  
- Demo Video

https://github.com/user-attachments/assets/30f4c927-cfbb-4c75-9926-61253849984e


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📚 References

- https://thedatafrog.com/en/articles/human-detection-video/  

- https://docs.ultralytics.com/tasks/detect/

- https://thesai.org/Downloads/Volume16No5/Paper_14-Human_Detection_and_Tracking_with_YOLO.pdf

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 👤 Author

Muqadas Ejaz

BS Computer Science (AI Specialization)

AI/ML Engineer

Kaggle Grand Master

Data Science & Gen AI 

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/muqadasejaz/)  

🌐 GitHub: [github.com/muqadasejaz](https://github.com/muqadasejaz)

📬 Kaggle: [Kaggle Profile](https://www.kaggle.com/muqaddasejaz) 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).

