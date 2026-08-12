# 🤖 AI-Based Real-Time Human & Animal Detection

A real-time AI camera detection system using **YOLOv8, OpenCV, Python, Flask, Docker, and GitHub Actions**.

The system detects humans and animals from a live camera, provides alerts and alarms, saves detection images, and maintains detection logs.

## 🚀 Features

- 📷 Live camera detection
- 👤 Human detection
- 🐾 Animal detection
- 🎯 Confidence score
- 🚨 Real-time alert
- 🔊 Alarm notification
- 📸 Detection image saving
- 📝 CSV detection logs
- 🗑️ Automatic deletion of images older than 5 days
- 🌐 Flask web dashboard
- 🐳 Docker support
- ⚙️ GitHub Actions CI/CD

## 🛠️ Technologies

- Python
- YOLOv8
- OpenCV
- Flask
- HTML/CSS/JavaScript
- Docker
- Git & GitHub
- GitHub Actions
- Linux

## 🏗️ Architecture

```text
📷 Camera
   ↓
OpenCV
   ↓
YOLOv8
   ↓
👤 Human / 🐾 Animal
   ↓
Flask Application
   ↓
🚨 Alert + 🔊 Alarm
   ↓
📸 Images + 📝 Logs


⚙️ Run Locally
1. Clone
git clone https://github.com/Divya12004/Camera-detection.git
cd Camera-detection
2. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure Environment
cp .env.example .env
5. Run Application
python app.py

Open:

http://localhost:5000

🐳 Run with Docker

Check camera:

ls /dev/video*

Build image:

docker build -t camera-detection .

Run container:

docker run -d \
  --name camera-detection \
  --device=/dev/video0:/dev/video0 \
  -p 5000:5000 \
  camera-detection

Check:

docker ps

View logs:

docker logs camera-detection

Open:

http://localhost:5000

🔄 CI/CD Pipeline

GitHub Actions automatically runs when code is pushed to the main branch.

Developer
    ↓
git push
    ↓
GitHub
    ↓
GitHub Actions
    ↓
Python Tests
    ↓
Syntax Check
    ↓
Docker Build
    ↓
✅ Pipeline Success

Workflow file:

.github/workflows/ci-cd.yml

📸 Detection Images

Images are stored in:

detections/

The application saves one image per detection event.

Images older than 5 days are automatically deleted.

📝 Detection Logs

Logs are stored in:

logs/detections.csv

Example:

timestamp,type,object,confidence,image
2026-08-12,HUMAN,person,98.5,detection.jpg
2026-08-12,ANIMAL,dog,96.2,detection.jpg

