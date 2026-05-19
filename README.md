# Smart Attendance System with Face Recognition and WhatsApp Alerts

A Streamlit-based Smart Attendance System that uses face recognition to mark student attendance automatically. It stores attendance records in SQL Server and sends WhatsApp notifications using Twilio.

## Features

- Real-time face detection using webcam
- Face recognition using FaceNet embeddings
- Person tracking using DeepSORT
- Automatic attendance marking
- SQL Server database integration
- WhatsApp notification after attendance is marked
- Attendance dashboard with daily, person-wise, and monthly reports

## Tech Stack

- Python
- Streamlit
- OpenCV
- FaceNet
- DeepSORT
- SQL Server
- Twilio WhatsApp API
- Pandas
- PyODBC

## Project Structure

```bash
Smart-Attendance-System/
│
├── app.py
├── known_faces/
│   ├── Sayan.jpg
│   ├── Suparna.jpg
│   └── Tanujit.jpg
│
├── requirements.txt
└── README.md
