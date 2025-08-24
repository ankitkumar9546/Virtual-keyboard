# Virtual Keyboard using OpenCV & Mediapipe

This project implements a **Virtual Keyboard** that uses a webcam to detect hand gestures and simulates key presses on your system. 
You can control the keyboard by moving your index finger over the keys and performing a "pinch" gesture with your thumb to press a key.

## 🚀 Features
- Real-time hand detection using **Mediapipe**
- Virtual keyboard overlay using **OpenCV**
- Pinch gesture detection for keypress events
- Typing directly on your system using **PyAutoGUI**
- Visual feedback for hover and pressed keys

## 🛠️ Tech Stack
- Python 3.x
- OpenCV (cv2)
- Mediapipe
- NumPy
- PyAutoGUI

## 📂 Project Structure
virtual_keyboard.py   # Main project file
requirements.txt      # Python dependencies
README.txt            # Project documentation

## ⚙️ Installation
1. Clone the repository:
   git clone https://github.com/your-username/virtual-keyboard.git
   cd virtual-keyboard

2. Install dependencies:
   pip install -r requirements.txt

## ▶️ Usage
Run the script using:
   python virtual_keyboard.py

- Move your index finger over keys → **hover**
- Pinch thumb + index → **press key**
- Press `Q` or `Esc` to exit

## 📷 How It Works
1. **Webcam Input** → Captures real-time video  
2. **Hand Tracking (Mediapipe)** → Detects index & thumb landmarks  
3. **Pinch Detection** → Identifies key presses  
4. **Virtual Keyboard Overlay** → Displays keys on the screen  
5. **PyAutoGUI** → Sends keystrokes to system  

## 📜 License
This project is open-source. Feel free to modify and use it for learning purposes.
