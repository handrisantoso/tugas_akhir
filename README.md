# 🧤 Gesture Glove ML Pipeline

An end-to-end Machine Learning pipeline for a gesture-controlled glove using **XIAO ESP32-S3** and the **MPU6050** IMU. This project allows you to record movement data, train high-performance models (Random Forest, MLP, CNN1D), and deploy them back to the ESP32 for real-time USB HID keyboard control.

---

## 🚀 Key Features

*   **End-to-End Pipeline**: From raw IMU data collection to embedded C header export.
*   **Triple Model Support**:
    *   **Random Forest (RF)**: Lightweight, pure C inference via `m2cgen`.
    *   **MLP (Multi-Layer Perceptron)**: Statistical feature-based neural network via TFLite Micro.
    *   **CNN1D (1D Convolutional Neural Network)**: Raw time-series deep learning via TFLite Micro.
*   **Dual Inference Architecture**:
    *   **Embedded**: Optimized for ESP32 (int8 quantized or pure C).
    *   **PC-Side**: High-accuracy float32 inference for real-time debugging.
*   **Live Debugging**: Side-by-side comparison of device vs. PC predictions.
*   **Hardware Integration**:
    *   **USB HID**: Control your PC directly via USB cable.
    *   **Wireless Smartboard (BLE)**: Control presentations (PowerPoint, PDF) remotely using Bluetooth Low Energy and Python.

---

## 🛠 Tech Stack

### Hardware
*   **Microcontroller**: [XIAO Seeed ESP32-S3](https://www.seeedstudio.com/Seeed-XIAO-ESP32S3-p-5627.html) (with USB HID support).
*   **Sensor**: MPU6050 (6-axis IMU: Accel + Gyro).
*   **Connection**: USB-C for Serial & HID.

### Software (Python)
*   **UI**: `CustomTkinter` (Modern Dark UI).
*   **ML**: `Scikit-Learn`, `TensorFlow 2.x`, `m2cgen`.
*   **Visualization**: `Matplotlib`, `Seaborn`.
*   **Connectivity & Automation**: `PySerial`, `Bleak` (BLE), `PyAutoGUI` (Smartboard).

### Firmware (C++/Arduino)
*   **Engine**: `TensorFlow Lite for Microcontrollers` (ESP32 optimized).
*   **IMU**: `Adafruit MPU6050` library.
*   **USB**: Native ESP32-S3 USB HID.

---

## 📦 Project Structure

```text
gesture_glove/
├── dataset/             # CSV recordings
├── firmware/            # ESP32 Arduino sketches
│   ├── gesture_glove_esp32/  # Unified main firmware
│   ├── gesture_glove_rf/     # Optimized RF-only
│   └── gesture_glove_mlp/    # TFLite-only (MLP/CNN)
├── models/              # Saved .keras and .joblib files
├── output_headers/      # Generated C headers for ESP32
├── ble_manager.py       # BLE connection handler for ESP32
├── config.py            # Central settings (Sample rate, Window size, etc.)
├── gui.py               # Main application entry
├── gui_recorder.py      # Dataset collection tab
├── gui_train.py         # Model training & export tab
├── gui_debug.py         # Live visualization & dual-inference tab
├── inference.py         # PC-side inference engine
├── models.py            # ML training & header generation logic
├── recorder.py          # Serial communication & recording manager
├── smartboard.py        # Wireless presentation control via BLE (pyautogui)
│
│   # --- Analysis & Utility Scripts ---
├── thesis_scripts/              # Folder for thesis evaluation and reporting scripts
├── augment_dataset.py           # Augments gesture dataset for robustness
├── regenerate_deploy.py         # Regenerates C headers without full retraining
├── test_actual_data.py          # Quick test script using real-world data
├── thesis_all_metrics.py        # Computes comprehensive thesis metrics
└── _inspect_data.py             # Script to inspect and debug dataset contents
```

---

## 🚦 Getting Started

### 1. Hardware Setup
Connect the MPU6050 to the XIAO ESP32-S3:
*   **VCC** → 3.3V
*   **GND** → GND
*   **SCL** → Pin D5 (GPIO 5)
*   **SDA** → Pin D4 (GPIO 4)
*   **AD0** → High (Pull to 3.3V) for I2C Address `0x69`.

### 2. Software Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/gesture_glove.git
cd gesture_glove

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage Flow

#### Step A: Record Data
1.  Open the GUI: `python gui.py`.
2.  Go to the **📡 Recorder** tab.
3.  Connect your ESP32.
4.  Select a gesture (e.g., "flick_up") and hit **Start Record**.
5.  Perform the gesture during the countdown. Save to CSV when finished.

#### Step B: Train Model
1.  Switch to the **🧠 Train & Export** tab.
2.  Choose your model type (**rf**, **mlp**, or **cnn1d**).
3.  Click **🚀 Train Model**.
4.  Once accurate, the pipeline automatically generates a C header in `output_headers/` and copies it to the `firmware/` folders.

#### Step C: Flash Firmware
1.  Open `firmware/gesture_glove_esp32/gesture_glove_esp32.ino` in Arduino IDE.
2.  Ensure you have `Adafruit MPU6050` and (if using MLP/CNN) `TensorFlowLite_ESP32` libraries installed.
3.  Uncomment the header for your model (e.g., `#include "rf_model_data.h"`).
4.  Upload to your XIAO ESP32-S3.

#### Step D: Live Debug
1.  Go to the **🔍 Live Debug** tab in the Python app.
2.  Click **Start Live**.
3.  Compare the real-time predictions from the ESP32 vs. the PC's high-accuracy engine.

#### Step E: Wireless Smartboard (BLE)
1.  Run `python smartboard.py` to open the standalone Smartboard app.
2.  Ensure your ESP32 is running a firmware with BLE enabled.
3.  Click **Scan** and **Connect** to your `GestureGlove` device.
4.  Perform gestures to control your PC:
    *   **flick_up**: F5 (Start Presentation)
    *   **wave_left**: Left Arrow (Previous Slide)
    *   **wave_right**: Right Arrow (Next Slide)
    *   **flick_down**: Escape (End Presentation)

---

## 🧩 Pipeline Details

### Data Format
The pipeline uses a **Long Format** CSV. Every gesture recording is a 2.5-second window (250 samples @ 100Hz).
*   **MLP/RF**: Extracts 36 statistical features (Mean, Std, Max, Min, RMS, Zero-Crossing) per axis.
*   **CNN1D**: Feeds raw 1500-point time-series data into 1D Convolution layers.

### Normalization
A `StandardScaler` is fitted during training. The mean and scale parameters are exported to the C header, ensuring that the ESP32 performs the **exact same normalization** as the training script.

---

## ⚖️ License
MIT License - See [LICENSE](LICENSE) for details.
