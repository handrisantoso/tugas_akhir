// scaler_params.h — StandardScaler parameters
// Apply per channel before CNN inference:
//   scaled = (raw - SCALER_MEAN[ch]) / SCALER_STD[ch]
#pragma once

const int WINDOW_SIZE  = 25;
const int N_IMU_CHANNELS = 12;

const float SCALER_MEAN[12] = {
    -2432.98876953f,  // ax1
    -17429.23437500f,  // ay1
    -9946.20410156f,  // az1
    -175.46310425f,  // gx1
    36.79816818f,  // gy1
    -52.00880051f,  // gz1
    3629.64111328f,  // ax2
    5866.31250000f,  // ay2
    -11915.72167969f,  // az2
    -129.84693909f,  // gx2
    50.68962097f,  // gy2
    -72.22756195f  // gz2
};

const float SCALER_STD[12] = {
    5394.58447266f,  // ax1
    8113.98144531f,  // ay1
    5292.03466797f,  // az1
    5173.42138672f,  // gx1
    8431.63769531f,  // gy1
    5183.42333984f,  // gz1
    6975.25683594f,  // ax2
    8895.53808594f,  // ay2
    7948.84375000f,  // az2
    9034.15722656f,  // gx2
    6229.82470703f,  // gy2
    8394.33105469f  // gz2
};
