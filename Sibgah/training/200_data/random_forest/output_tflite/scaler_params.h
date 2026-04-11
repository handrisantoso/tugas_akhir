// scaler_params.h — StandardScaler for Random Forest features
// Apply: scaled = (raw_feature - SCALER_MEAN[i]) / SCALER_STD[i]
// 12 features: mean_ax1, std_ax1, min_ax1, max_ax1 ...
#pragma once

const int RF_N_FEATURES = 12;

const float SCALER_MEAN[12] = {
    -2432.98877292f,  // mean_ax1
    -17429.23452750f,  // std_ax1
    -9946.20445698f,  // min_ax1
    -175.46310296f,  // max_ax1
    36.79816643f,  // median_ax1
    -52.00880113f,  // rms_ax1
    3629.64104372f,  // mean_ay1
    5866.31232722f,  // std_ay1
    -11915.72141044f,  // min_ay1
    -129.84693935f,  // max_ay1
    50.68961918f,  // median_ay1
    -72.22755994f  // rms_ay1
};

const float SCALER_STD[12] = {
    5394.58432888f,  // mean_ax1
    8113.98151612f,  // std_ax1
    5292.03460721f,  // min_ax1
    5173.42115136f,  // max_ax1
    8431.63754548f,  // median_ax1
    5183.42344061f,  // rms_ax1
    6975.25678555f,  // mean_ay1
    8895.53794186f,  // std_ay1
    7948.84351269f,  // min_ay1
    9034.15717535f,  // max_ay1
    6229.82448092f,  // median_ay1
    8394.33119424f  // rms_ay1
};
