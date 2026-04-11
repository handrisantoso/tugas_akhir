#pragma once
#include <math.h>

/*
 * Feature extractor for gesture recognition.
 *
 * Must stay in sync with compute_features() in train.py:
 *   - N_SAMPLES = 25 rows per window
 *   - 12 channels: ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2
 *   - 5 statistics per channel: mean, std, min, max, rms
 *   - Total: 5 × 12 = 60 features
 */

#define FE_N_SAMPLES  25
#define FE_N_CHANNELS 12
#define FE_N_FEATURES 60   // 5 stats × 12 channels

/*
 * raw_data[sample][channel] — fill this before calling extract_features().
 * Channel order: ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2
 */
static float raw_data[FE_N_SAMPLES][FE_N_CHANNELS];
static int   sample_index = 0;

inline bool window_full() {
    return sample_index >= FE_N_SAMPLES;
}

inline void reset_window() {
    sample_index = 0;
}

inline void push_sample(float ax1, float ay1, float az1,
                        float gx1, float gy1, float gz1,
                        float ax2, float ay2, float az2,
                        float gx2, float gy2, float gz2) {
    if (sample_index >= FE_N_SAMPLES) return;
    raw_data[sample_index][0]  = ax1;
    raw_data[sample_index][1]  = ay1;
    raw_data[sample_index][2]  = az1;
    raw_data[sample_index][3]  = gx1;
    raw_data[sample_index][4]  = gy1;
    raw_data[sample_index][5]  = gz1;
    raw_data[sample_index][6]  = ax2;
    raw_data[sample_index][7]  = ay2;
    raw_data[sample_index][8]  = az2;
    raw_data[sample_index][9]  = gx2;
    raw_data[sample_index][10] = gy2;
    raw_data[sample_index][11] = gz2;
    sample_index++;
}

/*
 * Fills `features[FE_N_FEATURES]` from the current window.
 * Layout: [mean×12, std×12, min×12, max×12, rms×12]
 */
inline void extract_features(float* features) {
    for (int ch = 0; ch < FE_N_CHANNELS; ch++) {
        float sum = 0, sum_sq = 0;
        float mn = raw_data[0][ch];
        float mx = raw_data[0][ch];

        for (int s = 0; s < FE_N_SAMPLES; s++) {
            float v = raw_data[s][ch];
            sum    += v;
            sum_sq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        float mean = sum / FE_N_SAMPLES;
        float var  = (sum_sq / FE_N_SAMPLES) - (mean * mean);
        float std  = sqrtf(var > 0 ? var : 0);
        float rms  = sqrtf(sum_sq / FE_N_SAMPLES);

        features[ch]                    = mean;   // offset 0
        features[FE_N_CHANNELS + ch]    = std;    // offset 12
        features[2 * FE_N_CHANNELS + ch] = mn;   // offset 24
        features[3 * FE_N_CHANNELS + ch] = mx;   // offset 36
        features[4 * FE_N_CHANNELS + ch] = rms;  // offset 48
    }
}
