/* rf_score_wrapper.c — bungkus model RF m2cgen agar bisa dipanggil via ctypes.
 * Meng-export score() dan rf_predict() yang IDENTIK dengan yang dijalankan ESP32,
 * sehingga latency PC vs ESP32 menjadi perbandingan C-vs-C yang setara.
 *
 * Build (dari developer prompt / setelah vcvars64.bat):
 *   cl /O2 /LD rf_score_wrapper.c /Fe:rf_score.dll
 */
#include <stdint.h>
#include "rf_model_data.h"   /* berisi definisi score() & rf_predict() */

/* score mentah: input 36 fitur (sudah ter-scale) -> output 5 vote */
__declspec(dllexport) void rf_score(float* input, float* output) {
    score(input, output);
}

/* klasifikasi lengkap: kembalikan argmax + tulis confidence */
__declspec(dllexport) int rf_classify(float* input, float* confidence) {
    return rf_predict(input, confidence);
}
