// measurements.h — shared ESP32 inference measurement utilities
// Measures: latency (µs), heap usage, FPS, stability over time
#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <Arduino.h>
#include <esp_timer.h>

struct InferenceStats {
    // Accumulated
    uint32_t total_count;
    uint32_t ok_count;
    uint32_t fail_count;
    int64_t latency_sum;
    int64_t latency_min;
    int64_t latency_max;
    int64_t heap_min;

    // Timers
    int64_t test_start_us;
    int64_t last_stats_us;
};

static inline void stats_init(InferenceStats* s) {
    s->total_count = 0;
    s->ok_count = 0;
    s->fail_count = 0;
    s->latency_sum = 0;
    s->latency_min = INT64_MAX;
    s->latency_max = 0;
    s->heap_min = INT64_MAX;
    s->test_start_us = esp_timer_get_time();
    s->last_stats_us = s->test_start_us;
}

static inline void stats_record(InferenceStats* s, int64_t latency_us, bool ok) {
    s->total_count++;
    int64_t heap_free = ESP.getFreeHeap();

    if (ok) {
        s->ok_count++;
        s->latency_sum += latency_us;
        if (latency_us < s->latency_min) s->latency_min = latency_us;
        if (latency_us > s->latency_max) s->latency_max = latency_us;
    } else {
        s->fail_count++;
    }

    if (heap_free < s->heap_min) s->heap_min = heap_free;
}

static inline float stats_avg_latency(InferenceStats* s) {
    return (s->ok_count > 0) ? (float)s->latency_sum / s->ok_count : 0;
}

static inline float stats_avg_fps(InferenceStats* s) {
    float avg = stats_avg_latency(s);
    return (avg > 0) ? 1000000.0f / avg : 0;
}

static inline void stats_print_periodic(InferenceStats* s, uint32_t interval_us) {
    int64_t now = esp_timer_get_time();
    if (now - s->last_stats_us >= interval_us) {
        float avg_lat = stats_avg_latency(s);
        float fps = stats_avg_fps(s);
        Serial.printf(
            "[STATS] count=%u ok=%u fail=%u min=%lldus max=%lldus avg=%.0fus fps=%.0f heap=%ld\n",
            s->total_count, s->ok_count, s->fail_count,
            s->latency_min, s->latency_max, avg_lat, fps, ESP.getFreeHeap()
        );
        s->last_stats_us = now;
    }
}

static inline void stats_print_stability(InferenceStats* s) {
    int64_t now = esp_timer_get_time();
    int64_t duration_sec = (now - s->test_start_us) / 1000000;
    float success_rate = (s->total_count > 0)
        ? 100.0f * (float)s->ok_count / s->total_count : 0;
    float avg_lat = stats_avg_latency(s);
    float avg_fps = stats_avg_fps(s);
    Serial.printf(
        "[STABILITY] duration=%ds total=%u ok=%u fail=%u rate=%.2f%% avg_lat=%.0fus avg_fps=%.0f heap_min=%lld\n",
        (int)duration_sec, s->total_count, s->ok_count, s->fail_count,
        success_rate, avg_lat, avg_fps, s->heap_min
    );
}

#endif // MEASUREMENTS_H