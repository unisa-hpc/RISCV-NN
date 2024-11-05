//
// Created by saleh on 11/5/24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <functional>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdlib>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class TimerScope; // Forward declaration
class TimerStats {
    friend class TimerScope;
private:
    const std::string m_strName;
    std::vector<float> m_vTimes;
public:
    TimerStats(const std::string& name) : m_strName(name) {

    }

    void AddSample(float time) {
        m_vTimes.push_back(time);
    }

    size_t GetSampleCount() const {
        return m_vTimes.size();
    }

    float GetAverage() const {
        // Calculate the average with vector
        float sum = 0;
        for (auto t : m_vTimes) {
            sum += t;
        }
        return sum / m_vTimes.size();
    }

    float GetMax() const {
        float max = m_vTimes[0];
        for (size_t i = 1; i < m_vTimes.size(); i++) {
            if (m_vTimes[i] > max) {
                max = m_vTimes[i];
            }
        }
        return max;
    }

    float GetMin() const {
        float min = m_vTimes[0];
        for (size_t i = 1; i < m_vTimes.size(); i++) {
            if (m_vTimes[i] < min) {
                min = m_vTimes[i];
            }
        }
        return min;
    }

    float GetMedian() const {
        std::vector<float> v = m_vTimes;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        if (n % 2 == 0) {
            return (v[n / 2 - 1] + v[n / 2]) / 2;
        } else {
            return v[n / 2];
        }
    }

    float GetVariance() const {
        float avg = GetAverage();
        float sum = 0;
        for (auto t : m_vTimes) {
            sum += (t - avg) * (t - avg);
        }
        return sum / m_vTimes.size();
    }

    void PrintStats() const {
        std::cout << "============================================" << std::endl;
        std::cout << "Stats for " << m_strName << ":" << std::endl;
        std::cout << ">>Median:  \t" << GetMedian() << std::endl;
        std::cout << "> Average: \t" << GetAverage() << std::endl;
        std::cout << "> Samples: \t" << GetSampleCount() << std::endl;
        std::cout << "> Variance:\t" << GetVariance() << std::endl;
        std::cout << "> Max:     \t" << GetMax() << std::endl;
        std::cout << "> Min:     \t" << GetMin() << std::endl;
        std::cout << "============================================" << std::endl;
    }

    ~TimerStats() {
        PrintStats();
    }
};

class TimerScope {
private:
    std::chrono::system_clock::time_point m_oTimerLast;
    const std::string m_strName;
    const bool m_bIsRoot;
    TimerStats *m_pStats; // to keep things simple, we are not using smart pointers.

public:
    TimerScope(const std::string& name) : m_strName(name), m_bIsRoot(true) {
        m_oTimerLast = high_resolution_clock::now();
        m_pStats = nullptr;
    }

    TimerScope(TimerStats &parent) : m_strName(""), m_bIsRoot(false) {
        m_oTimerLast = high_resolution_clock::now();
        m_pStats = &parent;
    }

    ~TimerScope() {
        if(m_bIsRoot) {
          ReportFromLast(m_strName);
        } else {
          m_pStats->AddSample(FromLast());
        }
    }

    template <class StdTimeResolution = std::milli>
    float FromLast() {
        auto now = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = now - m_oTimerLast;
        m_oTimerLast = now;
        return ms.count();
    }

    template <class StdTimeResolution = std::milli>
    float ReportFromLast(const std::string& msg = "") {
        auto t = FromLast<StdTimeResolution>();
        std::cout << "Elapsed " << msg << ": " << t << " ." << std::endl;
        return t;
    }

    template <class StdTimeResolution = std::milli>
    static inline float ForLambda(const std::function<void()>& operation) {
        auto t1 = high_resolution_clock::now();
        operation();
        auto t2 = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = t2 - t1;
        return ms.count();
    }

    template <class StdTimeResolution = std::milli>
    static inline float ReportForLambda(const std::function<void()>& operation) {
        auto t = ForLambda<StdTimeResolution>(operation);
        std::cout << "Elapsed: " << t << " ." << std::endl;
        return t;
    }
};

template <typename T>
T* aligned_alloc_array(size_t size, size_t alignment) {
    // Calculate the total size in bytes, ensuring it’s a multiple of alignment
    size_t total_size = size * sizeof(T);
    if (total_size % alignment != 0) {
        total_size += alignment - (total_size % alignment); // Round up to next multiple
    }
    // Allocate aligned memory
    void* ptr = std::aligned_alloc(alignment, total_size);
    if (!ptr) {
        throw std::bad_alloc();
    }

    // Return the pointer as type T*
    return static_cast<T*>(ptr);
}
