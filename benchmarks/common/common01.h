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
#include <fstream>
#include <map>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class timer_scope; // Forward declaration
class timer_stats {
    friend class timer_scope;
private:
    const std::string name;
    std::vector<float> samples;
    const std::map <std::string, int> pairs;

    std::string legalize_filename(const std::string &name) const {
        std::string result = name;
        std::replace(result.begin(), result.end(), ' ', '_');
        std::replace(result.begin(), result.end(), '/', '_');
        std::replace(result.begin(), result.end(), '\\', '_');
        return result;
    }

    std::string pairs_to_string() const {
        std::string result = ".";
        for (auto &pair : pairs) {
            result += legalize_filename(pair.first) + "_" + std::to_string(pair.second) + ".";
        }
        if (pairs.size() > 0) {
            result.pop_back();
        }
        return result;
    }

    std::string pairs_to_json() const {
        std::string result = "{";
        for (auto &pair : pairs) {
            result += "\"" + pair.first + "\": " + std::to_string(pair.second) + ", ";
        }
        // remove the last comma and space if its not empty
        if (pairs.size() > 0) {
            result.pop_back();
            result.pop_back();
        }
        result += "}";
        return result;
    }
public:
    timer_stats(const std::string& name) : name(name) {

    }

    timer_stats(const std::string& name, const std::map<std::string, int>& pairs) : name(name), pairs(pairs) {

    }

    void add_sample(float time) {
        samples.push_back(time);
    }

    size_t count() const {
        return samples.size();
    }

    float ave() const {
        // Calculate the average with vector
        float sum = 0;
        for (auto t : samples) {
            sum += t;
        }
        return sum / samples.size();
    }

    float max() const {
        float max = samples[0];
        for (size_t i = 1; i < samples.size(); i++) {
            if (samples[i] > max) {
                max = samples[i];
            }
        }
        return max;
    }

    float min() const {
        float min = samples[0];
        for (size_t i = 1; i < samples.size(); i++) {
            if (samples[i] < min) {
                min = samples[i];
            }
        }
        return min;
    }

    float median() const {
        std::vector<float> v = samples;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        if (n % 2 == 0) {
            return (v[n / 2 - 1] + v[n / 2]) / 2;
        } else {
            return v[n / 2];
        }
    }

    float variance() const {
        float avg = ave();
        float sum = 0;
        for (auto t : samples) {
            sum += (t - avg) * (t - avg);
        }
        return sum / samples.size();
    }

    void print() const {
        std::cout << "============================================" << std::endl;
        std::cout << "Stats for " << name << " with " << pairs_to_json() << " :" << std::endl;
        std::cout << ">>Median:  \t" << median() << std::endl;
        std::cout << "> Average: \t" << ave() << std::endl;
        std::cout << "> Samples: \t" << count() << std::endl;
        std::cout << "> Variance:\t" << variance() << std::endl;
        std::cout << "> Max:     \t" << max() << std::endl;
        std::cout << "> Min:     \t" << min() << std::endl;
        std::cout << "============================================" << std::endl;
    }

    void save() const {
        std::ofstream file;
        file.open("stats_" + legalize_filename(name) + pairs_to_string() + ".json", std::ios::app);
        // save it in json format
        file << "{\n";
        file << "\"name\": \"" << name << "\",\n";
        file << "\"pairs\": " << pairs_to_json() << ",\n";
        file << "\"samples\": " << count() << ",\n";
        file << "\"average\": " << ave() << ",\n";
        file << "\"median\": " << median() << ",\n";
        file << "\"variance\": " << variance() << ",\n";
        file << "\"max\": " << max() << ",\n";
        file << "\"min\": " << min() << "\n";
        file << "}\n";
        file.close();
    }

    ~timer_stats() {
        print();
        save();
    }
};

class timer_scope {
private:
    std::chrono::system_clock::time_point m_oTimerLast;
    const std::string name;
    const bool m_bIsRoot;
    timer_stats *m_pStats; // to keep things simple, we are not using smart pointers.

public:
    timer_scope(const std::string& name) : name(name), m_bIsRoot(true) {
        m_oTimerLast = high_resolution_clock::now();
        m_pStats = nullptr;
    }

    timer_scope(timer_stats &parent) : name(""), m_bIsRoot(false) {
        m_oTimerLast = high_resolution_clock::now();
        m_pStats = &parent;
    }

    ~timer_scope() {
        if(m_bIsRoot) {
          report_from_last(name);
        } else {
          m_pStats->add_sample(from_last());
        }
    }

    template <class StdTimeResolution = std::milli>
    float from_last() {
        auto now = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = now - m_oTimerLast;
        m_oTimerLast = now;
        return ms.count();
    }

    template <class StdTimeResolution = std::milli>
    float report_from_last(const std::string& msg = "") {
        auto t = from_last<StdTimeResolution>();
        std::cout << "Elapsed " << msg << ": " << t << " ." << std::endl;
        return t;
    }

    template <class StdTimeResolution = std::milli>
    static inline float for_lambda(const std::function<void()>& operation) {
        auto t1 = high_resolution_clock::now();
        operation();
        auto t2 = high_resolution_clock::now();
        duration<float, StdTimeResolution> ms = t2 - t1;
        return ms.count();
    }

    template <class StdTimeResolution = std::milli>
    static inline float report_for_lambda(const std::function<void()>& operation) {
        auto t = for_lambda<StdTimeResolution>(operation);
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
