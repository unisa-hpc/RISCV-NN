//
// Created by saleh on 11/5/24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>
#include <numeric>
#include <exception>
#include <random>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#ifdef AUTOVEC
#define SUFFIX _autovec
#else
#define SUFFIX _noautovec
#endif

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define FUNCTION_NAME(name) CONCATENATE(name, SUFFIX)

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

    std::string data_to_json() const {
        std::string result = "[";
        for (auto &s : samples) {
            result += std::to_string(s) + ", ";
        }
        // remove the last comma and space if its not empty
        if (pairs.size() > 0) {
            result.pop_back();
            result.pop_back();
        }
        result += "]";
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
        file << "\"data\": " << data_to_json() << ", \n";
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
protected:
    std::chrono::system_clock::time_point m_oTimerLast;
    const std::string name;
    const bool m_bIsRoot;
    timer_stats *m_pStats; // to keep things simple, we are not using smart pointers.
    bool report = true;

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
        if (report) {
            if(m_bIsRoot) {
              report_from_last(name);
            } else {
              m_pStats->add_sample(from_last());
            }
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

template <typename T>
class aligned_tensor {
    // Type punning is a undefined behavior in C++ when its done using unions.
    // The only exception is to use char as the dest type.
    // C++20 introduced std::bit_cast to do type punning in a safe way when possible.
    // For older versions of C++, we can use `std::memcpy` to do type punning.
    // Avoiding type punning is the best practice.

    size_t _size, _size_bytes;
    const int _alignment;
    std::vector<size_t> _shape;
    T* _data = nullptr;

protected:
    void infer_size_and_alloc() {
        if (_shape.empty())
            throw std::runtime_error("Shape not set");
        _size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
        _size_bytes = _size * sizeof(T);
        if (_size_bytes % _alignment != 0) {
            _size_bytes += _alignment - (_size_bytes % _alignment); // Round up to next multiple
        }
        void* ptr = std::aligned_alloc(_alignment, _size_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }

        if (_data) {
            std::free(_data);
        }
        _data = static_cast<T*>(ptr);
    }

public:
    enum class init_type {
        random,
        twos
    };

    ~aligned_tensor() {
        if (_data) {
            std::free(_data);
        }
    }

    aligned_tensor(size_t words, int alignment) : _alignment(alignment) {
        _shape.push_back(words);
        infer_size_and_alloc();
    }

    aligned_tensor(const std::vector<size_t>& shape, int alignment) : _alignment(alignment) {
        _shape = shape;
        infer_size_and_alloc();
    }

    /**
     * @return Number of elements of type T in the tensor
     */
    size_t sizet() {
        return _size;
    }

    /**
     * @return Number of elements of type uint16_t in the tensor
     */
    size_t sizeu16() {
        return _size * 2;
    }

    /**
     * @return Number of elements of type uint8_t in the tensor
     */
    size_t sizeu8() {
        return _size * 4;
    }

    /**
     * @return The size of the aligned buffer in bytes with padding.
     */
    size_t size_bytes() {
        return _size_bytes;
    }

    std::vector<size_t> shape() {
        return _shape;
    }

    /**
     * @return Pointer to the aligned buffer of type T.
     */
    T* data_t() {
        return _data;
    }

    /**
     * @return Pointer to the aligned buffer of type uint16_t.
     */
    uint16_t* data_u16() {
        throw std::runtime_error("Not implemented");
    }

    /**
     * @return Pointer to the aligned buffer of type uint8_t.
     */
    uint8_t* data_u8() {
        throw std::runtime_error("Not implemented");
    }

    void wipe() {
        for (size_t i = 0; i < _size; i++) {
            _data[i] = 0;
        }
    }

    void initialize(T val) {
        for (size_t i = 0; i < _size; i++) {
            _data[i] = val;
        }
    }

    void initialize(T* vals, size_t len) {
        if (len != _size) {
            throw std::runtime_error("Invalid length");
        }
        for (size_t i = 0; i < _size; i++) {
            _data[i] = vals[i];
        }
    }

    void initialize(std::vector<T>& vals) {
        if (vals.size() != _size) {
            throw std::runtime_error("Invalid length");
        }
        for (size_t i = 0; i < _size; i++) {
            _data[i] = vals[i];
        }
    }

    void initialize(init_type type, T lower_bound, T upper_bound) {
        if (type == init_type::random) {
            // Seed the random number generator with the current time
            std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
            if constexpr (std::is_floating_point<T>::value) {
                std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
                for (size_t i = 0; i < _size; ++i) {
                    _data[i] = dist(rng);
                }
            } else if constexpr (std::is_integral<T>::value) {
                std::uniform_int_distribution<T> dist(lower_bound, upper_bound);
                for (size_t i = 0; i < _size; ++i) {
                    _data[i] = dist(rng);
                }
            } else {
                throw std::runtime_error("Unsupported type");
            }
            return;
        }
        if (type == init_type::twos) {
          std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
            std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
            for (size_t i = 0; i < _size; ++i) {
                _data[i] = std::pow(2, std::floor(dist(rng)));
            }
            return;
        }
        throw std::runtime_error("Unsupported init type.");
    }

protected:
    void compute_indices(size_t flattened_idx, std::vector<size_t>& indices, const std::vector<size_t>& shape) {
        for (int axis = shape.size() - 1; axis >= 0; --axis) {
            indices[axis] = flattened_idx % shape[axis];
            flattened_idx /= shape[axis];
        }
    }

public:
    void compare(aligned_tensor<T>& other) {
        size_t total_size = 1;
        for (const auto& dim : _shape) {
            total_size *= dim;
        }

        auto *p2 = other.data_t();

        std::vector<size_t> indices(_shape.size());
        for (size_t flattened_idx = 0; flattened_idx < total_size; ++flattened_idx) {
            compute_indices(flattened_idx, indices, _shape);

            if (_data[flattened_idx] != p2[flattened_idx]) {
                std::cout << "Results mismatch at index ";
                for (size_t i = 0; i < indices.size(); ++i) {
                    std::cout << indices[i] << (i + 1 == indices.size() ? "" : ", ");
                }
                std::cout << std::endl;
                std::cout << "_data1 = " << _data[flattened_idx] << ", _data2 = " << p2[flattened_idx] << std::endl;
                std::cout << "Difference = " << _data[flattened_idx] - p2[flattened_idx] << std::endl;
                return;
            }
        }
        std::cout << "Results match!" << std::endl;
    }
};