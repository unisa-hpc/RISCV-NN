/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

//
// Created by saleh on 9/17/23.
//

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#include "common02.h"
#include "CRandFiller.h"
#include "cnpy.h"

#define CHECK(E) if(E!=cudaError_t::cudaSuccess) std::cerr<<"CUDA API FAILED, File: "<<__FILE__<<", Line: "<< __LINE__ << ", Error: "<< cudaGetErrorString(E) << std::endl;

enum class FillTypes {
    kRandom,
    kRandomPoT,
    kIncr,
    kIncrColMajor,
    kDecr,
    kCustom1,
    kConstant0,
    kConstant1,
};

template<typename T>
class CTensor {
protected:
    std::vector<size_t> m_vShape;
    size_t m_ulSize, m_ulSizeBytes;
    T *m_ptrDataDevice;
    T *m_ptrDataHost;

    static size_t _GetSize(const std::vector<size_t> &shape) {
        size_t size = 1;
        std::for_each(shape.begin(), shape.end(), [&size](size_t val) { size *= val; });
        return size;
    }

    static size_t _GetSizeBytes(const std::vector<size_t> &shape) {
        return _GetSize(shape) * sizeof(T);
    }

public:
    CTensor(const std::vector<size_t> &shape) {
        m_vShape = shape;
        m_ulSize = _GetSize(shape);
        m_ulSizeBytes = _GetSizeBytes(shape);
        m_ptrDataHost = new T[m_ulSize];
        CHECK(cudaMalloc((void **) &m_ptrDataDevice, m_ulSizeBytes));
    }

    ~CTensor() {
        delete[] m_ptrDataHost;
        CHECK(cudaFree(m_ptrDataDevice));
    }

    // copy ctor
    CTensor(const CTensor<T> &other) {
        m_vShape = other.m_vShape;
        m_ulSize = _GetSize(other.m_vShape);
        m_ulSizeBytes = _GetSizeBytes(other.m_vShape);

        m_ptrDataHost = new T[m_ulSize];
        CHECK(cudaMalloc((void **) &m_ptrDataDevice, m_ulSizeBytes));

        std::copy(other.m_ptrDataHost, other.m_ptrDataHost + other.m_ulSize, m_ptrDataHost);
        CHECK(cudaMemcpy(m_ptrDataDevice, other.m_ptrDataDevice, m_ulSizeBytes, cudaMemcpyDeviceToDevice));
    }

    // copy assignment
    CTensor &operator=(const CTensor &other) {
        if (this == &other) return *this;

        // This is not a ctor, so we have to release previously allocated resources.
        delete[] m_ptrDataHost;
        CHECK(cudaFree(m_ptrDataDevice));

        // And here we have to copy the content from `other` to `this`.
        m_vShape = other.m_vShape;
        m_ulSize = _GetSize(other.m_vShape);
        m_ulSizeBytes = _GetSizeBytes(other.m_vShape);

        m_ptrDataHost = new T[m_ulSize];
        CHECK(cudaMalloc((void **) &m_ptrDataDevice, m_ulSizeBytes));

        std::copy(other.m_ptrDataHost, other.m_ptrDataHost + other.m_ulSize, m_ptrDataHost);
        CHECK(cudaMemcpy(m_ptrDataDevice, other.m_ptrDataDevice, m_ulSizeBytes, cudaMemcpyDeviceToDevice));

        return *this;
    }

    // move ctor
    CTensor(CTensor<T> &&other) noexcept {
        m_vShape = std::move(other.m_vShape);
        m_ulSize = _GetSize(m_vShape);
        m_ulSizeBytes = _GetSizeBytes(m_vShape);
        other.m_ulSize = other.m_ulSizeBytes = 0;

        m_ptrDataHost = other.m_ptrDataHost;
        other.m_ptrDataHost = nullptr;

        m_ptrDataDevice = other.m_ptrDataDevice;
        other.m_ptrDataDevice = nullptr;
    }

    // move assignment
    CTensor &operator=(CTensor<T> &&other) noexcept {
        if (this == &other) return *this;

        // This is not a ctor, so we have to release previously allocated resources.
        delete[] m_ptrDataHost;
        CHECK(cudaFree(m_ptrDataDevice));

        // And here we have to copy the content from `other` to `this`.
        m_vShape = std::move(other.m_vShape);
        m_ulSize = _GetSize(m_vShape);
        m_ulSizeBytes = _GetSizeBytes(m_vShape);
        other.m_ulSize = other.m_ulSizeBytes = 0;

        m_ptrDataHost = other.m_ptrDataHost;
        other.m_ptrDataHost = nullptr;

        m_ptrDataDevice = other.m_ptrDataDevice;
        other.m_ptrDataDevice = nullptr;

        return *this;
    }

    T &operator[](size_t rowMajorIndex) {
        return m_ptrDataHost[rowMajorIndex];
    }

    T operator[](size_t rowMajorIndex) const {
        return m_ptrDataHost[rowMajorIndex];
    }

    void H2D() {
        // dest ptr, src ptr, size bytes, enum
        CHECK(cudaMemcpy(m_ptrDataDevice, m_ptrDataHost, m_ulSizeBytes, cudaMemcpyHostToDevice));
    }

    void D2H() {
        // dest ptr, src ptr, size bytes, enum
        CHECK(cudaMemcpy(m_ptrDataHost, m_ptrDataDevice, m_ulSizeBytes, cudaMemcpyDeviceToHost));
    }

    void Fill(CRandFiller<T> *randFiller, const FillTypes &type) {
        switch (type) {
            case FillTypes::kRandom: {
                assert(randFiller != nullptr);
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = randFiller->GetRand(); }
                break;
            }
            case FillTypes::kRandomPoT: {
                assert(randFiller != nullptr);
                for (size_t i = 0; i < m_ulSize; i++) {
                    auto r = randFiller->GetRand();
                    // Get the closest power of two to r
                    m_ptrDataHost[i] = (T) (1 << (int) (log2(r) + 0.5));
                }
                break;
            }
            case FillTypes::kIncr: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) i; }
                break;
            }
            case FillTypes::kIncrColMajor: {
                if (this->m_vShape.size() != 2) {
                    std::cerr << "kIncrColMajor is only supported for 2D tensors." << std::endl;
                    return;
                }

                for (size_t j = 0; j < this->m_vShape[0]; j++) {
                    for (size_t i = 0; i < this->m_vShape[1]; i++) {
                        // set col major index
                        const size_t val = i*this->m_vShape[0]+j;
                        m_ptrDataHost[j*this->m_vShape[1]+i] = (T) (val) + 0.5f;
                    }
                }
                break;
            }
            case FillTypes::kDecr: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) (m_ulSize - i - 1); }
                break;
            }
            case FillTypes::kCustom1: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) (i % 10); }
                break;
            }
            case FillTypes::kConstant0: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = 0; }
                break;
            }
            case FillTypes::kConstant1: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) 1; }
                break;
            }
            default: {
                std::cerr << "Unknown FillTypes." << std::endl;
                return;
            }
        }
        H2D();
    }

    std::vector<size_t> GetShape() const {
        return m_vShape;
    }

    size_t GetSize() const {
        return m_ulSize;
    }

    size_t GetSizeBytes() const {
        return m_ulSizeBytes;
    }

    size_t GetRank() const {
        return m_vShape.size();
    }

    T *GetPtrHost() { return m_ptrDataHost; }

    T *GetPtrDevice() { return m_ptrDataDevice; }

    const T *GetPtrHost() const { return m_ptrDataHost; }

    const T *GetPtrDevice() const { return m_ptrDataDevice; }

    bool CompareHostData(const CTensor<T> &other, T maxAllowedError) {
        if (m_vShape != other.m_vShape) return false;
        for (size_t idx=0; idx<m_ulSize; idx++) {
            auto diff = m_ptrDataHost[idx] - other.m_ptrDataHost[idx];
            if (diff < 0) {diff = -1 * diff;}
            if (diff > maxAllowedError) {
                std::cout << "Error: idx=" << idx << " m_ptrDataHost[idx]=" << m_ptrDataHost[idx] << " other.m_ptrDataHost[idx]=" << other.m_ptrDataHost[idx] << std::endl;
                return false; //early termination
            }
        }
        return true;
    }

    void CopyHostDataFrom(const T* buff) {
        std::copy(buff, buff + m_ulSize, m_ptrDataHost);
    }

    static CTensor<T> LoadFromNumpy(const std::string &fname) {
        auto f = cnpy::npy_load(fname);
        CTensor<T> tn(f.shape);
        tn.CopyHostDataFrom(f.data<T>());
        return std::move(tn);
    }

    void Wipe() {
        for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = 0; }
        H2D();
    }
};