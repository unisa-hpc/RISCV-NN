/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

//
// Created by saleh on 11/5/24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "common01.h"

#include <cuda_runtime.h>


class timer_scope_cuda : public timer_scope
{
protected:
    std::chrono::system_clock::time_point m_oTimerLast;
    cudaEvent_t m_oStartEvent, m_oStopEvent;
    cudaStream_t &m_oStream;
public:
    timer_scope_cuda(const std::string &name, cudaStream_t &stream) : timer_scope(name), m_oStream(stream)
    {
        cudaEventCreate(&m_oStartEvent);
        cudaEventCreate(&m_oStopEvent);
        MarkStart();
        report = false;
    }

    timer_scope_cuda(timer_stats &parent, cudaStream_t &stream) : timer_scope(parent),
                                                               m_oStream(stream) {
        cudaEventCreate(&m_oStartEvent);
        cudaEventCreate(&m_oStopEvent);
        MarkStart();
        report = false;
    }

    ~timer_scope_cuda()
    {
        MarkStop();
        if (m_bIsRoot)
        {
            // wait for m_oStartEvent and m_oStopEvent events to complete (start and stop)
            cudaEventSynchronize(m_oStopEvent);
            float ms;
            cudaEventElapsedTime(&ms, m_oStartEvent, m_oStopEvent);
            std::cout << "Elapsed (device)" << name << ": " << ms << " ms ." << std::endl;
        }
        else
        {
            cudaEventSynchronize(m_oStopEvent);
            float ms;
            cudaEventElapsedTime(&ms, m_oStartEvent, m_oStopEvent);
            m_pStats->add_sample(ms);
        }
    }

    void MarkStart()
    {
        cudaEventRecord(m_oStartEvent, m_oStream);
    }

    void MarkStop()
    {
        cudaEventRecord(m_oStopEvent, m_oStream);
    }

    template <class StdTimeResolution = std::milli>
    float FromLast() = delete;


    template <class StdTimeResolution = std::milli>
    float ReportFromLast(const std::string &msg = "") = delete;

    template <class StdTimeResolution = std::milli>
    static inline float ForLambda(const std::function<void()> &operation) = delete;

    template <class StdTimeResolution = std::milli>
    static inline float ReportForLambda(const std::function<void()> &operation) = delete;
};
