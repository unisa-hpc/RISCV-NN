/*
* Copyright (c) 2025 University of Salerno
* SPDX-License-Identifier: Apache-2.0
*/

//
// Created by saleh on 10/1/23.
//

#pragma once

#include <random>

template <typename T>
class CRandFiller {
public:
    CRandFiller(float min, float max) {
        m_min = min;
        m_max = max;
        m_uni = std::uniform_real_distribution<float>(m_min, m_max);
    }

    T GetRand() {
        return static_cast<T>(m_uni(m_eng));
    }

protected:
    std::mt19937 m_eng;
    std::uniform_real_distribution<float> m_uni;
    float m_min, m_max;
};