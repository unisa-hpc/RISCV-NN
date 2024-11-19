//
// Created by saleh on 11/19/24.
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
#include "common01.h"

constexpr size_t RUNS = 1000;
constexpr size_t N = 1024 * 1024; // 16M elements
constexpr size_t VECTOR_ELEMENTS = 8; // 8 elements in a vector (int32)
