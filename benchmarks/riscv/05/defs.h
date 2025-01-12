//
// Created by saleh on 11/19/24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <string>
#include <cmath>
#include <riscv_vector.h>
#include "common01.h"

constexpr size_t RUNS = 16;

// fallback to 1 if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 1
#endif

// fallback to 256 if not defined
#ifndef N
#define N 256
#endif
