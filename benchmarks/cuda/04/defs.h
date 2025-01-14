//
// Created by saleh on 11/19/24.
//

#pragma once

#include <iostream>

constexpr size_t RUNS = 128;

// fallback to the default if not defined
#ifndef UNROLL_FACTOR0
#define UNROLL_FACTOR0 1
#endif

// fallback to 256 if not defined
#ifndef N
#define N 256
#endif
