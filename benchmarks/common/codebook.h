//
// Created by saleh on 1/18/25.
//

// This file should be in sync with codebook.py .
// Any modification here should be reflected in codebook.py

#pragma once

#include <string>

enum kernel_kind {
  ScalarAutoVec,
  ScalarNoAutoVec,
  AVX2,
  AVX512,
  RVV,
  CUDA,
};

std::string get_code_name(int bench_id, kernel_kind kind, bool is_baseline, int kind_index) {
    // The idea is that each kind can be (baseline, ours) and there could be multiple versions of each kind.

    std::string codeName = "bId" + std::to_string(bench_id) + "_";
    codeName += is_baseline? "base_": "ours_";
    switch (kind) {
        case ScalarAutoVec:
            codeName += "SAV_";
            break;
        case ScalarNoAutoVec:
            codeName += "SNA_";
            break;
        case AVX2:
            codeName += "AVX2_";
            break;
        case AVX512:
            codeName += "AVX512_";
            break;
        case RVV:
            codeName += "RVV_";
            break;
        case CUDA:
            codeName += "CUDA_";
            break;
    }
    codeName += std::to_string(kind_index);

    return codeName;
}
