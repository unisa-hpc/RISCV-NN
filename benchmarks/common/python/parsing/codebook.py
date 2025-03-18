# This file should be in sync with codebook.h .
# Any modification here should be reflected in codebook.h

import pandas as pd


def translate_str_codename_to(codename: str, to_style: str) -> str:
    ret_val = ""

    if not (to_style == 'long1' or to_style == 'brief1') :
        print("Invalid to_style: " + to_style)
        return None

    # bId\d+_(base|ours)_(SAV,SNA,AVX2,AVX512,RVV,CUDA)_\d+
    # get subparts
    parts = codename.split('_')
    if len(parts) != 4:
        print("Invalid codename: " + codename)
        return None

    bench_id = int(parts[0][3:])
    is_base = parts[1] == 'base'
    kind = parts[2]
    kind_index = int(parts[3])

    if to_style=='long1':
        if bench_id == 0:
            ret_val = \
                "Scalar Vector Multiplication (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Vector Multiplication (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "Scalar Vector Shift (NoAutoVec)" if is_base and kind == "SNA" and kind_index==1 else \
                "Scalar Vector Shift (AutoVec)" if is_base and kind == "SAV" and kind_index==1 else \
                "AVX Vector Multiplication" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX Vector Multiplication" if (not is_base) and kind == "AVX2" and kind_index==0 else \
                None

        elif bench_id == 2:
            ret_val = \
                "Scalar Matmul With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Matmul With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2 Matmul With MUL" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX2 Matmul With SHIFT" if (not is_base) and kind == "AVX2" and kind_index==0 else \
                None
        elif bench_id == 3:
            ret_val = \
                "Scalar Direct OCHW Conv2D With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Direct OCHW Conv2D With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2 Direct OCHW Conv2D With MUL" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX2 Direct OCHW Conv2D With SHIFT" if (not is_base) and kind == "AVX2" and kind_index==0 else \
                None
        elif bench_id == 7:
            ret_val = \
                "Scalar Matmul With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Matmul With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2 F32:F32 Matmul With MUL" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX512 F32:F32 Matmul With MUL" if is_base and kind == "AVX512" and kind_index==0 else \
                "AVX512 F32:U8 NoPack Matmul With PoT - Unpack1" if (not is_base) and kind == "AVX512" and kind_index==0 else \
                None
        elif bench_id == 8:
            ret_val = \
                "Scalar Matmul With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Matmul With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2 F32:F32 Matmul With MUL" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX512 F32:F32 Matmul With MUL" if is_base and kind == "AVX512" and kind_index==0 else \
                "AVX512 F32:U8 NoPack Matmul With PoT - Unpack2" if (not is_base) and kind == "AVX512" and kind_index==0 else \
                None
        elif bench_id == 4:
            ret_val = \
                "Naive Matmul" if is_base and kind == "CUDA" and kind_index==0 else \
                "SMem 1D-Tiled Matmul" if is_base and kind == "CUDA" and kind_index==1 else \
                "SMem 2D-Tiled Matmul" if is_base and kind == "CUDA" and kind_index==2 else \
                "Naive F32:U8 NoPack Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==0 else \
                "Naive F32:U16 NoPack Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==1 else \
                "Naive F32:U8 NoPack Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==2 else \
                "Naive F32:U8 Pack2 Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==3 else \
                "SMem 1D-Tiled F32:U8 Pack2 Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==4 else \
                "SMem 1D-Tiled F32:U8 Pack4 Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==5 else \
                None
        elif bench_id == 1:
            ret_val = \
                "Scalar Matmul With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Matmul With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "RVV Matmul With MUL" if is_base and kind == "RVV" and kind_index==0 else \
                "RVV Matmul With SHIFT" if (not is_base) and kind == "RVV" and kind_index==0 else \
                None
        elif bench_id == 5:
            ret_val = \
                "Scalar Matmul With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Matmul With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "RVV Matmul With MUL" if is_base and kind == "RVV" and kind_index==0 else \
                "RVV F32:U8 NoPack Matmul With PoT" if (not is_base) and kind == "RVV" and kind_index==0 else \
                None
        elif bench_id == 6:
            ret_val = \
                "Scalar Matmul With MUL (NoAutoVec)" if is_base and kind == "SNA" and kind_index==0 else \
                "Scalar Matmul With MUL (AutoVec)" if is_base and kind == "SAV" and kind_index==0 else \
                "RVV Matmul With MUL" if is_base and kind == "RVV" and kind_index==0 else \
                "RVV F32:U8 Pack2 Matmul With PoT" if (not is_base) and kind == "RVV" and kind_index==0 else \
                None
        else:
            print("Invalid bench_id: " + str(bench_id))
            return None

    if to_style == 'brief1':
        if bench_id == 0:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "SNAS" if is_base and kind == "SNA" and kind_index==1 else \
                "SAVS" if is_base and kind == "SAV" and kind_index==1 else \
                "AVX2" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX2S" if (not is_base) and kind == "AVX2" and kind_index==0 else \
                None

        elif bench_id == 2:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX2S" if (not is_base) and kind == "AVX2" and kind_index==0 else \
                None
        elif bench_id == 3:
            # conv2d
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX2S" if (not is_base) and kind == "AVX2" and kind_index==0 else \
                None
        elif bench_id == 7:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX5" if is_base and kind == "AVX512" and kind_index==0 else \
                "AVX5P" if (not is_base) and kind == "AVX512" and kind_index==0 else \
                None
        elif bench_id == 8:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "AVX2" if is_base and kind == "AVX2" and kind_index==0 else \
                "AVX5" if is_base and kind == "AVX512" and kind_index==0 else \
                "AVX5P" if (not is_base) and kind == "AVX512" and kind_index==0 else \
                None
        elif bench_id == 4:
            ret_val = \
                "Naive Matmul" if is_base and kind == "CUDA" and kind_index==0 else \
                "SMem 1D-Tiled Matmul" if is_base and kind == "CUDA" and kind_index==1 else \
                "SMem 2D-Tiled Matmul" if is_base and kind == "CUDA" and kind_index==2 else \
                "Naive F32:U8 NoPack Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==0 else \
                "Naive F32:U16 NoPack Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==1 else \
                "Naive F32:U8 NoPack Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==2 else \
                "Naive F32:U8 Pack2 Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==3 else \
                "SMem 1D-Tiled F32:U8 Pack2 Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==4 else \
                "SMem 1D-Tiled F32:U8 Pack4 Matmul With PoT" if (not is_base) and kind == "CUDA" and kind_index==5 else \
                None
        elif bench_id == 1:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "RVV" if is_base and kind == "RVV" and kind_index==0 else \
                "RVVS" if (not is_base) and kind == "RVV" and kind_index==0 else \
                None
        elif bench_id == 5:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "RVV" if is_base and kind == "RVV" and kind_index==0 else \
                "RVVP" if (not is_base) and kind == "RVV" and kind_index==0 else \
                None
        elif bench_id == 6:
            ret_val = \
                "SNA" if is_base and kind == "SNA" and kind_index==0 else \
                "SAV" if is_base and kind == "SAV" and kind_index==0 else \
                "RVV" if is_base and kind == "RVV" and kind_index==0 else \
                "RVVP" if (not is_base) and kind == "RVV" and kind_index==0 else \
                None
        else:
            print("Invalid bench_id: " + str(bench_id))
            return None

    return ret_val


def translate_codename_to(series: pd.Series, to_style: str = 'brief1') -> pd.Series:
    return series.apply(lambda x: translate_str_codename_to(x, to_style))

def translate_str_compiler_name_to(compiler_name: str, to_style: str) -> str:
    ret_val = ""

    if not (to_style == 'long1' or to_style == 'brief1'):
        print("Invalid to_style: " + to_style)
        return None

    if to_style == 'long1':
        ret_val = \
            "GCC13.3" if "g++" in compiler_name and "13" in compiler_name else \
            "GCC14.2" if "g++" in compiler_name and "14" in compiler_name else \
            "LLVM17" if "clang" in compiler_name and "17" in compiler_name else \
            "LLVM18" if "clang" in compiler_name and "18" in compiler_name else \
            None

    if to_style == 'brief1':
        ret_val = \
            "G13" if "g++" in compiler_name and "13" in compiler_name else \
            "G14" if "g++" in compiler_name and "14" in compiler_name else \
            "C17" if "clang" in compiler_name and "17" in compiler_name else \
            "C18" if "clang" in compiler_name and "18" in compiler_name else \
            None

    return ret_val

def translate_compiler_name_to(series: pd.Series, to_style: str = 'long1') -> pd.Series:
    return series.apply(lambda x: translate_str_compiler_name_to(x, to_style))

def translate_str_benchId_to(benchId: str, to_style: str, reverse=False) -> str:
    ret_val = ""

    if not (to_style == 'long1' or to_style == 'brief1'):
        print("Invalid to_style: " + to_style)
        return None

    if to_style == 'long1':
        raise NotImplementedError("Not implemented yet")

    if to_style == 'brief1':
        if not reverse:
            ret_val = \
                "F32:U8:E5:P1 Unpack1" if benchId == 7 else \
                "F32:U16:E8:P1 Unpack2" if benchId == 8 else \
                "F32:U8:E5:P1 Unpack1+InfNan" if benchId == 10 else \
                "FXPoT AVX512" if benchId == 2 else \
                "F32:U8:E5:P1 Unpack 1" if benchId == 5 else \
                "F32:U8:E3:P2 Unpack 1" if benchId == 6 else \
                "FXPoT RVV" if benchId == 1 else \
                None
        else:
            ret_val = \
                7 if benchId == "F32:U8:E5:P1 Unpack1" else \
                8 if benchId == "F32:U16:E8:P1 Unpack2" else \
                10 if benchId == "F32:U8:E5:P1 Unpack1+InfNan" else \
                2 if benchId == "FXPoT AVX512" else \
                5 if benchId == "F32:U8:E5:P1 Unpack 1" else \
                6 if benchId == "F32:U8:E3:P2 Unpack 1" else \
                1 if benchId == "FXPoT RVV" else \
                None

    return ret_val

def translate_benchId_to(series: pd.Series, to_style: str = 'brief1') -> pd.Series:
    return series.apply(lambda x: translate_str_benchId_to(x, to_style))