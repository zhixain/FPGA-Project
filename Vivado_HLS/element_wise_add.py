#include <ap_fixed.h>
#include <iostream>

// 使用定點數
typedef float fixed_point_t;

// Element-wise Add（跳躍連接）
void elementwise_add(
    ap_uint<16> size,    // 向量大小
    fixed_point_t input1[],  // 輸入 1
    fixed_point_t input2[],  // 輸入 2
    fixed_point_t output[]   // 輸出向量
) {
    #pragma HLS INTERFACE mode=m_axi port=output
    #pragma HLS INTERFACE mode=m_axi port=input2
    #pragma HLS INTERFACE mode=m_axi port=input1
    #pragma HLS INTERFACE mode=s_axilite port=size
    #pragma HLS INTERFACE mode=s_axilite port=return

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        output[i] = input1[i] + input2[i];  // 逐元素加法
    }
}
