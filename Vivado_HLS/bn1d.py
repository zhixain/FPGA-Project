#include <ap_fixed.h>
#include <iostream>

typedef float fixed_point_t;

// BatchNorm1d 層實作 (優化版)
void batch_norm1d(
    ap_uint<16> in_size,    // 輸入大小
    fixed_point_t feature_in[],  // 輸入向量
    fixed_point_t scale[],       // 預計算的 scale
    fixed_point_t bias[],        // 預計算的 bias
    fixed_point_t feature_out[]  // 輸出向量
) {
#pragma HLS INTERFACE mode=m_axi port=feature_out
#pragma HLS INTERFACE mode=m_axi port=bias
#pragma HLS INTERFACE mode=m_axi port=scale
#pragma HLS INTERFACE mode=m_axi port=feature_in
#pragma HLS INTERFACE mode=s_axilite port=in_size
#pragma HLS INTERFACE mode=s_axilite port=return

    // --- 加上Array Partition ---
#pragma HLS ARRAY_PARTITION variable=scale complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    // 遍歷整個輸入向量
    for (int i = 0; i < in_size; i++) {
#pragma HLS PIPELINE II=1
        feature_out[i] = feature_in[i] * scale[i] + bias[i];
    }
}
