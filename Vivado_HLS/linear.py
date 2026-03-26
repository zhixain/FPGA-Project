#include <ap_fixed.h>
#include <iostream>

// 使用 INT16 定點數 (8-bit 整數部分 + 8-bit 小數部分)
typedef float fixed_point_t;

// Linear 層 (Fully Connected Layer) 無 Bias (優化版)
void linear(
    ap_uint<16> in_size,    // 輸入向量大小
    ap_uint<16> out_size,   // 輸出向量大小
    fixed_point_t feature_in[],  // 輸入向量
    fixed_point_t weight[],      // 權重矩陣
    fixed_point_t feature_out[]  // 輸出向量
) {
#pragma HLS INTERFACE mode=m_axi port=feature_out
#pragma HLS INTERFACE mode=m_axi port=weight
#pragma HLS INTERFACE mode=m_axi port=feature_in
#pragma HLS INTERFACE mode=s_axilite port=in_size
#pragma HLS INTERFACE mode=s_axilite port=out_size
#pragma HLS INTERFACE mode=s_axilite port=return

    // --- 加上Array Partition ---
#pragma HLS ARRAY_PARTITION variable=feature_in complete dim=1

    // 遍歷輸出節點
    for (int o = 0; o < out_size; o++) {
        fixed_point_t sum = 0;  // 移除 Bias，sum 初始值設為 0

        for (int i = 0; i < in_size; i++) {
#pragma HLS PIPELINE II=1
            sum += feature_in[i] * weight[o * in_size + i];
        }

        feature_out[o] = sum;
    }
}
