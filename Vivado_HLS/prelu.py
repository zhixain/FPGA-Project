#include <ap_fixed.h>
#include <iostream>

// 使用 INT16 定點數 (8-bit 整數部分 + 8-bit 小數部分)
typedef float fixed_point_t;

// PReLU 層實作 (優化版)
void prelu(
    ap_uint<16> in_C,    // 通道數
    ap_uint<16> in_H,    // 特徵圖高度
    ap_uint<16> in_W,    // 特徵圖寬度
    fixed_point_t feature_in[],  // 輸入特徵圖
    fixed_point_t alpha[],       // 每個通道的 PReLU 參數
    fixed_point_t feature_out[]  // 輸出特徵圖
) {
#pragma HLS INTERFACE mode=m_axi port=feature_out
#pragma HLS INTERFACE mode=m_axi port=alpha
#pragma HLS INTERFACE mode=m_axi port=feature_in
#pragma HLS INTERFACE mode=s_axilite port=in_W
#pragma HLS INTERFACE mode=s_axilite port=in_H
#pragma HLS INTERFACE mode=s_axilite port=in_C
#pragma HLS INTERFACE mode=s_axilite port=return

    // --- 加上Array Partition ---
#pragma HLS ARRAY_PARTITION variable=alpha complete dim=1

    // 遍歷整個特徵圖
    for (int c = 0; c < in_C; c++) {  // 遍歷通道
        for (int h = 0; h < in_H; h++) {  // 遍歷高度
            for (int w = 0; w < in_W; w++) {  // 遍歷寬度
#pragma HLS PIPELINE II=1
                int index = c * in_H * in_W + h * in_W + w;
                fixed_point_t x = feature_in[index];

                // PReLU 計算
                fixed_point_t neg_part = alpha[c] * x;
                feature_out[index] = (x > 0) ? x : neg_part;
            }
        }
    }
}
