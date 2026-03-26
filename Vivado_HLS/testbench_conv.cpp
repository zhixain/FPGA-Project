#include <iostream>
#include "ap_fixed.h"

typedef float fixed_point_t;

// 測試參數
#define TEST_IN_C 4    // 測試輸入通道
#define TEST_OUT_C 4   // 測試輸出通道
#define TEST_IN_H 4    // 測試輸入高度
#define TEST_IN_W 4    // 測試輸入寬度
#define TEST_K 3       // Kernel 大小
#define TEST_STRIDE 1  // 步長
#define TEST_PADDING 1 // Padding
#define TEST_GROUPS 2  // 分組數量

// 宣告 Grouped Convolution 函數
void conv2d(
    ap_uint<16> in_c, ap_uint<16> out_c,
    ap_uint<16> in_h, ap_uint<16> in_w,
    ap_uint<8> k, ap_uint<8> stride, ap_uint<8> padding, ap_uint<8> groups,
    fixed_point_t feature_in[], fixed_point_t weight[], fixed_point_t feature_out[]
);

int main() {
    // 初始化輸入 Feature Map
    fixed_point_t feature_in[TEST_IN_C * TEST_IN_H * TEST_IN_W];
    fixed_point_t weight[TEST_OUT_C * (TEST_IN_C / TEST_GROUPS) * TEST_K * TEST_K];
    fixed_point_t feature_out[TEST_OUT_C * TEST_IN_H * TEST_IN_W];

    for (int i = 0; i < TEST_IN_C * TEST_IN_H * TEST_IN_W; i++) {
        feature_in[i] = i % 4; // 模擬輸入資料
    }

    for (int i = 0; i < TEST_OUT_C * (TEST_IN_C / TEST_GROUPS) * TEST_K * TEST_K; i++) {
        weight[i] = i*0.1; // 模擬權重
    }

    // 執行 Grouped Convolution
    conv2d(TEST_IN_C, TEST_OUT_C, TEST_IN_H, TEST_IN_W, TEST_K, TEST_STRIDE, TEST_PADDING, TEST_GROUPS,
                   feature_in, weight, feature_out);

    // 輸出結果
    std::cout << "Grouped Convolution Output:\n";
    for (int c = 0; c < TEST_OUT_C; c++) {
        std::cout << "Channel " << c << ":\n";
        for (int h = 0; h < TEST_IN_H; h++) {
            for (int w = 0; w < TEST_IN_W; w++) {
                int idx = c * TEST_IN_H * TEST_IN_W + h * TEST_IN_W + w;
                std::cout << feature_out[idx] << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
