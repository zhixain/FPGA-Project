#include <iostream>
#include "ap_fixed.h"

typedef ap_fixed<24,12> fixed_point_t;

// 測試參數
#define C 2   // 通道數
#define H 4   // 高度
#define W 4   // 寬度

// 宣告 PReLU 函數
void prelu(
    ap_uint<16> in_C, ap_uint<16> in_H, ap_uint<16> in_W,
    fixed_point_t feature_in[], fixed_point_t alpha[], fixed_point_t feature_out[]
);

int main() {
    // 設定輸入特徵圖
    fixed_point_t feature_in[C * H * W] = {
        1.0, -1.0, 2.0, -2.0,
        3.0, -3.0, 4.0, -4.0,
        5.0, -5.0, 6.0, -6.0,
        7.0, -7.0, 8.0, -8.0,

        -1.0, 1.0, -2.0, 2.0,
        -3.0, 3.0, -4.0, 4.0,
        -5.0, 5.0, -6.0, 6.0,
        -7.0, 7.0, -8.0, 8.0
    };

    // PReLU 參數（第一個通道 α = 0.1，第二個通道 α = 0.5）
    fixed_point_t alpha[C] = { 0.1, 0.5 };

    // 輸出特徵圖
    fixed_point_t feature_out[C * H * W];

    // 執行 PReLU
    prelu(C, H, W, feature_in, alpha, feature_out);

    // 輸出結果
    std::cout << "PReLU Output:\n";
    for (int c = 0; c < C; c++) {
        std::cout << "Channel " << c << ":\n";
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                std::cout << feature_out[c * H * W + h * W + w] << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
