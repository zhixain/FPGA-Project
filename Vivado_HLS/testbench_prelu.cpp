#include <iostream>
#include <ap_fixed.h>
typedef float fixed_point_t;

#define IN_H 2
#define IN_W 2
#define IN_C 3

void prelu(
    ap_uint<16> in_C,    // 通道數
    ap_uint<16> in_H,    // 特徵圖高度
    ap_uint<16> in_W,    // 特徵圖寬度
    fixed_point_t feature_in[],  // 輸入特徵圖 [H][W][C]
    fixed_point_t alpha[],       // 每個通道的 PReLU 參數
    fixed_point_t feature_out[]  // 輸出特徵圖 [H][W][C]
);

int main() {
    const int size = IN_H * IN_W * IN_C;

    fixed_point_t feature_in[size];
    fixed_point_t feature_out[size];
    fixed_point_t alpha[IN_C] = {0.1f, 0.2f, 0.3f};  // 每通道的 PReLU α 值

    // 初始化輸入：feature_in[h][w][c] = (h + w + c - 2)
    for (int h = 0; h < IN_H; h++) {
        for (int w = 0; w < IN_W; w++) {
            for (int c = 0; c < IN_C; c++) {
                int idx = h * IN_W * IN_C + w * IN_C + c;
                feature_in[idx] = static_cast<fixed_point_t>(h + w + c - 2);
            }
        }
    }

    // 呼叫 prelu 函式
    prelu(IN_C, IN_H, IN_W, feature_in, alpha, feature_out);

    // 顯示結果
    std::cout << "HLS PReLU Output [H][W][C]:\n";
    for (int h = 0; h < IN_H; h++) {
        for (int w = 0; w < IN_W; w++) {
            std::cout << "[";
            for (int c = 0; c < IN_C; c++) {
                int idx = h * IN_W * IN_C + w * IN_C + c;
                std::cout << feature_out[idx];
                if (c < IN_C - 1) std::cout << ", ";
            }
            std::cout << "] ";
        }
        std::cout << "\n";
    }

    return 0;
}
