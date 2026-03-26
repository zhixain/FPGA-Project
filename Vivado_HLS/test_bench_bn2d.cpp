#include <iostream>
#include "ap_fixed.h"

typedef ap_fixed<16,8> fixed_point_t;

// 代刚把计
#define C 2   // 硄笵计
#define H 4   // 蔼
#define W 4   // 糴

//  BatchNorm2d ㄧ计
void batch_norm2d(
    ap_uint<16> in_C, ap_uint<16> in_H, ap_uint<16> in_W,
    fixed_point_t feature_in[], fixed_point_t scale[], fixed_point_t bias[], fixed_point_t feature_out[]
);

int main() {
    // 砞﹚块疭紉瓜 (场 1.0)
    fixed_point_t feature_in[C * H * W];
    fixed_point_t scale[C];
    fixed_point_t bias[C];
    fixed_point_t feature_out[C * H * W];

    // ﹍て块计沮
    for (int i = 0; i < C * H * W; i++) {
        feature_in[i] = 2.0;
    }

    // ﹍て BatchNorm 把计
    for (int i = 0; i < C; i++) {
        scale[i] = 2.0;  // 砞﹚ `scale = 2`
        bias[i] = -1.0;  // 砞﹚ `bias = -1`
    }

    // 磅︽ BatchNorm2d
    batch_norm2d(C, H, W, feature_in, scale, bias, feature_out);

    // 块挡狦
    std::cout << "BatchNorm2D Output:\n";
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
