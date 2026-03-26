#include <iostream>
#include "ap_fixed.h"

// ㄏノ INT16 ﹚翴计 (8-bit 俱计场だ + 8-bit 计场だ)
typedef ap_fixed<16,8> fixed_point_t;

// 代刚把计
#define TEST_SIZE 5   // 代刚块

//  BatchNorm1d ㄧ计
void batch_norm1d(
    ap_uint<16> in_size,
    fixed_point_t feature_in[], fixed_point_t scale[], fixed_point_t bias[], fixed_point_t feature_out[]
);

int main() {
    // 砞﹚块秖
    fixed_point_t feature_in[TEST_SIZE] = {1.0, -1.0, 2.0, -2.0, 0.0};

    // 砞﹚ BatchNorm 把计箇璸衡 scale ㎝ bias
    fixed_point_t scale[TEST_SIZE] = {2.0, 0.5, -1.0, 1.5, 1.0};
    fixed_point_t bias[TEST_SIZE] = {1.0, -1.0, 0.5, -0.5, 0.0};

    // 块秖
    fixed_point_t feature_out[TEST_SIZE];

    // 磅︽ BatchNorm1d
    batch_norm1d(TEST_SIZE, feature_in, scale, bias, feature_out);

    // 块挡狦
    std::cout << "BatchNorm1D Output:\n";
    for (int i = 0; i < TEST_SIZE; i++) {
        std::cout << feature_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
