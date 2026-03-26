#include <iostream>
#include "ap_fixed.h"

// ㄏノ INT16 ﹚翴计 (8-bit 俱计场だ + 8-bit 计场だ)
typedef ap_fixed<16,8> fixed_point_t;

// 代刚把计
#define TEST_IN_SIZE 4   // 代刚块
#define TEST_OUT_SIZE 3  // 代刚块

//  Linear ㄧ计
void linear(
    ap_uint<16> in_size, ap_uint<16> out_size,
    fixed_point_t feature_in[], fixed_point_t weight[], fixed_point_t feature_out[]
);

int main() {
    // 砞﹚块疭紉秖
    fixed_point_t feature_in[TEST_IN_SIZE] = {1.0, 2.0, 3.0, 4.0};

    // 砞﹚舦痻皚 (out_size x in_size)
    fixed_point_t weight[TEST_OUT_SIZE * TEST_IN_SIZE] = {
        1.0, 0.5, 0.0, -0.5,
        -0.5, 1.0, 0.5, 0.0,
        0.0, -0.5, 1.0, 0.5
    };

    // 块秖
    fixed_point_t feature_out[TEST_OUT_SIZE];

    // 磅︽ Linear
    linear(TEST_IN_SIZE, TEST_OUT_SIZE, feature_in, weight, feature_out);

    // 块挡狦
    std::cout << "Linear Output (No Bias):\n";
    for (int i = 0; i < TEST_OUT_SIZE; i++) {
        std::cout << feature_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
