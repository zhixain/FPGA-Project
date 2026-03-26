#include <iostream>
#include "ap_fixed.h"

// 使用 INT16 定點數
typedef ap_fixed<16,8> fixed_point_t;

// 測試參數
#define TEST_SIZE 5   // 測試向量大小

// 宣告 Element-wise Add 函數
void elementwise_add(
    ap_uint<16> size, fixed_point_t input1[], fixed_point_t input2[], fixed_point_t output[]
);

int main() {
    // 設定輸入向量
    fixed_point_t input1[TEST_SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0};
    fixed_point_t input2[TEST_SIZE] = {0.5, -0.5, 1.5, -1.5, 0.0};

    // 輸出向量
    fixed_point_t output[TEST_SIZE];

    // 執行 Element-wise Add
    elementwise_add(TEST_SIZE, input1, input2, output);

    // 輸出結果
    std::cout << "Element-wise Add Output:\n";
    for (int i = 0; i < TEST_SIZE; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
