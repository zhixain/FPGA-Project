#include <iostream>
#include <ap_fixed.h>
typedef float fixed_point_t;

#define IN_C 1
#define OUT_C 1
#define IN_H 3
#define IN_W 3
#define K 3
#define STRIDE 1
#define PADDING 0
#define GROUPS 1

void conv2d(
    ap_uint<16> in_c, ap_uint<16> out_c,
    ap_uint<16> in_h, ap_uint<16> in_w,
    ap_uint<8> k, ap_uint<8> stride,
    ap_uint<8> padding, ap_uint<8> groups,
    fixed_point_t feature_in[],
    fixed_point_t weight[],
    fixed_point_t feature_out[]
);

int main() {
    const int in_size = IN_H * IN_W * IN_C;
    const int weight_size = OUT_C * IN_C * K * K;
    const int out_h = (IN_H + 2 * PADDING - K) / STRIDE + 1;
    const int out_w = (IN_W + 2 * PADDING - K) / STRIDE + 1;
    const int out_size = out_h * out_w * OUT_C;

    fixed_point_t feature_in[in_size];
    fixed_point_t weight[weight_size];
    fixed_point_t feature_out[out_size];

    // 初始化輸入 feature_in[h][w][c] = h + w
    for (int h = 0; h < IN_H; h++) {
        for (int w = 0; w < IN_W; w++) {
            for (int c = 0; c < IN_C; c++) {
                int idx = h * IN_W * IN_C + w * IN_C + c;
                feature_in[idx] = h + w;
            }
        }
    }

    // 初始化權重全部為 1
    for (int i = 0; i < weight_size; i++) {
        weight[i] = 1.0f*i;
    }

    conv2d(IN_C, OUT_C, IN_H, IN_W, K, STRIDE, PADDING, GROUPS,
           feature_in, weight, feature_out);

    std::cout << "HLS conv2d Output [H][W][C]:\n";
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            std::cout << "[";
            for (int c = 0; c < OUT_C; c++) {
                int idx = h * out_w * OUT_C + w * OUT_C + c;
                std::cout << feature_out[idx];
                if (c < OUT_C - 1) std::cout << ", ";
            }
            std::cout << "] ";
        }
        std::cout << "\n";
    }

    return 0;
}
