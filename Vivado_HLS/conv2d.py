#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

typedef float fixed_point_t;

void conv2d(
    ap_uint<16> in_c, ap_uint<16> out_c,
    ap_uint<16> in_h, ap_uint<16> in_w,
    ap_uint<8> k, ap_uint<8> stride,
    ap_uint<8> padding, ap_uint<16> groups,
    fixed_point_t feature_in[],
    fixed_point_t weight[],
    fixed_point_t feature_out[]
) {
#pragma HLS INTERFACE m_axi port=feature_out offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=feature_in offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=in_c
#pragma HLS INTERFACE s_axilite port=out_c
#pragma HLS INTERFACE s_axilite port=in_h
#pragma HLS INTERFACE s_axilite port=in_w
#pragma HLS INTERFACE s_axilite port=k
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=padding
#pragma HLS INTERFACE s_axilite port=groups
#pragma HLS INTERFACE s_axilite port=return

    // --- 加上Array Partition ---
#pragma HLS ARRAY_PARTITION variable=feature_in cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=8 dim=1

    ap_uint<16> group_in_c = in_c / groups;
    ap_uint<16> group_out_c = out_c / groups;
    ap_uint<16> out_h = (in_h + 2 * padding - k) / stride + 1;
    ap_uint<16> out_w = (in_w + 2 * padding - k) / stride + 1;

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < group_out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {

                    int out_idx = (g * group_out_c + oc) * out_h * out_w + oh * out_w + ow;
                    fixed_point_t sum = 0;

                    for (int ic = 0; ic < group_in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
#pragma HLS PIPELINE II=1
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && iw >= 0 && ih < in_h && iw < in_w) {
                                    int in_idx = (g * group_in_c + ic) * in_h * in_w + ih * in_w + iw;
                                    int w_idx = (g * group_out_c + oc) * group_in_c * k * k
                                              + ic * k * k + kh * k + kw;

                                    sum += feature_in[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    feature_out[out_idx] = sum;
                }
            }
        }
    }
}
