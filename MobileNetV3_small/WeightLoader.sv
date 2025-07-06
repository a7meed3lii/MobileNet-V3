// Weight Loader Module for MobileNetV3_Small - REFACTORED
// This version loads the pre-calculated, inference-optimized weights for the fused
// BatchNorm layers and removes all incorrect bias and old BatchNorm parameters.

module WeightLoader #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 32,
    parameter NUM_CLASSES = 15
)(
    input  logic clk,
    input  logic rst,
    
    // Memory-mapped interface for weight loading
    input  logic [ADDR_WIDTH-1:0] weight_addr,
    input  logic [DATA_WIDTH-1:0] weight_data,
    input  logic weight_write_en,
    input  logic weight_load_start,
    output logic weight_load_done,
    output logic weight_load_error,
    
    // REFACTORED: Weight outputs for all layers
    output logic signed [DATA_WIDTH-1:0] conv1_weights [0:15][0:0][0:2][0:2],
    output logic signed [DATA_WIDTH-1:0] conv1_bn_w [0:15],
    output logic signed [DATA_WIDTH-1:0] conv1_bn_b [0:15],
    
    output logic signed [DATA_WIDTH-1:0] conv2_weights [0:575][0:95],
    output logic signed [DATA_WIDTH-1:0] conv2_bn_w [0:575],
    output logic signed [DATA_WIDTH-1:0] conv2_bn_b [0:575],
    
    output logic signed [DATA_WIDTH-1:0] linear3_weights [0:1279][0:575],
    output logic signed [DATA_WIDTH-1:0] linear3_bn_w [0:1279],
    output logic signed [DATA_WIDTH-1:0] linear3_bn_b [0:1279],
    output logic signed [DATA_WIDTH-1:0] linear4_weights [0:NUM_CLASSES-1][0:1279],
    output logic signed [DATA_WIDTH-1:0] linear4_bias [0:NUM_CLASSES-1],
    
    output logic signed [DATA_WIDTH-1:0] block_expand_weights [0:10][0:575][0:575],
    output logic signed [DATA_WIDTH-1:0] block_expand_bn_w [0:10][0:575],
    output logic signed [DATA_WIDTH-1:0] block_expand_bn_b [0:10][0:575],
    
    output logic signed [DATA_WIDTH-1:0] block_dw_weights [0:10][0:575][0:4][0:4],
    output logic signed [DATA_WIDTH-1:0] block_dw_bn_w [0:10][0:575],
    output logic signed [DATA_WIDTH-1:0] block_dw_bn_b [0:10][0:575],
    
    output logic signed [DATA_WIDTH-1:0] block_pw_weights [0:10][0:95][0:575],
    output logic signed [DATA_WIDTH-1:0] block_pw_bn_w [0:10][0:95],
    output logic signed [DATA_WIDTH-1:0] block_pw_bn_b [0:10][0:95],
    
    output logic signed [DATA_WIDTH-1:0] block_se_conv1_weights [0:10][0:23][0:95],
    output logic signed [DATA_WIDTH-1:0] block_se_conv1_bn_w [0:10][0:23],
    output logic signed [DATA_WIDTH-1:0] block_se_conv1_bn_b [0:10][0:23],
    
    output logic signed [DATA_WIDTH-1:0] block_se_conv2_weights [0:10][0:95][0:23],
    output logic signed [DATA_WIDTH-1:0] block_se_conv2_bn_w [0:10][0:95],
    output logic signed [DATA_WIDTH-1:0] block_se_conv2_bn_b [0:10][0:95],
    
    output logic signed [DATA_WIDTH-1:0] block_shortcut_weights [0:10][0:95][0:95],
    output logic signed [DATA_WIDTH-1:0] block_shortcut_bn_w [0:10][0:95],
    output logic signed [DATA_WIDTH-1:0] block_shortcut_bn_b [0:10][0:95],
    
    output logic [10:0] block_use_se,
    output logic [4:0] block_kernel_size [0:10],
    output logic [1:0] block_stride [0:10],
    output logic [9:0] block_in_channels [0:10],
    output logic [9:0] block_expand_channels [0:10],
    output logic [9:0] block_out_channels [0:10],
    output logic [9:0] block_se_reduce_channels [0:10]
);

    // ... (State machine and loading logic remains the same)

    // REFACTORED: Weight mapping from internal memory to output arrays
    always_comb begin
        // ... (Mapping logic for all layers)
    end

    // ... (Configuration initialization remains the same)

endmodule