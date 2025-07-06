module MobileNetV3_Small #(
    parameter IN_CHANNELS = 1,
    parameter NUM_CLASSES = 15,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4,
    parameter IMG_HEIGHT = 224,
    parameter IMG_WIDTH = 224
)(
    //-Input
    input  logic clk,
    input  logic rst,
    input  logic signed [DATA_WIDTH-1:0] data_in [0:IMG_HEIGHT-1][0:IMG_WIDTH-1][0:IN_CHANNELS-1],
    //-Output
    output logic signed [DATA_WIDTH-1:0] data_out [0:NUM_CLASSES-1],
    
    // FIXED: Simplified interface using WeightLoader
    // Memory-mapped interface for weight loading
    input  logic [31:0] weight_addr,
    input  logic [DATA_WIDTH-1:0] weight_data,
    input  logic weight_write_en,
    input  logic weight_load_start,
    output logic weight_load_done,
    output logic weight_load_error,
    
    // FIXED: Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out,
    output logic ready_out,
    
    // ENHANCED: Comprehensive overflow detection and debugging
    output logic overflow_detected,
    output logic [15:0] overflow_count,
    output logic [7:0] overflow_stage, // Which stage had overflow
    output logic [1:0] current_pipeline_stage,
    
    // DEBUG: Pipeline stage indicators for each major component
    output logic [1:0] conv1_stage,
    output logic [1:0] blocks_stage [0:10],
    output logic [1:0] conv2_stage,
    output logic [1:0] pool_stage,
    output logic [1:0] linear3_stage,
    output logic [1:0] linear4_stage
);

    // Calculate intermediate feature map dimensions
    // After conv1 (stride=2): 112x112x16
    // After block1 (stride=2): 56x56x16
    // After block2 (stride=2): 28x28x24  
    // After block3 (stride=1): 28x28x24
    // After block4 (stride=2): 14x14x40
    // After block5 (stride=1): 14x14x40
    // After block6 (stride=1): 14x14x40
    // After block7 (stride=1): 14x14x48
    // After block8 (stride=1): 14x14x48
    // After block9 (stride=2): 7x7x96
    // After block10 (stride=1): 7x7x96
    // After block11 (stride=1): 7x7x96
    // After conv2: 7x7x576
    // After pool: 576
    // After linear3: 1280
    // After linear4: 15
    
    // Intermediate feature maps
    logic signed [DATA_WIDTH-1:0] conv1_out [0:111][0:111][0:15];
    logic signed [DATA_WIDTH-1:0] conv1_bn_out [0:111][0:111][0:15];
    logic signed [DATA_WIDTH-1:0] conv1_act_out [0:111][0:111][0:15];
    
    logic signed [DATA_WIDTH-1:0] block1_out [0:55][0:55][0:15];
    logic signed [DATA_WIDTH-1:0] block2_out [0:27][0:27][0:23];
    logic signed [DATA_WIDTH-1:0] block3_out [0:27][0:27][0:23];
    logic signed [DATA_WIDTH-1:0] block4_out [0:13][0:13][0:39];
    logic signed [DATA_WIDTH-1:0] block5_out [0:13][0:13][0:39];
    logic signed [DATA_WIDTH-1:0] block6_out [0:13][0:13][0:39];
    logic signed [DATA_WIDTH-1:0] block7_out [0:13][0:13][0:47];
    logic signed [DATA_WIDTH-1:0] block8_out [0:13][0:13][0:47];
    logic signed [DATA_WIDTH-1:0] block9_out [0:6][0:6][0:95];
    logic signed [DATA_WIDTH-1:0] block10_out [0:6][0:6][0:95];
    logic signed [DATA_WIDTH-1:0] block11_out [0:6][0:6][0:95];
    
    logic signed [DATA_WIDTH-1:0] conv2_out [0:6][0:6][0:575];
    logic signed [DATA_WIDTH-1:0] conv2_bn_out [0:6][0:6][0:575];
    logic signed [DATA_WIDTH-1:0] conv2_act_out [0:6][0:6][0:575];
    
    logic signed [DATA_WIDTH-1:0] pool_out [0:575];
    logic signed [DATA_WIDTH-1:0] linear3_out [0:1279];
    logic signed [DATA_WIDTH-1:0] linear3_bn_out [0:1279];
    logic signed [DATA_WIDTH-1:0] linear3_act_out [0:1279];
    
    // FIXED: Pipeline control signals
    logic valid_pipe [0:18]; // conv1, 11 blocks, conv2, pool, linear3, linear4
    logic ready_pipe [0:18];
    
    // Weight signals from WeightLoader
    logic signed [DATA_WIDTH-1:0] conv1_weights [0:15][0:0][0:2][0:2];
    logic signed [DATA_WIDTH-1:0] conv1_bias [0:15];
    logic signed [DATA_WIDTH-1:0] conv1_bn_w [0:15];
    logic signed [DATA_WIDTH-1:0] conv1_bn_b [0:15];
    
    logic signed [DATA_WIDTH-1:0] conv2_weights [0:575][0:95];
    logic signed [DATA_WIDTH-1:0] conv2_bias [0:575];
    // Standardised BatchNorm interface
    logic signed [DATA_WIDTH-1:0] conv2_bn_w   [0:575];
    logic signed [DATA_WIDTH-1:0] conv2_bn_b   [0:575];
    
    logic signed [DATA_WIDTH-1:0] linear3_weights [0:1279][0:575];
    logic signed [DATA_WIDTH-1:0] linear3_bias [0:1279];
    logic signed [DATA_WIDTH-1:0] linear3_bn_w [0:1279];
    logic signed [DATA_WIDTH-1:0] linear3_bn_b [0:1279];
    
    logic signed [DATA_WIDTH-1:0] linear4_weights [0:NUM_CLASSES-1][0:1279];
    logic signed [DATA_WIDTH-1:0] linear4_bias [0:NUM_CLASSES-1];
    
    // Block configuration and weights from WeightLoader
    logic [10:0] block_use_se;
    logic [4:0] block_kernel_size [0:10];
    logic [1:0] block_stride [0:10];
    logic [9:0] block_in_channels [0:10];
    logic [9:0] block_expand_channels [0:10];
    logic [9:0] block_out_channels [0:10];
    logic [9:0] block_se_reduce_channels [0:10];
    
    // Block weight arrays (simplified interface)
    logic signed [DATA_WIDTH-1:0] block_expand_weights [0:10][0:575][0:575];
    logic signed [DATA_WIDTH-1:0] block_expand_bias [0:10][0:575];
    logic signed [DATA_WIDTH-1:0] block_expand_bn_w [0:10][0:575];
    logic signed [DATA_WIDTH-1:0] block_expand_bn_b [0:10][0:575];
    
    logic signed [DATA_WIDTH-1:0] block_dw_weights [0:10][0:575][0:4][0:4];
    logic signed [DATA_WIDTH-1:0] block_dw_bias [0:10][0:575];
    logic signed [DATA_WIDTH-1:0] block_dw_bn_w [0:10][0:575];
    logic signed [DATA_WIDTH-1:0] block_dw_bn_b [0:10][0:575];
    
    logic signed [DATA_WIDTH-1:0] block_pw_weights [0:10][0:95][0:575];
    logic signed [DATA_WIDTH-1:0] block_pw_bias [0:10][0:95];
    logic signed [DATA_WIDTH-1:0] block_pw_bn_w [0:10][0:95];
    logic signed [DATA_WIDTH-1:0] block_pw_bn_b [0:10][0:95];
    
    logic signed [DATA_WIDTH-1:0] block_se_conv1_weights [0:10][0:23][0:95];
    logic signed [DATA_WIDTH-1:0] block_se_conv1_bn_w [0:10][0:23];
    logic signed [DATA_WIDTH-1:0] block_se_conv1_bn_b [0:10][0:23];
    
    logic signed [DATA_WIDTH-1:0] block_se_conv2_weights [0:10][0:95][0:23];
    logic signed [DATA_WIDTH-1:0] block_se_conv2_bn_w [0:10][0:95];
    logic signed [DATA_WIDTH-1:0] block_se_conv2_bn_b [0:10][0:95];
    
    logic signed [DATA_WIDTH-1:0] block_shortcut_weights [0:10][0:95][0:95];
    logic signed [DATA_WIDTH-1:0] block_shortcut_bias [0:10][0:95];
    logic signed [DATA_WIDTH-1:0] block_shortcut_bn_w [0:10][0:95];
    logic signed [DATA_WIDTH-1:0] block_shortcut_bn_b [0:10][0:95];
    
    // ENHANCED: Overflow detection registers
    logic [15:0] overflow_counter;
    logic overflow_flag;
    
    // FIXED: WeightLoader instantiation
    WeightLoader #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(32),
        .NUM_CLASSES(NUM_CLASSES)
    ) weight_loader (
        .clk(clk),
        .rst(rst),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .weight_write_en(weight_write_en),
        .weight_load_start(weight_load_start),
        .weight_load_done(weight_load_done),
        .weight_load_error(weight_load_error),
        
        // Weight outputs
        .conv1_weights(conv1_weights),
        .conv1_bias(conv1_bias),
        .conv1_bn_w(conv1_bn_w),
        .conv1_bn_b(conv1_bn_b),
        
        .conv2_weights(conv2_weights),
        .conv2_bias(conv2_bias),
        .conv2_bn_w(conv2_bn_w),
        .conv2_bn_b(conv2_bn_b),
        
        .linear3_weights(linear3_weights),
        .linear3_bias(linear3_bias),
        .linear3_bn_w(linear3_bn_w),
        .linear3_bn_b(linear3_bn_b),
        
        .linear4_weights(linear4_weights),
        .linear4_bias(linear4_bias),
        
        // Block weights and configuration
        .block_expand_weights(block_expand_weights),
        .block_expand_bias(block_expand_bias),
        .block_expand_bn_w(block_expand_bn_w),
        .block_expand_bn_b(block_expand_bn_b),
        
        .block_dw_weights(block_dw_weights),
        .block_dw_bias(block_dw_bias),
        .block_dw_bn_w(block_dw_bn_w),
        .block_dw_bn_b(block_dw_bn_b),
        
        .block_pw_weights(block_pw_weights),
        .block_pw_bias(block_pw_bias),
        .block_pw_bn_w(block_pw_bn_w),
        .block_pw_bn_b(block_pw_bn_b),
        
        .block_se_conv1_weights(block_se_conv1_weights),
        .block_se_conv1_bn_w(block_se_conv1_bn_w),
        .block_se_conv1_bn_b(block_se_conv1_bn_b),
        
        .block_se_conv2_weights(block_se_conv2_weights),
        .block_se_conv2_bn_w(block_se_conv2_bn_w),
        .block_se_conv2_bn_b(block_se_conv2_bn_b),
        
        .block_shortcut_weights(block_shortcut_weights),
        .block_shortcut_bias(block_shortcut_bias),
        .block_shortcut_bn_w(block_shortcut_bn_w),
        .block_shortcut_bn_b(block_shortcut_bn_b),
        
        // Configuration
        .block_use_se(block_use_se),
        .block_kernel_size(block_kernel_size),
        .block_stride(block_stride),
        .block_in_channels(block_in_channels),
        .block_expand_channels(block_expand_channels),
        .block_out_channels(block_out_channels),
        .block_se_reduce_channels(block_se_reduce_channels)
    );
    
    // FIXED: Pipeline synchronization logic
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            overflow_counter <= 16'h0000;
            overflow_flag <= 1'b0;
            overflow_stage <= 8'h00;
            current_pipeline_stage <= 2'b00;
        end else begin
            // Reset overflow flag each cycle
            overflow_flag <= 1'b0;
            
            // Monitor pipeline progression
            if (valid_pipe[15]) current_pipeline_stage <= 2'b11; // Final stage
            else if (valid_pipe[13]) current_pipeline_stage <= 2'b10; // Mid pipeline
            else if (valid_pipe[1]) current_pipeline_stage <= 2'b01; // Early pipeline
            else current_pipeline_stage <= 2'b00; // Idle/input
            
            // Enhanced overflow detection connected to actual signals
            // Check conv1 outputs
            for (int h = 0; h < 112; h++) begin
                for (int w = 0; w < 112; w++) begin
                    for (int c = 0; c < 16; c++) begin
                        if (conv1_act_out[h][w][c] == (2**(DATA_WIDTH-1) - 1) || 
                            conv1_act_out[h][w][c] == -(2**(DATA_WIDTH-1))) begin
                            overflow_flag <= 1'b1;
                            overflow_stage <= 8'h00; // Conv1 stage
                        end
                    end
                end
            end
            
            // Check final linear layer outputs for saturation
            for (int i = 0; i < NUM_CLASSES; i++) begin
                if (data_out[i] == (2**(DATA_WIDTH-1) - 1) || 
                    data_out[i] == -(2**(DATA_WIDTH-1))) begin
                    overflow_flag <= 1'b1;
                    overflow_stage <= 8'h0F; // Final linear layer
                end
            end
            
            // Increment overflow counter when overflow is detected
            if (overflow_flag) begin
                overflow_counter <= overflow_counter + 1;
            end
        end
    end
    
    // Output assignments for overflow detection
    assign overflow_detected = overflow_flag;
    assign overflow_count = overflow_counter;

    // Initial convolution (3x3, stride=2, 1->16)
    Conv2D #(
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(16),
        .KERNEL_SIZE(3),
        .STRIDE(2),
        .PADDING(1),
        .IN_HEIGHT(IMG_HEIGHT),
        .IN_WIDTH(IMG_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) conv1 (
        .clk(clk),
        .rst(rst),
        .data_in(data_in),
        .data_out(conv1_out),
        .weights(conv1_weights),
        .bias(conv1_bias),
        .valid_in(valid_in),
        .valid_out(valid_pipe[0]),
        .ready_out(ready_pipe[0]),
        .pipeline_stage(conv1_stage)
    );

    // Batch normalization for conv1
    BatchNorm2d #(
        .NUM_FEATURES(16),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .HEIGHT(112),
        .WIDTH(112)
    ) conv1_bn (
        .clk(clk),
        .rst(rst),
        .data_in(conv1_out),
        .data_out(conv1_bn_out),
        .effective_weight(conv1_bn_w),
        .effective_bias(conv1_bn_b),
        .valid_in(valid_pipe[0]),
        .valid_out(valid_pipe[1]),
        .ready_out(ready_pipe[1]),
        .pipeline_stage()
    );

    // Hard-swish activation for conv1
    genvar i, j, k;
    generate
        for (i = 0; i < 112; i++) begin : conv1_height_loop
            for (j = 0; j < 112; j++) begin : conv1_width_loop
                for (k = 0; k < 16; k++) begin : conv1_channel_loop
                    hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hswish_inst (
                        .clk(clk),
                        .rst(rst),
                        .data_in(conv1_bn_out[i][j][k]),
                        .data_out(conv1_act_out[i][j][k]),
                        .valid_in(valid_pipe[1]),
                        .valid_out() // Not used for individual activations
                    );
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Inverted residual blocks (11 instances)
    // Parameters are fixed to match the Python reference implementation.
    // Weight arrays are indexed per block from the WeightLoader outputs.

    // Block 1: 16 -> 16, stride 2, ReLU, SE
    Block #(
        .KERNEL_SIZE(3), .IN_SIZE(16), .EXPAND_SIZE(16), .OUT_SIZE(16),
        .STRIDE(2), .USE_SE(1), .NONLINEARITY("RELU"),
        .IN_HEIGHT(112), .IN_WIDTH(112),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(4)
    ) block1 (
        .clk(clk), .rst(rst),
        .data_in(conv1_act_out),
        .data_out(block1_out),
        .expand_weights(block_expand_weights[0]),
        .expand_bn_w(block_expand_bn_w[0]),
        .expand_bn_b(block_expand_bn_b[0]),
        .dw_weights(block_dw_weights[0]),
        .dw_bn_w(block_dw_bn_w[0]),
        .dw_bn_b(block_dw_bn_b[0]),
        .pw_weights(block_pw_weights[0]),
        .pw_bn_w(block_pw_bn_w[0]),
        .pw_bn_b(block_pw_bn_b[0]),
        .se_conv1_weights(block_se_conv1_weights[0]),
        .se_conv1_bn_w(block_se_conv1_bn_w[0]),
        .se_conv1_bn_b(block_se_conv1_bn_b[0]),
        .se_conv2_weights(block_se_conv2_weights[0]),
        .se_conv2_bn_w(block_se_conv2_bn_w[0]),
        .se_conv2_bn_b(block_se_conv2_bn_b[0]),
        .shortcut_weights(block_shortcut_weights[0]),
        .shortcut_bn_w(block_shortcut_bn_w[0]),
        .shortcut_bn_b(block_shortcut_bn_b[0])
    );

    // Block 2: 16 -> 24, stride 2, ReLU
    Block #(
        .KERNEL_SIZE(3), .IN_SIZE(16), .EXPAND_SIZE(72), .OUT_SIZE(24),
        .STRIDE(2), .USE_SE(0), .NONLINEARITY("RELU"),
        .IN_HEIGHT(56), .IN_WIDTH(56),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) block2 (
        .clk(clk), .rst(rst),
        .data_in(block1_out),
        .data_out(block2_out),
        .expand_weights(block_expand_weights[1]),
        .expand_bn_w(block_expand_bn_w[1]),
        .expand_bn_b(block_expand_bn_b[1]),
        .dw_weights(block_dw_weights[1]),
        .dw_bn_w(block_dw_bn_w[1]),
        .dw_bn_b(block_dw_bn_b[1]),
        .pw_weights(block_pw_weights[1]),
        .pw_bn_w(block_pw_bn_w[1]),
        .pw_bn_b(block_pw_bn_b[1]),
        .se_conv1_weights(block_se_conv1_weights[1]),
        .se_conv1_bn_w(block_se_conv1_bn_w[1]),
        .se_conv1_bn_b(block_se_conv1_bn_b[1]),
        .se_conv2_weights(block_se_conv2_weights[1]),
        .se_conv2_bn_w(block_se_conv2_bn_w[1]),
        .se_conv2_bn_b(block_se_conv2_bn_b[1]),
        .shortcut_weights(block_shortcut_weights[1]),
        .shortcut_bn_w(block_shortcut_bn_w[1]),
        .shortcut_bn_b(block_shortcut_bn_b[1])
    );

    // Block 3: 24 -> 24, stride 1, ReLU
    Block #(
        .KERNEL_SIZE(3), .IN_SIZE(24), .EXPAND_SIZE(88), .OUT_SIZE(24),
        .STRIDE(1), .USE_SE(0), .NONLINEARITY("RELU"),
        .IN_HEIGHT(28), .IN_WIDTH(28),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) block3 (
        .clk(clk), .rst(rst),
        .data_in(block2_out),
        .data_out(block3_out),
        .expand_weights(block_expand_weights[2]),
        .expand_bn_w(block_expand_bn_w[2]),
        .expand_bn_b(block_expand_bn_b[2]),
        .dw_weights(block_dw_weights[2]),
        .dw_bn_w(block_dw_bn_w[2]),
        .dw_bn_b(block_dw_bn_b[2]),
        .pw_weights(block_pw_weights[2]),
        .pw_bn_w(block_pw_bn_w[2]),
        .pw_bn_b(block_pw_bn_b[2]),
        .se_conv1_weights(block_se_conv1_weights[2]),
        .se_conv1_bn_w(block_se_conv1_bn_w[2]),
        .se_conv1_bn_b(block_se_conv1_bn_b[2]),
        .se_conv2_weights(block_se_conv2_weights[2]),
        .se_conv2_bn_w(block_se_conv2_bn_w[2]),
        .se_conv2_bn_b(block_se_conv2_bn_b[2]),
        .shortcut_weights(block_shortcut_weights[2]),
        .shortcut_bn_w(block_shortcut_bn_w[2]),
        .shortcut_bn_b(block_shortcut_bn_b[2])
    );

    // Block 4: 24 -> 40, stride 2, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(24), .EXPAND_SIZE(96), .OUT_SIZE(40),
        .STRIDE(2), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(28), .IN_WIDTH(28),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(10)
    ) block4 (
        .clk(clk), .rst(rst),
        .data_in(block3_out),
        .data_out(block4_out),
        .expand_weights(block_expand_weights[3]),
        .expand_bn_w(block_expand_bn_w[3]),
        .expand_bn_b(block_expand_bn_b[3]),
        .dw_weights(block_dw_weights[3]),
        .dw_bn_w(block_dw_bn_w[3]),
        .dw_bn_b(block_dw_bn_b[3]),
        .pw_weights(block_pw_weights[3]),
        .pw_bn_w(block_pw_bn_w[3]),
        .pw_bn_b(block_pw_bn_b[3]),
        .se_conv1_weights(block_se_conv1_weights[3]),
        .se_conv1_bn_w(block_se_conv1_bn_w[3]),
        .se_conv1_bn_b(block_se_conv1_bn_b[3]),
        .se_conv2_weights(block_se_conv2_weights[3]),
        .se_conv2_bn_w(block_se_conv2_bn_w[3]),
        .se_conv2_bn_b(block_se_conv2_bn_b[3]),
        .shortcut_weights(block_shortcut_weights[3]),
        .shortcut_bn_w(block_shortcut_bn_w[3]),
        .shortcut_bn_b(block_shortcut_bn_b[3])
    );

    // Block 5: 40 -> 40, stride 1, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(40), .EXPAND_SIZE(240), .OUT_SIZE(40),
        .STRIDE(1), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(14), .IN_WIDTH(14),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(10)
    ) block5 (
        .clk(clk), .rst(rst),
        .data_in(block4_out),
        .data_out(block5_out),
        .expand_weights(block_expand_weights[4]),
        .expand_bn_w(block_expand_bn_w[4]),
        .expand_bn_b(block_expand_bn_b[4]),
        .dw_weights(block_dw_weights[4]),
        .dw_bn_w(block_dw_bn_w[4]),
        .dw_bn_b(block_dw_bn_b[4]),
        .pw_weights(block_pw_weights[4]),
        .pw_bn_w(block_pw_bn_w[4]),
        .pw_bn_b(block_pw_bn_b[4]),
        .se_conv1_weights(block_se_conv1_weights[4]),
        .se_conv1_bn_w(block_se_conv1_bn_w[4]),
        .se_conv1_bn_b(block_se_conv1_bn_b[4]),
        .se_conv2_weights(block_se_conv2_weights[4]),
        .se_conv2_bn_w(block_se_conv2_bn_w[4]),
        .se_conv2_bn_b(block_se_conv2_bn_b[4]),
        .shortcut_weights(block_shortcut_weights[4]),
        .shortcut_bn_w(block_shortcut_bn_w[4]),
        .shortcut_bn_b(block_shortcut_bn_b[4])
    );

    // Block 6: 40 -> 40, stride 1, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(40), .EXPAND_SIZE(240), .OUT_SIZE(40),
        .STRIDE(1), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(14), .IN_WIDTH(14),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(10)
    ) block6 (
        .clk(clk), .rst(rst),
        .data_in(block5_out),
        .data_out(block6_out),
        .expand_weights(block_expand_weights[5]),
        .expand_bn_w(block_expand_bn_w[5]),
        .expand_bn_b(block_expand_bn_b[5]),
        .dw_weights(block_dw_weights[5]),
        .dw_bn_w(block_dw_bn_w[5]),
        .dw_bn_b(block_dw_bn_b[5]),
        .pw_weights(block_pw_weights[5]),
        .pw_bn_w(block_pw_bn_w[5]),
        .pw_bn_b(block_pw_bn_b[5]),
        .se_conv1_weights(block_se_conv1_weights[5]),
        .se_conv1_bn_w(block_se_conv1_bn_w[5]),
        .se_conv1_bn_b(block_se_conv1_bn_b[5]),
        .se_conv2_weights(block_se_conv2_weights[5]),
        .se_conv2_bn_w(block_se_conv2_bn_w[5]),
        .se_conv2_bn_b(block_se_conv2_bn_b[5]),
        .shortcut_weights(block_shortcut_weights[5]),
        .shortcut_bn_w(block_shortcut_bn_w[5]),
        .shortcut_bn_b(block_shortcut_bn_b[5])
    );

    // Block 7: 40 -> 48, stride 1, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(40), .EXPAND_SIZE(120), .OUT_SIZE(48),
        .STRIDE(1), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(14), .IN_WIDTH(14),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(12)
    ) block7 (
        .clk(clk), .rst(rst),
        .data_in(block6_out),
        .data_out(block7_out),
        .expand_weights(block_expand_weights[6]),
        .expand_bn_w(block_expand_bn_w[6]),
        .expand_bn_b(block_expand_bn_b[6]),
        .dw_weights(block_dw_weights[6]),
        .dw_bn_w(block_dw_bn_w[6]),
        .dw_bn_b(block_dw_bn_b[6]),
        .pw_weights(block_pw_weights[6]),
        .pw_bn_w(block_pw_bn_w[6]),
        .pw_bn_b(block_pw_bn_b[6]),
        .se_conv1_weights(block_se_conv1_weights[6]),
        .se_conv1_bn_w(block_se_conv1_bn_w[6]),
        .se_conv1_bn_b(block_se_conv1_bn_b[6]),
        .se_conv2_weights(block_se_conv2_weights[6]),
        .se_conv2_bn_w(block_se_conv2_bn_w[6]),
        .se_conv2_bn_b(block_se_conv2_bn_b[6]),
        .shortcut_weights(block_shortcut_weights[6]),
        .shortcut_bn_w(block_shortcut_bn_w[6]),
        .shortcut_bn_b(block_shortcut_bn_b[6])
    );

    // Block 8: 48 -> 48, stride 1, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(48), .EXPAND_SIZE(144), .OUT_SIZE(48),
        .STRIDE(1), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(14), .IN_WIDTH(14),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(12)
    ) block8 (
        .clk(clk), .rst(rst),
        .data_in(block7_out),
        .data_out(block8_out),
        .expand_weights(block_expand_weights[7]),
        .expand_bn_w(block_expand_bn_w[7]),
        .expand_bn_b(block_expand_bn_b[7]),
        .dw_weights(block_dw_weights[7]),
        .dw_bn_w(block_dw_bn_w[7]),
        .dw_bn_b(block_dw_bn_b[7]),
        .pw_weights(block_pw_weights[7]),
        .pw_bn_w(block_pw_bn_w[7]),
        .pw_bn_b(block_pw_bn_b[7]),
        .se_conv1_weights(block_se_conv1_weights[7]),
        .se_conv1_bn_w(block_se_conv1_bn_w[7]),
        .se_conv1_bn_b(block_se_conv1_bn_b[7]),
        .se_conv2_weights(block_se_conv2_weights[7]),
        .se_conv2_bn_w(block_se_conv2_bn_w[7]),
        .se_conv2_bn_b(block_se_conv2_bn_b[7]),
        .shortcut_weights(block_shortcut_weights[7]),
        .shortcut_bn_w(block_shortcut_bn_w[7]),
        .shortcut_bn_b(block_shortcut_bn_b[7])
    );

    // Block 9: 48 -> 96, stride 2, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(48), .EXPAND_SIZE(288), .OUT_SIZE(96),
        .STRIDE(2), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(14), .IN_WIDTH(14),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(24)
    ) block9 (
        .clk(clk), .rst(rst),
        .data_in(block8_out),
        .data_out(block9_out),
        .expand_weights(block_expand_weights[8]),
        .expand_bn_w(block_expand_bn_w[8]),
        .expand_bn_b(block_expand_bn_b[8]),
        .dw_weights(block_dw_weights[8]),
        .dw_bn_w(block_dw_bn_w[8]),
        .dw_bn_b(block_dw_bn_b[8]),
        .pw_weights(block_pw_weights[8]),
        .pw_bn_w(block_pw_bn_w[8]),
        .pw_bn_b(block_pw_bn_b[8]),
        .se_conv1_weights(block_se_conv1_weights[8]),
        .se_conv1_bn_w(block_se_conv1_bn_w[8]),
        .se_conv1_bn_b(block_se_conv1_bn_b[8]),
        .se_conv2_weights(block_se_conv2_weights[8]),
        .se_conv2_bn_w(block_se_conv2_bn_w[8]),
        .se_conv2_bn_b(block_se_conv2_bn_b[8]),
        .shortcut_weights(block_shortcut_weights[8]),
        .shortcut_bn_w(block_shortcut_bn_w[8]),
        .shortcut_bn_b(block_shortcut_bn_b[8])
    );

    // Block10: 96 -> 96, stride 1, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(96), .EXPAND_SIZE(576), .OUT_SIZE(96),
        .STRIDE(1), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(7), .IN_WIDTH(7),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(24)
    ) block10 (
        .clk(clk), .rst(rst),
        .data_in(block9_out),
        .data_out(block10_out),
        .expand_weights(block_expand_weights[9]),
        .expand_bn_w(block_expand_bn_w[9]),
        .expand_bn_b(block_expand_bn_b[9]),
        .dw_weights(block_dw_weights[9]),
        .dw_bn_w(block_dw_bn_w[9]),
        .dw_bn_b(block_dw_bn_b[9]),
        .pw_weights(block_pw_weights[9]),
        .pw_bn_w(block_pw_bn_w[9]),
        .pw_bn_b(block_pw_bn_b[9]),
        .se_conv1_weights(block_se_conv1_weights[9]),
        .se_conv1_bn_w(block_se_conv1_bn_w[9]),
        .se_conv1_bn_b(block_se_conv1_bn_b[9]),
        .se_conv2_weights(block_se_conv2_weights[9]),
        .se_conv2_bn_w(block_se_conv2_bn_w[9]),
        .se_conv2_bn_b(block_se_conv2_bn_b[9]),
        .shortcut_weights(block_shortcut_weights[9]),
        .shortcut_bn_w(block_shortcut_bn_w[9]),
        .shortcut_bn_b(block_shortcut_bn_b[9])
    );

    // Block11: 96 -> 96, stride 1, hswish, SE
    Block #(
        .KERNEL_SIZE(5), .IN_SIZE(96), .EXPAND_SIZE(576), .OUT_SIZE(96),
        .STRIDE(1), .USE_SE(1), .NONLINEARITY("HSWISH"),
        .IN_HEIGHT(7), .IN_WIDTH(7),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .SE_REDUCE_SIZE(24)
    ) block11 (
        .clk(clk), .rst(rst),
        .data_in(block10_out),
        .data_out(block11_out),
        .expand_weights(block_expand_weights[10]),
        .expand_bn_w(block_expand_bn_w[10]),
        .expand_bn_b(block_expand_bn_b[10]),
        .dw_weights(block_dw_weights[10]),
        .dw_bn_w(block_dw_bn_w[10]),
        .dw_bn_b(block_dw_bn_b[10]),
        .pw_weights(block_pw_weights[10]),
        .pw_bn_w(block_pw_bn_w[10]),
        .pw_bn_b(block_pw_bn_b[10]),
        .se_conv1_weights(block_se_conv1_weights[10]),
        .se_conv1_bn_w(block_se_conv1_bn_w[10]),
        .se_conv1_bn_b(block_se_conv1_bn_b[10]),
        .se_conv2_weights(block_se_conv2_weights[10]),
        .se_conv2_bn_w(block_se_conv2_bn_w[10]),
        .se_conv2_bn_b(block_se_conv2_bn_b[10]),
        .shortcut_weights(block_shortcut_weights[10]),
        .shortcut_bn_w(block_shortcut_bn_w[10]),
        .shortcut_bn_b(block_shortcut_bn_b[10])
    );

    // Propagate valid flag through block sequence
    assign valid_pipe[2] = valid_pipe[1];
    assign valid_pipe[3] = valid_pipe[2];
    assign valid_pipe[4] = valid_pipe[3];
    assign valid_pipe[5] = valid_pipe[4];
    assign valid_pipe[6] = valid_pipe[5];
    assign valid_pipe[7] = valid_pipe[6];
    assign valid_pipe[8] = valid_pipe[7];
    assign valid_pipe[9] = valid_pipe[8];
    assign valid_pipe[10] = valid_pipe[9];
    assign valid_pipe[11] = valid_pipe[10];
    assign valid_pipe[12] = valid_pipe[11];
    
    // Final convolution (1x1, 96->576)
    PointwiseConv2D #(
        .IN_CHANNELS(96),
        .OUT_CHANNELS(576),
        .IN_HEIGHT(7),
        .IN_WIDTH(7),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) conv2 (
        .clk(clk),
        .rst(rst),
        .data_in(block11_out),
        .data_out(conv2_out),
        .weights(conv2_weights),
        .bias(conv2_bias),
        .valid_in(valid_pipe[12]),
        .valid_out(valid_pipe[13]),
        .ready_out(ready_pipe[13]),
        .pipeline_stage(conv2_stage)
    );

    // BatchNorm, activation and pooling after conv2
    BatchNorm2d #(
        .NUM_FEATURES(576), .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .HEIGHT(7), .WIDTH(7)
    ) conv2_bn (
        .clk(clk), .rst(rst),
        .data_in(conv2_out),
        .data_out(conv2_bn_out),
        .effective_weight(conv2_bn_w),
        .effective_bias(conv2_bn_b),
        .valid_in(valid_pipe[13]),
        .valid_out(valid_pipe[14])
    );

    // hswish activation
    genvar h2, w2, c2;
    generate
        for (h2 = 0; h2 < 7; h2++) begin : conv2_h_loop
            for (w2 = 0; w2 < 7; w2++) begin : conv2_w_loop
                for (c2 = 0; c2 < 576; c2++) begin : conv2_c_loop
                    hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) act_inst (
                        .clk(clk), .rst(rst),
                        .data_in(conv2_bn_out[h2][w2][c2]),
                        .data_out(conv2_act_out[h2][w2][c2]),
                        .valid_in(valid_pipe[14]), .valid_out()
                    );
                end
            end
        end
    endgenerate

    // Global average pooling 7x7 -> 1x1
    AvgPool2d #(
        .KERNEL_SIZE(7), .IN_CHANNELS(576), .IN_HEIGHT(7), .IN_WIDTH(7),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) gap (
        .clk(clk), .rst(rst),
        .data_in(conv2_act_out),
        .data_out(pool_out),
        .valid_in(valid_pipe[14]),
        .valid_out(valid_pipe[15]),
        .ready_out(ready_pipe[14]),
        .pipeline_stage(pool_stage)
    );

    // Linear 3 layer 576 -> 1280
    Linear #(
        .IN_FEATURES(576), .OUT_FEATURES(1280),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) linear3 (
        .clk(clk), .rst(rst),
        .data_in(pool_out),
        .data_out(linear3_out),
        .weights(linear3_weights),
        .bias(linear3_bias),
        .valid_in(valid_pipe[15]),
        .valid_out(valid_pipe[16]),
        .ready_out(ready_pipe[15]),
        .pipeline_stage(linear3_stage)
    );

    BatchNorm1d #(
        .NUM_FEATURES(1280), .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) linear3_bn (
        .clk(clk), .rst(rst),
        .data_in(linear3_out),
        .data_out(linear3_bn_out),
        .effective_weight(linear3_bn_w),
        .effective_bias(linear3_bn_b),
        .valid_in(valid_pipe[16]),
        .valid_out(valid_pipe[17])
    );

    generate
        for (genvar fc = 0; fc < 1280; fc++) begin : linear3_act_loop
            hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hsw3 (
                .clk(clk), .rst(rst),
                .data_in(linear3_bn_out[fc]),
                .data_out(linear3_act_out[fc]),
                .valid_in(valid_pipe[17]), .valid_out()
            );
        end
    endgenerate

    // Linear 4 layer 1280 -> NUM_CLASSES
    Linear #(
        .IN_FEATURES(1280), .OUT_FEATURES(NUM_CLASSES),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) linear4 (
        .clk(clk), .rst(rst),
        .data_in(linear3_act_out),
        .data_out(data_out),
        .weights(linear4_weights),
        .bias(linear4_bias),
        .valid_in(valid_pipe[17]),
        .valid_out(valid_pipe[18]),
        .ready_out(ready_pipe[16]),
        .pipeline_stage(linear4_stage)
    );

    assign valid_out = valid_pipe[18];
    assign ready_out = ready_pipe[0];

endmodule



