// Block module for MobileNetV3 - REFACTORED: Bias and old BatchNorm parameters removed.

module Block #(
    parameter KERNEL_SIZE = 3,
    parameter IN_SIZE = 16,
    parameter EXPAND_SIZE = 16,
    parameter OUT_SIZE = 16,
    parameter STRIDE = 1,
    parameter logic USE_SE = 1,
    parameter string NONLINEARITY = "RELU",
    parameter IN_HEIGHT = 112,
    parameter IN_WIDTH = 112,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4,
    parameter SE_REDUCE_SIZE = 4
)(
    //-Input
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:IN_SIZE-1],
    //-Output
    output logic signed [DATA_WIDTH-1:0] data_out [0:IN_HEIGHT / STRIDE - 1][0:IN_WIDTH / STRIDE - 1][0:OUT_SIZE-1],
    
    // REFACTORED: Weights for expand conv (1x1) - bias removed
    input  wire signed [DATA_WIDTH-1:0] expand_weights [0:EXPAND_SIZE-1][0:IN_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] expand_bn_w [0:EXPAND_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] expand_bn_b [0:EXPAND_SIZE-1],
    
    // REFACTORED: Weights for depthwise conv - bias removed
    input  wire signed [DATA_WIDTH-1:0] dw_weights [0:EXPAND_SIZE-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] dw_bn_w [0:EXPAND_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] dw_bn_b [0:EXPAND_SIZE-1],
    
    // REFACTORED: Weights for pointwise conv (1x1) - bias removed
    input  wire signed [DATA_WIDTH-1:0] pw_weights [0:OUT_SIZE-1][0:EXPAND_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] pw_bn_w [0:OUT_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] pw_bn_b [0:OUT_SIZE-1],
    
    // REFACTORED: SE module weights - bias removed
    input  wire signed [DATA_WIDTH-1:0] se_conv1_weights [0:SE_REDUCE_SIZE-1][0:OUT_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] se_conv1_bn_w [0:SE_REDUCE_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] se_conv1_bn_b [0:SE_REDUCE_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] se_conv2_weights [0:OUT_SIZE-1][0:SE_REDUCE_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] se_conv2_bn_w [0:OUT_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] se_conv2_bn_b [0:OUT_SIZE-1],
    
    // REFACTORED: Shortcut conv weights - bias removed
    input  wire signed [DATA_WIDTH-1:0] shortcut_weights [0:OUT_SIZE-1][0:IN_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] shortcut_bn_w [0:OUT_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] shortcut_bn_b [0:OUT_SIZE-1]
);

    // Calculate output dimensions
    localparam OUT_HEIGHT = IN_HEIGHT / STRIDE;
    localparam OUT_WIDTH = IN_WIDTH / STRIDE;
    
    // Intermediate feature maps
    logic signed [DATA_WIDTH-1:0] expand_out [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:EXPAND_SIZE-1];
    logic signed [DATA_WIDTH-1:0] expand_bn_out [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:EXPAND_SIZE-1];
    logic signed [DATA_WIDTH-1:0] expand_act_out [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:EXPAND_SIZE-1];
    
    logic signed [DATA_WIDTH-1:0] dw_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:EXPAND_SIZE-1];
    logic signed [DATA_WIDTH-1:0] dw_bn_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:EXPAND_SIZE-1];
    logic signed [DATA_WIDTH-1:0] dw_act_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:EXPAND_SIZE-1];
    
    logic signed [DATA_WIDTH-1:0] pw_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_SIZE-1];
    logic signed [DATA_WIDTH-1:0] pw_bn_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_SIZE-1];
    
    logic signed [DATA_WIDTH-1:0] se_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_SIZE-1];
    logic signed [DATA_WIDTH-1:0] shortcut_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_SIZE-1];

    // 1. Expand convolution (1x1)
    PointwiseConv2D #(
        .IN_CHANNELS(IN_SIZE),
        .OUT_CHANNELS(EXPAND_SIZE),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) expand_conv (
        .clk(clk),
        .rst(rst),
        .data_in(data_in),
        .data_out(expand_out),
        .weights(expand_weights)
    );
    
    // Batch normalization for expand conv
    BatchNorm2d #(
        .NUM_FEATURES(EXPAND_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .HEIGHT(IN_HEIGHT),
        .WIDTH(IN_WIDTH)
    ) expand_bn (
        .clk(clk),
        .rst(rst),
        .data_in(expand_out),
        .data_out(expand_bn_out),
        .effective_weight(expand_bn_w),
        .effective_bias(expand_bn_b)
    );
    
    // Activation function for expand conv
    generate
    if (NONLINEARITY == "RELU") begin : expand_relu
        genvar i, j, k;
        for (i = 0; i < IN_HEIGHT; i++) begin : height_loop
            for (j = 0; j < IN_WIDTH; j++) begin : width_loop
                for (k = 0; k < EXPAND_SIZE; k++) begin : channel_loop
                    relu #(.DATA_WIDTH(DATA_WIDTH)) relu_inst (
                        .clk(clk),
                        .rst(rst),
                        .data_in(expand_bn_out[i][j][k]),
                        .data_out(expand_act_out[i][j][k])
                    );
                end
            end
        end
    end else if (NONLINEARITY == "HSWISH") begin : expand_hswish
        genvar i, j, k;
        for (i = 0; i < IN_HEIGHT; i++) begin : height_loop
            for (j = 0; j < IN_WIDTH; j++) begin : width_loop
                for (k = 0; k < EXPAND_SIZE; k++) begin : channel_loop
                    hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hswish_inst (
                        .clk(clk),
                        .rst(rst),
                        .data_in(expand_bn_out[i][j][k]),
                        .data_out(expand_act_out[i][j][k])
                    );
                end
            end
        end
    end
    endgenerate

    // 2. Depthwise convolution
    DepthwiseConv2D #(
        .CHANNELS(EXPAND_SIZE),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .PADDING(KERNEL_SIZE/2),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) depthwise_conv (
        .clk(clk),
        .rst(rst),
        .data_in(expand_act_out),
        .data_out(dw_out),
        .weights(dw_weights)
    );

    // Batch normalization for depthwise conv
    BatchNorm2d #(
        .NUM_FEATURES(EXPAND_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .HEIGHT(OUT_HEIGHT),
        .WIDTH(OUT_WIDTH)
    ) dw_bn (
        .clk(clk),
        .rst(rst),
        .data_in(dw_out),
        .data_out(dw_bn_out),
        .effective_weight(dw_bn_w),
        .effective_bias(dw_bn_b)
    );

    // Activation function for depthwise conv
    generate
        if (NONLINEARITY == "RELU") begin : dw_relu
            genvar i, j, k;
            for (i = 0; i < OUT_HEIGHT; i++) begin : height_loop
                for (j = 0; j < OUT_WIDTH; j++) begin : width_loop
                    for (k = 0; k < EXPAND_SIZE; k++) begin : channel_loop
                        relu #(.DATA_WIDTH(DATA_WIDTH)) relu_inst (
                            .clk(clk),
                            .rst(rst),
                            .data_in(dw_bn_out[i][j][k]),
                            .data_out(dw_act_out[i][j][k])
                        );
                    end
                end
            end
        end else if (NONLINEARITY == "HSWISH") begin : dw_hswish
            genvar i, j, k;
            for (i = 0; i < OUT_HEIGHT; i++) begin : height_loop
                for (j = 0; j < OUT_WIDTH; j++) begin : width_loop
                    for (k = 0; k < EXPAND_SIZE; k++) begin : channel_loop
                        hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hswish_inst (
                            .clk(clk),
                            .rst(rst),
                            .data_in(dw_bn_out[i][j][k]),
                            .data_out(dw_act_out[i][j][k])
                        );
                    end
                end
            end
        end
    endgenerate

    // 3. Pointwise convolution (1x1)
    PointwiseConv2D #(
        .IN_CHANNELS(EXPAND_SIZE),
        .OUT_CHANNELS(OUT_SIZE),
        .IN_HEIGHT(OUT_HEIGHT),
        .IN_WIDTH(OUT_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) pointwise_conv (
        .clk(clk),
        .rst(rst),
        .data_in(dw_act_out),
        .data_out(pw_out),
        .weights(pw_weights)
    );

    // Batch normalization for pointwise conv
    BatchNorm2d #(
        .NUM_FEATURES(OUT_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .HEIGHT(OUT_HEIGHT),
        .WIDTH(OUT_WIDTH)
    ) pw_bn (
        .clk(clk),
        .rst(rst),
        .data_in(pw_out),
        .data_out(pw_bn_out),
        .effective_weight(pw_bn_w),
        .effective_bias(pw_bn_b)
    );

    // 4. Squeeze-and-Excitation module (optional)
    generate
        if (USE_SE) begin : se_block
            SeModule #(
                .IN_SIZE(OUT_SIZE),
                .REDUCTION(4),
                .IN_HEIGHT(OUT_HEIGHT),
                .IN_WIDTH(OUT_WIDTH),
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS),
                .SQUEEZE_SIZE(SE_REDUCE_SIZE)
            ) se_inst (
                .clk(clk),
                .rst(rst),
                .data_in(pw_bn_out),
                .data_out(se_out),
                .conv1_weights(se_conv1_weights),
                .conv1_bn_w(se_conv1_bn_w),
                .conv1_bn_b(se_conv1_bn_b),
                .conv2_weights(se_conv2_weights),
                .conv2_bn_w(se_conv2_bn_w),
                .conv2_bn_b(se_conv2_bn_b)
            );
        end else begin : no_se_block
            assign se_out = pw_bn_out;
        end
    endgenerate

    // 5. Shortcut connection logic
    generate
        if (STRIDE == 1 && IN_SIZE == OUT_SIZE) begin : direct_shortcut
            // Direct connection
            assign shortcut_out = data_in;
        end else if (STRIDE == 1 && IN_SIZE != OUT_SIZE) begin : conv_shortcut
            // 1x1 conv + BN for shortcut
            logic signed [DATA_WIDTH-1:0] shortcut_conv_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_SIZE-1];
            
            PointwiseConv2D #(
                .IN_CHANNELS(IN_SIZE),
                .OUT_CHANNELS(OUT_SIZE),
                .IN_HEIGHT(IN_HEIGHT),
                .IN_WIDTH(IN_WIDTH),
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) shortcut_conv (
                .clk(clk),
                .rst(rst),
                .data_in(data_in),
                .data_out(shortcut_conv_out),
                .weights(shortcut_weights)
            );
            
            BatchNorm2d #(
                .NUM_FEATURES(OUT_SIZE),
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS),
                .HEIGHT(OUT_HEIGHT),
                .WIDTH(OUT_WIDTH)
            ) shortcut_bn (
                .clk(clk),
                .rst(rst),
                .data_in(shortcut_conv_out),
                .data_out(shortcut_out),
                .effective_weight(shortcut_bn_w),
                .effective_bias(shortcut_bn_b)
            );
        end else begin : no_shortcut
            // No shortcut connection
            assign shortcut_out = '0;
        end
    endgenerate

    // 6. Final output with shortcut addition
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= '0;
        end else begin
            if (STRIDE == 1) begin
                for (int h = 0; h < OUT_HEIGHT; h++) begin
                    for (int w = 0; w < OUT_WIDTH; w++) begin
                        for (int c = 0; c < OUT_SIZE; c++) begin
                            data_out[h][w][c] <= se_out[h][w][c] + shortcut_out[h][w][c];
                        end
                    end
                end
            end else begin
                data_out <= se_out;
            end
        end
    end

endmodule
