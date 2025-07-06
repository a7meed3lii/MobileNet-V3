// Squeeze-and-Excite module - REFACTORED: Bias and old BatchNorm parameters removed.

module SeModule #(
    parameter IN_SIZE = 16,
    parameter REDUCTION = 4,
    parameter IN_HEIGHT = 56,
    parameter IN_WIDTH = 56,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4,
    parameter SQUEEZE_SIZE = (IN_SIZE >= REDUCTION) ? (IN_SIZE / REDUCTION) : 1
)(
    //-Input
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:IN_SIZE-1],
    //-Output
    output logic signed [DATA_WIDTH-1:0] data_out [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:IN_SIZE-1],
    
    // REFACTORED: Weights for first 1x1 conv - bias removed
    input  wire signed [DATA_WIDTH-1:0] conv1_weights [0:SQUEEZE_SIZE-1][0:IN_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] conv1_bn_w [0:SQUEEZE_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] conv1_bn_b [0:SQUEEZE_SIZE-1],
    
    // REFACTORED: Weights for second 1x1 conv - bias removed
    input  wire signed [DATA_WIDTH-1:0] conv2_weights [0:IN_SIZE-1][0:SQUEEZE_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] conv2_bn_w [0:IN_SIZE-1],
    input  wire signed [DATA_WIDTH-1:0] conv2_bn_b [0:IN_SIZE-1],
    
    // Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);
    
    // Intermediate signals
    logic signed [DATA_WIDTH-1:0] pool_out [0:IN_SIZE-1];
    logic signed [DATA_WIDTH-1:0] conv1_out [0:SQUEEZE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] conv1_bn_out [0:SQUEEZE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] conv1_act_out [0:SQUEEZE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] conv2_out [0:IN_SIZE-1];
    logic signed [DATA_WIDTH-1:0] conv2_bn_out [0:IN_SIZE-1];
    logic signed [DATA_WIDTH-1:0] scale_factor [0:IN_SIZE-1];
    
    // Pipeline control
    logic [6:0] valid_pipe; // 7-stage pipeline
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_pipe <= 7'b0;
        end else begin
            valid_pipe <= {valid_pipe[5:0], valid_in};
        end
    end
    
    assign valid_out = valid_pipe[6];

    // 1. Global Average Pooling
    AvgPool2d #(
        .KERNEL_SIZE(IN_HEIGHT),
        .IN_CHANNELS(IN_SIZE),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) pool (
        .clk(clk),
        .rst(rst),
        .data_in(data_in),
        .data_out(pool_out),
        .valid_in(valid_in),
        .valid_out(valid_pipe[0])
    );

    // 2. First 1x1 convolution
    Linear #(
        .IN_FEATURES(IN_SIZE),
        .OUT_FEATURES(SQUEEZE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) conv1 (
        .clk(clk),
        .rst(rst),
        .data_in(pool_out),
        .data_out(conv1_out),
        .weights(conv1_weights),
        .valid_in(valid_pipe[0]),
        .valid_out(valid_pipe[1])
    );
    
    // Batch normalization
    BatchNorm1d #(
        .NUM_FEATURES(SQUEEZE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) bn1 (
        .clk(clk),
        .rst(rst),
        .data_in(conv1_out),
        .data_out(conv1_bn_out),
        .effective_weight(conv1_bn_w),
        .effective_bias(conv1_bn_b),
        .valid_in(valid_pipe[1]),
        .valid_out(valid_pipe[2])
    );
    
    // ReLU activation
    generate
        for (genvar i = 0; i < SQUEEZE_SIZE; i++) begin
            relu #(.DATA_WIDTH(DATA_WIDTH)) relu_inst (
                .clk(clk), .rst(rst), .data_in(conv1_bn_out[i]), .data_out(conv1_act_out[i]), .valid_in(valid_pipe[2]), .valid_out()
            );
        end
    endgenerate
    
    // 3. Second 1x1 convolution
    Linear #(
        .IN_FEATURES(SQUEEZE_SIZE),
        .OUT_FEATURES(IN_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) conv2 (
        .clk(clk),
        .rst(rst),
        .data_in(conv1_act_out),
        .data_out(conv2_out),
        .weights(conv2_weights),
        .valid_in(valid_pipe[3]),
        .valid_out(valid_pipe[4])
    );
    
    // Batch normalization
    BatchNorm1d #(
        .NUM_FEATURES(IN_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) bn2 (
        .clk(clk),
        .rst(rst),
        .data_in(conv2_out),
        .data_out(conv2_bn_out),
        .effective_weight(conv2_bn_w),
        .effective_bias(conv2_bn_b),
        .valid_in(valid_pipe[4]),
        .valid_out(valid_pipe[5])
    );
    
    // h-sigmoid activation
    generate
        for (genvar i = 0; i < IN_SIZE; i++) begin
            hsigmoid #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hsigmoid_inst (
                .clk(clk), .rst(rst), .data_in(conv2_bn_out[i]), .data_out(scale_factor[i]), .valid_in(valid_pipe[5]), .valid_out()
            );
        end
    endgenerate
    
    // 4. Scale original input
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int h = 0; h < IN_HEIGHT; h++) begin
                for (int w = 0; w < IN_WIDTH; w++) begin
                    for (int c = 0; c < IN_SIZE; c++) begin
                        data_out[h][w][c] <= 0;
                    end
                end
            end
        end else if (valid_pipe[6]) begin // Registering final output
            for (int h = 0; h < IN_HEIGHT; h++) begin
                for (int w = 0; w < IN_WIDTH; w++) begin
                    for (int c = 0; c < IN_SIZE; c++) begin
                        logic signed [DATA_WIDTH*2-1:0] scaled_val;
                        scaled_val = data_in[h][w][c] * scale_factor[c];
                        data_out[h][w][c] <= scaled_val >>> FRAC_BITS;
                    end
                end
            end
        end
    end

endmodule
