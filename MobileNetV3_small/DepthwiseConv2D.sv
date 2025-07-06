// Depthwise 2D Convolution module - REFACTORED: Bias term removed to match software model (bias=False).

module DepthwiseConv2D #(
    parameter CHANNELS = 1,
    parameter KERNEL_SIZE = 3,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    parameter IN_HEIGHT = 112,
    parameter IN_WIDTH = 112,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    //-Input
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:CHANNELS-1],
    //-Output  
    output logic signed [DATA_WIDTH-1:0] data_out [0:(IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE][0:(IN_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE][0:CHANNELS-1],
    // Weights (bias removed)
    input  wire signed [DATA_WIDTH-1:0] weights [0:CHANNELS-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    
    // Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);

    // Calculate output dimensions
    localparam OUT_HEIGHT = (IN_HEIGHT - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;
    localparam OUT_WIDTH = (IN_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // Enhanced accumulator width calculation
    localparam MULT_WIDTH = DATA_WIDTH * 2;
    localparam ACC_WIDTH = MULT_WIDTH + $clog2(KERNEL_SIZE * KERNEL_SIZE) + 6;

    localparam PADDED_HEIGHT = IN_HEIGHT + 2 * PADDING;
    localparam PADDED_WIDTH = IN_WIDTH + 2 * PADDING;
    
    // Intermediate padded input
    logic signed [DATA_WIDTH-1:0] padded_data_in [0:PADDED_HEIGHT-1][0:PADDED_WIDTH-1][0:CHANNELS-1];
    
    // Pipelined implementation registers
    logic signed [ACC_WIDTH-1:0] accumulator [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:CHANNELS-1];
    
    // Pipeline control
    logic [1:0] valid_pipe; // 2-stage pipeline
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_pipe <= 2'b00;
        end else begin
            valid_pipe <= {valid_pipe[0], valid_in};
        end
    end
    
    assign valid_out = valid_pipe[1];

    // Padding logic (combinational)
    always_comb begin
        for (int h = 0; h < PADDED_HEIGHT; h++) begin
            for (int w = 0; w < PADDED_WIDTH; w++) begin
                for (int c = 0; c < CHANNELS; c++) begin
                    if (h < PADDING || h >= PADDED_HEIGHT - PADDING || w < PADDING || w >= PADDED_WIDTH - PADDING) begin
                        padded_data_in[h][w][c] = 0;
                    end else begin
                        padded_data_in[h][w][c] = data_in[h - PADDING][w - PADDING][c];
                    end
                end
            end
        end
    end

    // Pipelined depthwise convolution implementation (no bias)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int h = 0; h < OUT_HEIGHT; h++) begin
                for (int w = 0; w < OUT_WIDTH; w++) begin
                    for (int c = 0; c < CHANNELS; c++) begin
                        accumulator[h][w][c] <= 0;
                        data_out[h][w][c] <= 0;
                    end
                end
            end
        end else begin
            // Stage 0: Convolution computation
            if (valid_in) begin
                for (int c = 0; c < CHANNELS; c++) begin
                    for (int h = 0; h < OUT_HEIGHT; h++) begin
                        for (int w = 0; w < OUT_WIDTH; w++) begin
                            logic signed [ACC_WIDTH-1:0] sum = 0;
                            for (int kh = 0; kh < KERNEL_SIZE; kh++) begin
                                for (int kw = 0; kw < KERNEL_SIZE; kw++) begin
                                    sum += padded_data_in[h*STRIDE+kh][w*STRIDE+kw][c] * weights[c][kh][kw];
                                end
                            end
                            accumulator[h][w][c] <= sum;
                        end
                    end
                end
            end
            
            // Stage 1: Saturation and output
            if (valid_pipe[0]) begin
                for (int h = 0; h < OUT_HEIGHT; h++) begin
                    for (int w = 0; w < OUT_WIDTH; w++) begin
                        for (int c = 0; c < CHANNELS; c++) begin
                            logic signed [ACC_WIDTH-1:0] result = accumulator[h][w][c];
                            // Right-shift to align fractional bits before saturation
                            logic signed [ACC_WIDTH-FRAC_BITS:0] shifted_result = result >>> FRAC_BITS;

                            if (shifted_result > (2**(DATA_WIDTH-1) - 1)) begin
                                data_out[h][w][c] <= 2**(DATA_WIDTH-1) - 1;
                            end else if (shifted_result < -(2**(DATA_WIDTH-1))) begin
                                data_out[h][w][c] <= -(2**(DATA_WIDTH-1));
                            end else begin
                                data_out[h][w][c] <= shifted_result[DATA_WIDTH-1:0];
                            end
                        end
                    end
                end
            end
        end
    end
endmodule
