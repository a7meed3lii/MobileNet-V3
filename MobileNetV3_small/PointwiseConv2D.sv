// Pointwise 2D Convolution module (1x1 convolution) - REFACTORED: Bias term removed.

module PointwiseConv2D #(
    parameter IN_CHANNELS = 1,
    parameter OUT_CHANNELS = 1,
    parameter IN_HEIGHT = 112,
    parameter IN_WIDTH = 112,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    //-Input
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:IN_CHANNELS-1],
    //-Output
    output logic signed [DATA_WIDTH-1:0] data_out [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:OUT_CHANNELS-1],
    // Weights (bias removed)
    input  wire signed [DATA_WIDTH-1:0] weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1],
    
    // Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);

    // Enhanced accumulator width calculation
    localparam MULT_WIDTH = DATA_WIDTH * 2;
    localparam ACC_WIDTH = MULT_WIDTH + $clog2(IN_CHANNELS);

    // Pipelined implementation registers
    logic signed [ACC_WIDTH-1:0] accumulator [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:OUT_CHANNELS-1];
    
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

    // 2-stage pipeline for pointwise convolution (no bias)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int h = 0; h < IN_HEIGHT; h++) begin
                for (int w = 0; w < IN_WIDTH; w++) begin
                    for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                        accumulator[h][w][oc] <= 0;
                        data_out[h][w][oc] <= 0;
                    end
                end
            end
        end else begin
            
            // Stage 0: Pointwise convolution computation
            if (valid_in) begin
                for (int h = 0; h < IN_HEIGHT; h++) begin
                    for (int w = 0; w < IN_WIDTH; w++) begin
                        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                            automatic logic signed [ACC_WIDTH-1:0] sum = 0;
                            for (int ic = 0; ic < IN_CHANNELS; ic++) {
                                sum += data_in[h][w][ic] * weights[oc][ic];
                            }
                            accumulator[h][w][oc] <= sum;
                        end
                    end
                end
            end
            
            // Stage 1: Saturation and output
            if (valid_pipe[0]) begin
                for (int h = 0; h < IN_HEIGHT; h++) begin
                    for (int w = 0; w < IN_WIDTH; w++) begin
                        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                            automatic logic signed [ACC_WIDTH-1:0] result = accumulator[h][w][oc];
                            // Right-shift to align fractional bits before saturation
                            automatic logic signed [ACC_WIDTH-FRAC_BITS:0] shifted_result = result >>> FRAC_BITS;

                            if (shifted_result > (2**(DATA_WIDTH-1) - 1)) begin
                                data_out[h][w][oc] <= 2**(DATA_WIDTH-1) - 1;
                            end else if (shifted_result < -(2**(DATA_WIDTH-1))) begin
                                data_out[h][w][oc] <= -(2**(DATA_WIDTH-1));
                            end else begin
                                data_out[h][w][oc] <= shifted_result[DATA_WIDTH-1:0];
                            end
                        end
                    end
                end
            end
        end
    end

endmodule

