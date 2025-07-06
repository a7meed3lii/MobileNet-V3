// 2D Convolution module - REFACTORED: Bias term removed to match software model (bias=False).

module Conv2D #(
    parameter IN_CHANNELS = 1,
    parameter OUT_CHANNELS = 1,
    parameter KERNEL_SIZE = 3,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    parameter IN_HEIGHT = 224,
    parameter IN_WIDTH = 224,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4,
    // Derived parameters declared here so they are available for port sizing
    parameter OUT_HEIGHT = (IN_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1,
    parameter OUT_WIDTH  = (IN_WIDTH  + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1
)(
    
    //-Input
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:IN_HEIGHT-1][0:IN_WIDTH-1][0:IN_CHANNELS-1],
    //-Output
    output logic signed [DATA_WIDTH-1:0] data_out [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_CHANNELS-1],
    // Weights (bias removed)
    input  wire signed [DATA_WIDTH-1:0] weights [0:OUT_CHANNELS-1][0:IN_CHANNELS-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1],
    
    // Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);

    // Calculate output dimensions (already defined as parameters)

    // Enhanced accumulator width calculation
    localparam MULT_WIDTH = DATA_WIDTH * 2;
    localparam ACC_WIDTH = MULT_WIDTH + $clog2(KERNEL_SIZE * KERNEL_SIZE * IN_CHANNELS) + 6;

    // Pipelined implementation registers
    logic signed [ACC_WIDTH-1:0] accumulator [0:OUT_HEIGHT-1][0:OUT_WIDTH-1][0:OUT_CHANNELS-1];
    
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

    // 2-stage pipeline for convolution (no bias)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int oy = 0; oy < OUT_HEIGHT; oy++) begin
                for (int ox = 0; ox < OUT_WIDTH; ox++) begin
                    for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                        accumulator[oy][ox][oc] <= 0;
                        data_out[oy][ox][oc] <= 0;
                    end
                end
            end
        end else begin
            
            // Stage 0: Convolution computation
            if (valid_in) begin
                for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                    for (int oy = 0; oy < OUT_HEIGHT; oy++) begin
                        for (int ox = 0; ox < OUT_WIDTH; ox++) begin
                            logic signed [ACC_WIDTH-1:0] sum = 0;
                            for (int ic = 0; ic < IN_CHANNELS; ic++) begin
                                for (int ky = 0; ky < KERNEL_SIZE; ky++) begin
                                    for (int kx = 0; kx < KERNEL_SIZE; kx++) begin
                                        int ix, iy;
                                        ix = ox * STRIDE + kx - PADDING;
                                        iy = oy * STRIDE + ky - PADDING;
                                        
                                        if (ix >= 0 && ix < IN_WIDTH && iy >= 0 && iy < IN_HEIGHT) begin
                                            logic signed [MULT_WIDTH-1:0] mult_result;
                                            mult_result = data_in[iy][ix][ic] * weights[oc][ic][ky][kx];
                                            sum += mult_result;
                                        end
                                    end
                                end
                            end
                            accumulator[oy][ox][oc] <= sum;
                        end
                    end
                end
            end
            
            // Stage 1: Saturation and output
            if (valid_pipe[0]) begin
                for (int oy = 0; oy < OUT_HEIGHT; oy++) begin
                    for (int ox = 0; ox < OUT_WIDTH; ox++) begin
                        for (int oc = 0; oc < OUT_CHANNELS; oc++) begin
                            logic signed [ACC_WIDTH-1:0] result = accumulator[oy][ox][oc];
                            // Right-shift to align fractional bits before saturation
                            logic signed [ACC_WIDTH-FRAC_BITS:0] shifted_result = result >>> FRAC_BITS;

                            if (shifted_result > (2**(DATA_WIDTH-1) - 1)) begin
                                data_out[oy][ox][oc] <= 2**(DATA_WIDTH-1) - 1;
                            end else if (shifted_result < -(2**(DATA_WIDTH-1))) begin
                                data_out[oy][ox][oc] <= -(2**(DATA_WIDTH-1));
                            end else begin
                                data_out[oy][ox][oc] <= shifted_result[DATA_WIDTH-1:0];
                            end
                        end
                    end
                end
            end
        end
    end

endmodule

