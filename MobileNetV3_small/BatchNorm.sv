// Batch Normalization modules for MobileNetV3
// REFACTORED: Fused BatchNorm for inference.
// This version uses pre-calculated weights and biases to replace expensive
// on-the-fly division and square root operations with a single multiply-add.
// The pre-calculation is:
// effective_weight = gamma / sqrt(running_var + epsilon)
// effective_bias = beta - (running_mean * effective_weight)

module BatchNorm2d #(
    parameter NUM_FEATURES = 16,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4,
    parameter HEIGHT = 32,
    parameter WIDTH = 32
)(
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:HEIGHT-1][0:WIDTH-1][0:NUM_FEATURES-1],
    output logic signed [DATA_WIDTH-1:0] data_out [0:HEIGHT-1][0:WIDTH-1][0:NUM_FEATURES-1],
    
    // REFACTORED: Use pre-calculated effective weight and bias for inference
    input  wire signed [DATA_WIDTH-1:0] effective_weight [0:NUM_FEATURES-1],
    input  wire signed [DATA_WIDTH-1:0] effective_bias [0:NUM_FEATURES-1],
    
    // Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);

    // Internal pipeline register for valid signal
    logic valid_reg;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_reg <= 1'b0;
        end else begin
            valid_reg <= valid_in;
        end
    end

    assign valid_out = valid_reg;

    // Combinational logic for fused batch normalization
    // Operation: y = x * effective_weight + effective_bias
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int h = 0; h < HEIGHT; h++) begin
                for (int w = 0; w < WIDTH; w++) begin
                    for (int c = 0; c < NUM_FEATURES; c++) begin
                        data_out[h][w][c] <= 0;
                    end
                end
            end
        end else if (valid_in) begin
            for (int h = 0; h < HEIGHT; h++) begin
                for (int w = 0; w < WIDTH; w++) begin
                    for (int c = 0; c < NUM_FEATURES; c++) begin
                        // High-precision intermediate calculation
                        automatic logic signed [DATA_WIDTH*2-1:0] mult_res;
                        automatic logic signed [DATA_WIDTH*2:0] add_res;

                        // Multiplication: data_in * effective_weight
                        mult_res = data_in[h][w][c] * effective_weight[c];
                        
                        // Addition: result + effective_bias
                        // Shift bias to align fractional points before adding
                        add_res = (mult_res >>> FRAC_BITS) + effective_bias[c];

                        // Saturation to DATA_WIDTH
                        if (add_res > (2**(DATA_WIDTH-1) - 1)) begin
                            data_out[h][w][c] <= 2**(DATA_WIDTH-1) - 1;
                        end else if (add_res < -(2**(DATA_WIDTH-1))) begin
                            data_out[h][w][c] <= -(2**(DATA_WIDTH-1));
                        end else begin
                            data_out[h][w][c] <= add_res[DATA_WIDTH-1:0];
                        end
                    end
                end
            end
        end
    end

endmodule

module BatchNorm1d #(
    parameter NUM_FEATURES = 1280,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:NUM_FEATURES-1],
    output logic signed [DATA_WIDTH-1:0] data_out [0:NUM_FEATURES-1],

    // REFACTORED: Use pre-calculated effective weight and bias for inference
    input  wire signed [DATA_WIDTH-1:0] effective_weight [0:NUM_FEATURES-1],
    input  wire signed [DATA_WIDTH-1:0] effective_bias [0:NUM_FEATURES-1],

    // Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);

    // Internal pipeline register for valid signal
    logic valid_reg;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_reg <= 1'b0;
        end else begin
            valid_reg <= valid_in;
        end
    end

    assign valid_out = valid_reg;

    // Combinational logic for fused batch normalization (1D)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < NUM_FEATURES; i++) begin
                data_out[i] <= 0;
            end
        end else if (valid_in) begin
            for (int i = 0; i < NUM_FEATURES; i++) begin
                // High-precision intermediate calculation
                automatic logic signed [DATA_WIDTH*2-1:0] mult_res;
                automatic logic signed [DATA_WIDTH*2:0] add_res;

                // Multiplication: data_in * effective_weight
                mult_res = data_in[i] * effective_weight[i];

                // Addition: result + effective_bias
                add_res = (mult_res >>> FRAC_BITS) + effective_bias[i];

                // Saturation to DATA_WIDTH
                if (add_res > (2**(DATA_WIDTH-1) - 1)) begin
                    data_out[i] <= 2**(DATA_WIDTH-1) - 1;
                end else if (add_res < -(2**(DATA_WIDTH-1))) begin
                    data_out[i] <= -(2**(DATA_WIDTH-1));
                end else begin
                    data_out[i] <= add_res[DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule 
