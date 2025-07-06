// Fully connected layer module - FIXED: Standardized 4-stage pipeline with proper valid signals

module Linear #(
    parameter IN_FEATURES = 576,
    parameter OUT_FEATURES = 1280,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    //-Input
    input  logic clk,
    input  logic rst,
    input  wire signed [DATA_WIDTH-1:0] data_in [0:IN_FEATURES-1],
    //-Output
    output logic signed [DATA_WIDTH-1:0] data_out [0:OUT_FEATURES-1],
    // Weights and biases
    input  wire signed [DATA_WIDTH-1:0] weights [0:OUT_FEATURES-1][0:IN_FEATURES-1],
    input  wire signed [DATA_WIDTH-1:0] bias [0:OUT_FEATURES-1],
    
    // FIXED: Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out,
    output logic ready_out,
    output logic [1:0] pipeline_stage
);

    // This module performs a fully connected (linear) operation. It takes a
    // vector as input and produces a vector as output.

    // FIXED: Enhanced accumulator width calculation for large feature dimensions with better safety margin
    localparam MULT_WIDTH = DATA_WIDTH * 2;
    localparam ACC_WIDTH = MULT_WIDTH + $clog2(IN_FEATURES) + 6; // +6 for extra safety

    // FIXED: Standardized 4-stage pipeline registers
    logic signed [DATA_WIDTH-1:0] data_in_reg [0:IN_FEATURES-1];
    logic signed [ACC_WIDTH-1:0] accumulator [0:OUT_FEATURES-1];
    logic signed [ACC_WIDTH-1:0] biased_accumulator [0:OUT_FEATURES-1];
    logic signed [DATA_WIDTH-1:0] saturated_out [0:OUT_FEATURES-1];
    
    // Pipeline control
    logic [3:0] valid_pipe; // 4-stage pipeline: [3:0] = [stage3, stage2, stage1, stage0]
    logic [3:0] ready_pipe;
    
    // Pipeline stage tracking
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_pipe <= 4'b0000;
            ready_pipe <= 4'b1111; // Ready to accept new data
            pipeline_stage <= 2'b00;
        end else begin
            // Shift valid signals through pipeline
            valid_pipe <= {valid_pipe[2:0], valid_in};
            
            // Update pipeline stage indicator
            if (valid_pipe[3]) pipeline_stage <= 2'b11; // Stage 3 (output)
            else if (valid_pipe[2]) pipeline_stage <= 2'b10; // Stage 2 (saturation)
            else if (valid_pipe[1]) pipeline_stage <= 2'b01; // Stage 1 (bias)
            else if (valid_pipe[0]) pipeline_stage <= 2'b00; // Stage 0 (computation)
            else pipeline_stage <= 2'b00; // Idle
        end
    end
    
    // Output control signals
    assign valid_out = valid_pipe[3];
    assign ready_out = ready_pipe[0];

    // FIXED: 4-stage pipeline linear transformation with improved precision and overflow protection
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            // Reset all pipeline stages
            for (int i = 0; i < IN_FEATURES; i++) begin
                data_in_reg[i] <= 0;
            end
            for (int i = 0; i < OUT_FEATURES; i++) begin
                accumulator[i] <= 0;
                biased_accumulator[i] <= 0;
                saturated_out[i] <= 0;
                data_out[i] <= 0;
            end
        end else begin
            
            // Stage 0: Input registration and linear computation
            if (valid_in) begin
                // Register inputs
                for (int i = 0; i < IN_FEATURES; i++) begin
                    data_in_reg[i] <= data_in[i];
                end
                
                // Linear transformation computation
                for (int i = 0; i < OUT_FEATURES; i++) begin
                    logic signed [ACC_WIDTH-1:0] sum;
                    sum = 0;
                    
                    for (int j = 0; j < IN_FEATURES; j++) begin
                        // FIXED: Proper multiplication with extended precision
                        logic signed [MULT_WIDTH-1:0] mult_result;
                        mult_result = data_in[j] * weights[i][j];
                        sum += mult_result;
                    end
                    
                    accumulator[i] <= sum;
                end
            end
            
            // Stage 1: Bias addition
            if (valid_pipe[0]) begin
                for (int i = 0; i < OUT_FEATURES; i++) begin
                    // FIXED: Proper bias addition with extended precision
                    logic signed [ACC_WIDTH-1:0] bias_extended;
                    bias_extended = {{(ACC_WIDTH-DATA_WIDTH){bias[i][DATA_WIDTH-1]}}, bias[i]};
                    biased_accumulator[i] <= accumulator[i] + bias_extended;
                end
            end
            
            // Stage 2: Scaling and saturation
            if (valid_pipe[1]) begin
                for (int i = 0; i < OUT_FEATURES; i++) begin
                    // Apply FRAC_BITS shift before saturation
                    logic signed [ACC_WIDTH-1:0] result;
                    logic signed [ACC_WIDTH-FRAC_BITS:0] scaled_result;
                    result = biased_accumulator[i];
                    scaled_result = result >>> FRAC_BITS;

                    if (scaled_result > (2**(DATA_WIDTH-1) - 1)) begin
                        saturated_out[i] <= 2**(DATA_WIDTH-1) - 1;
                    end else if (scaled_result < -(2**(DATA_WIDTH-1))) begin
                        saturated_out[i] <= -(2**(DATA_WIDTH-1));
                    end else begin
                        saturated_out[i] <= scaled_result[DATA_WIDTH-1:0];
                    end
                end
            end
            
            // Stage 3: Final output registration
            if (valid_pipe[2]) begin
                for (int i = 0; i < OUT_FEATURES; i++) begin
                    data_out[i] <= saturated_out[i];
                end
            end
        end
    end

endmodule

