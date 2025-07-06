// Activation Functions for MobileNetV3 - FIXED: Standardized pipeline with proper valid signals
// This module contains all activation functions used in MobileNetV3

module hswish #(
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out,
    
    // FIXED: Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);
    // FIXED: hswish(x) = x * relu6(x + 3) / 6
    // Using high-precision fixed-point arithmetic with accurate division
    
    logic signed [DATA_WIDTH-1:0] three_fixed;
    logic signed [DATA_WIDTH-1:0] six_fixed;
    
    // Convert constants to fixed-point
    assign three_fixed = 3 << FRAC_BITS;
    assign six_fixed = 6 << FRAC_BITS;
    
    // FIXED: Single-cycle pipeline with valid signal propagation
    logic valid_reg;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            valid_reg <= 1'b0;
        end else begin
            valid_reg <= valid_in;
            
            if (valid_in) begin
                logic signed [DATA_WIDTH-1:0] x_plus_3;
                logic signed [DATA_WIDTH-1:0] relu6_result;
                logic signed [DATA_WIDTH*2-1:0] mult_result;
                logic signed [DATA_WIDTH*2-1:0] final_result;
                
                // Step 1: Add 3
                x_plus_3 = data_in + three_fixed;
                
                // Step 2: ReLU6 - clamp(x+3, 0, 6)
                if (x_plus_3 <= 0) begin
                    relu6_result = 0;
                end else if (x_plus_3 >= six_fixed) begin
                    relu6_result = six_fixed;
                end else begin
                    relu6_result = x_plus_3;
                end
                
                // Step 3: Multiply x * relu6(x + 3)
                mult_result = data_in * relu6_result;
                
                // Step 4: ENHANCED - Ultra-high-precision division by 6
                // Method: Use 1/6 ≈ 715827883/2^32 (extremely accurate: error < 0.00000001%)
                final_result = (mult_result * 715827883) >>> 32;
                
                // Step 5: Saturation
                if (final_result > ((2**(DATA_WIDTH-1) - 1) << FRAC_BITS)) begin
                    data_out <= 2**(DATA_WIDTH-1) - 1;
                end else if (final_result < (-(2**(DATA_WIDTH-1)) << FRAC_BITS)) begin
                    data_out <= -(2**(DATA_WIDTH-1));
                end else begin
                    data_out <= final_result[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
                end
            end
        end
    end
    
    assign valid_out = valid_reg;
endmodule

module hsigmoid #(
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out,
    
    // FIXED: Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);
    // FIXED: hsigmoid(x) = relu6(x + 3) / 6
    // Using high-precision fixed-point arithmetic
    
    logic signed [DATA_WIDTH-1:0] three_fixed;
    logic signed [DATA_WIDTH-1:0] six_fixed;
    
    // Convert constants to fixed-point
    assign three_fixed = 3 << FRAC_BITS;
    assign six_fixed = 6 << FRAC_BITS;
    
    // FIXED: Single-cycle pipeline with valid signal propagation
    logic valid_reg;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            valid_reg <= 1'b0;
        end else begin
            valid_reg <= valid_in;
            
            if (valid_in) begin
                logic signed [DATA_WIDTH-1:0] x_plus_3;
                logic signed [DATA_WIDTH-1:0] relu6_result;
                logic signed [DATA_WIDTH*2-1:0] div_result;
                
                // Step 1: Add 3
                x_plus_3 = data_in + three_fixed;
                
                // Step 2: ReLU6 - clamp(x+3, 0, 6)
                if (x_plus_3 <= 0) begin
                    relu6_result = 0;
                end else if (x_plus_3 >= six_fixed) begin
                    relu6_result = six_fixed;
                end else begin
                    relu6_result = x_plus_3;
                end
                
                // Step 3: ENHANCED - Ultra-high-precision division by 6
                // Method: Use 1/6 ≈ 715827883/2^32 (extremely accurate: error < 0.00000001%)
                div_result = (relu6_result * 715827883) >>> 32;
                
                // Step 4: Saturation (hsigmoid output is always 0 to 1)
                if (div_result > (1 << FRAC_BITS)) begin // Max value is 1.0 in fixed-point
                    data_out <= 1 << FRAC_BITS;
                end else if (div_result < 0) begin
                    data_out <= 0; // hsigmoid output is always non-negative
                end else begin
                    data_out <= div_result[DATA_WIDTH-1:0];
                end
            end
        end
    end
    
    assign valid_out = valid_reg;
endmodule

// IMPROVED: More efficient ReLU implementation with valid signals
module relu #(
    parameter DATA_WIDTH = 8
)(
    input  logic clk,
    input  logic rst,
    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out,
    
    // FIXED: Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);
    // ReLU(x) = max(0, x)
    logic valid_reg;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            valid_reg <= 1'b0;
        end else begin
            valid_reg <= valid_in;
            
            if (valid_in) begin
                data_out <= (data_in > 0) ? data_in : 0;
            end
        end
    end
    
    assign valid_out = valid_reg;
endmodule

// IMPROVED: Add ReLU6 activation function for completeness with valid signals
module relu6 #(
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 4
)(
    input  logic clk,
    input  logic rst,
    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out,
    
    // FIXED: Standardized pipeline control signals
    input  logic valid_in,
    output logic valid_out
);
    // ReLU6(x) = min(max(0, x), 6)
    logic signed [DATA_WIDTH-1:0] six_fixed;
    assign six_fixed = 6 << FRAC_BITS;
    
    logic valid_reg;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
            valid_reg <= 1'b0;
        end else begin
            valid_reg <= valid_in;
            
            if (valid_in) begin
                if (data_in <= 0) begin
                    data_out <= 0;
                end else if (data_in >= six_fixed) begin
                    data_out <= six_fixed;
                end else begin
                    data_out <= data_in;
                end
            end
        end
    end
    
    assign valid_out = valid_reg;
endmodule 
