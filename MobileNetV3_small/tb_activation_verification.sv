// Comprehensive Activation Function Verification Testbench
// This testbench verifies hswish and hsigmoid functions against mathematical definitions

`timescale 1ns/1ps

module tb_activation_verification();
    parameter DATA_WIDTH = 8;
    parameter FRAC_BITS = 4;
    parameter CLK_PERIOD = 10;
    
    logic clk, rst;
    logic signed [DATA_WIDTH-1:0] test_input;
    logic signed [DATA_WIDTH-1:0] relu_out, hswish_out, hsigmoid_out;
    
    // Test control
    int test_count;
    int pass_count;
    int fail_count;
    logic all_tests_pass;
    
    // Expected values (calculated offline)
    logic signed [DATA_WIDTH-1:0] expected_hswish;
    logic signed [DATA_WIDTH-1:0] expected_hsigmoid;
    logic signed [DATA_WIDTH-1:0] expected_relu;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Instantiate activation functions
    relu #(.DATA_WIDTH(DATA_WIDTH)) relu_dut (
        .clk(clk), .rst(rst), .data_in(test_input), .data_out(relu_out)
    );
    
    hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hswish_dut (
        .clk(clk), .rst(rst), .data_in(test_input), .data_out(hswish_out)
    );
    
    hsigmoid #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hsigmoid_dut (
        .clk(clk), .rst(rst), .data_in(test_input), .data_out(hsigmoid_out)
    );
    
    // Function to calculate expected hswish value
    function automatic logic signed [DATA_WIDTH-1:0] calc_hswish(input logic signed [DATA_WIDTH-1:0] x);
        automatic real x_real, relu6_result, hswish_result;
        automatic logic signed [DATA_WIDTH-1:0] result;
        
        // Convert fixed-point to real
        x_real = real'(x) / (2.0 ** FRAC_BITS);
        
        // hswish(x) = x * relu6(x + 3) / 6
        // relu6(x) = min(max(x, 0), 6)
        relu6_result = (x_real + 3.0 < 0.0) ? 0.0 : 
                      (x_real + 3.0 > 6.0) ? 6.0 : (x_real + 3.0);
        
        hswish_result = x_real * relu6_result / 6.0;
        
        // Convert back to fixed-point
        result = $signed($rtoi(hswish_result * (2.0 ** FRAC_BITS)));
        
        // Apply saturation
        if (hswish_result > (2.0**(DATA_WIDTH-1) - 1) / (2.0**FRAC_BITS)) begin
            result = 2**(DATA_WIDTH-1) - 1;
        end else if (hswish_result < -(2.0**(DATA_WIDTH-1)) / (2.0**FRAC_BITS)) begin
            result = -(2**(DATA_WIDTH-1));
        end
        
        return result;
    endfunction
    
    // Function to calculate expected hsigmoid value
    function automatic logic signed [DATA_WIDTH-1:0] calc_hsigmoid(input logic signed [DATA_WIDTH-1:0] x);
        automatic real x_real, relu6_result, hsigmoid_result;
        automatic logic signed [DATA_WIDTH-1:0] result;
        
        // Convert fixed-point to real
        x_real = real'(x) / (2.0 ** FRAC_BITS);
        
        // hsigmoid(x) = relu6(x + 3) / 6
        relu6_result = (x_real + 3.0 < 0.0) ? 0.0 : 
                      (x_real + 3.0 > 6.0) ? 6.0 : (x_real + 3.0);
        
        hsigmoid_result = relu6_result / 6.0;
        
        // Convert back to fixed-point
        result = $signed($rtoi(hsigmoid_result * (2.0 ** FRAC_BITS)));
        
        // hsigmoid output is always 0 to 1
        if (hsigmoid_result > 1.0) begin
            result = 1 << FRAC_BITS;
        end else if (hsigmoid_result < 0.0) begin
            result = 0;
        end
        
        return result;
    endfunction
    
    // Function to calculate expected ReLU value
    function automatic logic signed [DATA_WIDTH-1:0] calc_relu(input logic signed [DATA_WIDTH-1:0] x);
        return (x > 0) ? x : 0;
    endfunction
    
    // Task to run a single test
    task run_test(input logic signed [DATA_WIDTH-1:0] input_val, input string test_name);
        test_input = input_val;
        expected_relu = calc_relu(input_val);
        expected_hswish = calc_hswish(input_val);
        expected_hsigmoid = calc_hsigmoid(input_val);
        
        #(CLK_PERIOD * 2); // Wait for computation
        
        test_count++;
        
        // Check ReLU
        if (relu_out == expected_relu) begin
            $display("PASS: %s ReLU(%d) = %d (expected %d)", test_name, input_val, relu_out, expected_relu);
            pass_count++;
        end else begin
            $display("FAIL: %s ReLU(%d) = %d (expected %d)", test_name, input_val, relu_out, expected_relu);
            fail_count++;
        end
        
        // Check hswish (allow ±1 tolerance for fixed-point approximation)
        if ((hswish_out >= expected_hswish - 1) && (hswish_out <= expected_hswish + 1)) begin
            $display("PASS: %s hswish(%d) = %d (expected %d)", test_name, input_val, hswish_out, expected_hswish);
            pass_count++;
        end else begin
            $display("FAIL: %s hswish(%d) = %d (expected %d)", test_name, input_val, hswish_out, expected_hswish);
            fail_count++;
        end
        
        // Check hsigmoid (allow ±1 tolerance for fixed-point approximation)
        if ((hsigmoid_out >= expected_hsigmoid - 1) && (hsigmoid_out <= expected_hsigmoid + 1)) begin
            $display("PASS: %s hsigmoid(%d) = %d (expected %d)", test_name, input_val, hsigmoid_out, expected_hsigmoid);
            pass_count++;
        end else begin
            $display("FAIL: %s hsigmoid(%d) = %d (expected %d)", test_name, input_val, hsigmoid_out, expected_hsigmoid);
            fail_count++;
        end
        
        test_count += 2; // We tested 3 functions, but counted ReLU separately
    endtask
    
    // Main test sequence
    initial begin
        $display("=== Starting Activation Function Verification ===");
        
        rst = 1;
        test_input = 0;
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        all_tests_pass = 1;
        
        #(CLK_PERIOD * 5);
        rst = 0;
        #(CLK_PERIOD * 2);
        
        $display("\n--- Testing Key Values ---");
        
        // Test zero
        run_test(0, "Zero");
        
        // Test positive values
        run_test(16, "Positive_1.0"); // 1.0 in fixed-point
        run_test(32, "Positive_2.0"); // 2.0 in fixed-point
        run_test(48, "Positive_3.0"); // 3.0 in fixed-point
        run_test(96, "Positive_6.0"); // 6.0 in fixed-point
        
        // Test negative values
        run_test(-16, "Negative_-1.0"); // -1.0 in fixed-point
        run_test(-32, "Negative_-2.0"); // -2.0 in fixed-point
        run_test(-48, "Negative_-3.0"); // -3.0 in fixed-point
        run_test(-64, "Negative_-4.0"); // -4.0 in fixed-point
        
        // Test edge cases
        run_test(127, "Max_Positive");   // Maximum positive value
        run_test(-128, "Max_Negative");  // Maximum negative value
        
        // Test fractional values
        run_test(8, "Fractional_0.5");   // 0.5 in fixed-point
        run_test(24, "Fractional_1.5");  // 1.5 in fixed-point
        run_test(-8, "Fractional_-0.5"); // -0.5 in fixed-point
        
        #(CLK_PERIOD * 5);
        
        // Summary
        $display("\n=== Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Pass Rate: %.1f%%", (real'(pass_count) / real'(test_count)) * 100.0);
        
        if (fail_count == 0) begin
            $display("OVERALL RESULT: ALL TESTS PASSED ✅");
            all_tests_pass = 1;
        end else begin
            $display("OVERALL RESULT: SOME TESTS FAILED ❌");
            all_tests_pass = 0;
        end
        
        $display("=== Activation Function Verification Complete ===");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #(CLK_PERIOD * 1000);
        $display("ERROR: Test timeout!");
        $finish;
    end
    
endmodule 
