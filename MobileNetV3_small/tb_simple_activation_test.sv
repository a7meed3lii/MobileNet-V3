// Simple activation function test
`timescale 1ns/1ps

module tb_simple_activation_test();
    parameter DATA_WIDTH = 8;
    parameter FRAC_BITS = 4;
    
    logic clk, rst;
    logic signed [DATA_WIDTH-1:0] test_input;
    logic signed [DATA_WIDTH-1:0] relu_out, hswish_out, hsigmoid_out;
    logic test_pass;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Instantiate activation functions
    relu #(.DATA_WIDTH(DATA_WIDTH)) relu_inst (
        .clk(clk), .rst(rst), .data_in(test_input), .data_out(relu_out)
    );
    
    hswish #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hswish_inst (
        .clk(clk), .rst(rst), .data_in(test_input), .data_out(hswish_out)
    );
    
    hsigmoid #(.DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)) hsigmoid_inst (
        .clk(clk), .rst(rst), .data_in(test_input), .data_out(hsigmoid_out)
    );
    
    // Test sequence
    initial begin
        rst = 1;
        test_input = 0;
        test_pass = 1;
        
        #100;
        rst = 0;
        #50;
        
        // Test ReLU with zero
        test_input = 0;
        #20;
        if (relu_out == 0) begin
            $display("PASS: ReLU(0) = 0");
        end else begin
            $display("FAIL: ReLU(0) should be 0, got %d", relu_out);
            test_pass = 0;
        end
        
        // Test ReLU with positive
        test_input = 32; // 2.0 in fixed point
        #20;
        if (relu_out == 32) begin
            $display("PASS: ReLU(32) = 32");
        end else begin
            $display("FAIL: ReLU(32) should be 32, got %d", relu_out);
            test_pass = 0;
        end
        
        // Test ReLU with negative
        test_input = -16; // -1.0 in fixed point
        #20;
        if (relu_out == 0) begin
            $display("PASS: ReLU(-16) = 0");
        end else begin
            $display("FAIL: ReLU(-16) should be 0, got %d", relu_out);
            test_pass = 0;
        end
        
        if (test_pass) begin
            $display("OVERALL: PASS - All activation tests passed");
        end else begin
            $display("OVERALL: FAIL - Some activation tests failed");
        end
        
        $finish;
    end
    
endmodule
