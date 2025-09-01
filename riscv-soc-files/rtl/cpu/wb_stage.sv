`timescale 1ns/1ps

module wb_stage #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // From MEM stage
    input  logic [XLEN-1:0]  alu_result_in,
    input  logic [XLEN-1:0]  mem_data_in,
    input  logic [4:0]       rd_in,
    input  logic             valid_in,
    input  logic             reg_write_in,
    input  logic [1:0]       wb_sel_in,
    
    // Register file write interface
    output logic             rf_we,
    output logic [4:0]       rf_waddr,
    output logic [XLEN-1:0]  rf_wdata
);

    // Writeback data selection
    always_comb begin
        case (wb_sel_in)
            2'b00: rf_wdata = alu_result_in;  // ALU result
            2'b01: rf_wdata = mem_data_in;    // Memory data
            2'b10: rf_wdata = alu_result_in;  // PC+4 (already calculated in EX)
            default: rf_wdata = alu_result_in;
        endcase
    end
    
    // Register file write control
    assign rf_we = reg_write_in && valid_in && (rd_in != 5'd0);
    assign rf_waddr = rd_in;

endmodule