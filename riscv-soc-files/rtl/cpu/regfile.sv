`timescale 1ns/1ps

module regfile #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // Write port
    input  logic             we,
    input  logic [4:0]       waddr,
    input  logic [XLEN-1:0]  wdata,
    
    // Read port 1
    input  logic [4:0]       raddr1,
    output logic [XLEN-1:0]  rdata1,
    
    // Read port 2
    input  logic [4:0]       raddr2,
    output logic [XLEN-1:0]  rdata2
);

    // 32 general-purpose registers
    logic [XLEN-1:0] rf [31:0];
    
    // Initialize registers on reset
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 32; i++) begin
                rf[i] <= '0;
            end
        end else if (we && (waddr != 5'd0)) begin
            // x0 is hardwired to zero, never write to it
            rf[waddr] <= wdata;
        end
    end
    
    // Read port 1 - x0 always returns 0
    assign rdata1 = (raddr1 == 5'd0) ? '0 : rf[raddr1];
    
    // Read port 2 - x0 always returns 0
    assign rdata2 = (raddr2 == 5'd0) ? '0 : rf[raddr2];

endmodule