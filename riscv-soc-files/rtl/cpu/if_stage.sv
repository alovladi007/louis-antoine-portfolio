`timescale 1ns/1ps

module if_stage #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // Control signals
    input  logic             stall,
    input  logic             flush,
    
    // Branch/Jump signals
    input  logic             branch_taken,
    input  logic [XLEN-1:0]  branch_target,
    
    // Instruction memory interface
    output logic [XLEN-1:0]  imem_addr,
    output logic             imem_req,
    input  logic [XLEN-1:0]  imem_rdata,
    input  logic             imem_ready,
    
    // Output to ID stage
    output logic [XLEN-1:0]  pc_out,
    output logic [XLEN-1:0]  instr_out,
    output logic             valid_out
);

    logic [XLEN-1:0] pc_reg;
    logic [XLEN-1:0] pc_next;
    logic [XLEN-1:0] pc_plus4;
    
    // PC increment
    assign pc_plus4 = pc_reg + 32'd4;
    
    // Next PC selection
    always_comb begin
        if (branch_taken) begin
            pc_next = branch_target;
        end else if (!stall) begin
            pc_next = pc_plus4;
        end else begin
            pc_next = pc_reg;
        end
    end
    
    // PC register
    always_ff @(posedge clk) begin
        if (rst) begin
            pc_reg <= 32'h0000_0000;  // Reset vector
        end else if (!stall || branch_taken) begin
            pc_reg <= pc_next;
        end
    end
    
    // Instruction memory interface
    assign imem_addr = pc_reg;
    assign imem_req = !rst && !stall;
    
    // Pipeline register
    always_ff @(posedge clk) begin
        if (rst || flush) begin
            pc_out <= '0;
            instr_out <= 32'h0000_0013;  // NOP (ADDI x0, x0, 0)
            valid_out <= 1'b0;
        end else if (!stall) begin
            pc_out <= pc_reg;
            instr_out <= imem_ready ? imem_rdata : 32'h0000_0013;
            valid_out <= imem_ready;
        end
    end

endmodule