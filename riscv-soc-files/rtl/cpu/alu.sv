`timescale 1ns/1ps

module alu #(
    parameter XLEN = 32
)(
    input  logic [XLEN-1:0] a,
    input  logic [XLEN-1:0] b,
    input  logic [3:0]      alu_op,
    output logic [XLEN-1:0] result,
    output logic            zero,
    output logic            lt,
    output logic            ltu
);

    // ALU operation encoding
    localparam ALU_ADD  = 4'b0000;
    localparam ALU_SUB  = 4'b0001;
    localparam ALU_SLL  = 4'b0010;
    localparam ALU_SLT  = 4'b0011;
    localparam ALU_SLTU = 4'b0100;
    localparam ALU_XOR  = 4'b0101;
    localparam ALU_SRL  = 4'b0110;
    localparam ALU_SRA  = 4'b0111;
    localparam ALU_OR   = 4'b1000;
    localparam ALU_AND  = 4'b1001;

    logic [XLEN-1:0] add_sub_result;
    logic [XLEN-1:0] slt_result;
    logic [XLEN-1:0] sltu_result;
    logic [4:0]      shamt;
    
    // Shift amount (lower 5 bits for RV32)
    assign shamt = b[4:0];
    
    // Addition/Subtraction
    assign add_sub_result = (alu_op == ALU_SUB) ? (a - b) : (a + b);
    
    // Set less than (signed)
    assign lt = ($signed(a) < $signed(b));
    assign slt_result = {{(XLEN-1){1'b0}}, lt};
    
    // Set less than (unsigned)
    assign ltu = (a < b);
    assign sltu_result = {{(XLEN-1){1'b0}}, ltu};
    
    // Main ALU mux
    always_comb begin
        case (alu_op)
            ALU_ADD:  result = add_sub_result;
            ALU_SUB:  result = add_sub_result;
            ALU_SLL:  result = a << shamt;
            ALU_SLT:  result = slt_result;
            ALU_SLTU: result = sltu_result;
            ALU_XOR:  result = a ^ b;
            ALU_SRL:  result = a >> shamt;
            ALU_SRA:  result = $signed(a) >>> shamt;
            ALU_OR:   result = a | b;
            ALU_AND:  result = a & b;
            default:  result = '0;
        endcase
    end
    
    // Zero flag
    assign zero = (result == '0);

endmodule