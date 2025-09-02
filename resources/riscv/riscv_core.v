//==============================================================================
// RISC-V RV32IMC Processor Core
// Author: Louis Antoine
// Date: December 2024
// Description: 5-stage pipelined RISC-V processor with compressed instruction support
//==============================================================================

`timescale 1ns / 1ps

module riscv_core #(
    parameter XLEN = 32,
    parameter RESET_ADDR = 32'h0000_0000,
    parameter HART_ID = 0
)(
    input  wire                 clk,
    input  wire                 rst_n,
    
    // Instruction Memory Interface
    output wire [XLEN-1:0]      imem_addr,
    output wire                 imem_req,
    input  wire [31:0]          imem_rdata,
    input  wire                 imem_ready,
    
    // Data Memory Interface
    output wire [XLEN-1:0]      dmem_addr,
    output wire [XLEN-1:0]      dmem_wdata,
    output wire [3:0]           dmem_we,
    output wire                 dmem_req,
    input  wire [XLEN-1:0]      dmem_rdata,
    input  wire                 dmem_ready,
    
    // Interrupt Interface
    input  wire                 ext_irq,
    input  wire                 timer_irq,
    input  wire                 software_irq
);

//==============================================================================
// Internal Signals
//==============================================================================

// Program Counter
reg  [XLEN-1:0] pc_reg;
wire [XLEN-1:0] pc_next;
wire [XLEN-1:0] pc_plus_4;
wire [XLEN-1:0] pc_plus_2;
wire            pc_stall;

// Pipeline Registers
// IF/ID Stage
reg  [XLEN-1:0] if_id_pc;
reg  [31:0]     if_id_instr;
reg             if_id_valid;
reg             if_id_compressed;

// ID/EX Stage
reg  [XLEN-1:0] id_ex_pc;
reg  [XLEN-1:0] id_ex_rs1_data;
reg  [XLEN-1:0] id_ex_rs2_data;
reg  [4:0]      id_ex_rs1;
reg  [4:0]      id_ex_rs2;
reg  [4:0]      id_ex_rd;
reg  [XLEN-1:0] id_ex_imm;
reg  [6:0]      id_ex_opcode;
reg  [2:0]      id_ex_funct3;
reg  [6:0]      id_ex_funct7;
reg             id_ex_valid;

// Control signals
reg             id_ex_alu_src;
reg  [3:0]      id_ex_alu_op;
reg             id_ex_mem_read;
reg             id_ex_mem_write;
reg             id_ex_reg_write;
reg             id_ex_branch;
reg             id_ex_jump;

// EX/MEM Stage
reg  [XLEN-1:0] ex_mem_alu_result;
reg  [XLEN-1:0] ex_mem_rs2_data;
reg  [4:0]      ex_mem_rd;
reg  [XLEN-1:0] ex_mem_pc;
reg             ex_mem_valid;
reg             ex_mem_mem_read;
reg             ex_mem_mem_write;
reg             ex_mem_reg_write;

// MEM/WB Stage
reg  [XLEN-1:0] mem_wb_data;
reg  [4:0]      mem_wb_rd;
reg             mem_wb_valid;
reg             mem_wb_reg_write;

// Hazard Detection
wire            stall;
wire            flush;
wire [1:0]      forward_a;
wire [1:0]      forward_b;

// Branch Prediction
wire            branch_taken;
wire [XLEN-1:0] branch_target;
wire            branch_mispredict;

//==============================================================================
// Instruction Fetch Stage
//==============================================================================

assign pc_plus_4 = pc_reg + 4;
assign pc_plus_2 = pc_reg + 2;

// PC Mux
always @(*) begin
    if (branch_mispredict)
        pc_next = branch_target;
    else if (stall)
        pc_next = pc_reg;
    else if (if_id_compressed)
        pc_next = pc_plus_2;
    else
        pc_next = pc_plus_4;
end

// PC Register
always @(posedge clk) begin
    if (!rst_n)
        pc_reg <= RESET_ADDR;
    else if (!pc_stall)
        pc_reg <= pc_next;
end

// Instruction Memory Interface
assign imem_addr = pc_reg;
assign imem_req = 1'b1;

// IF/ID Pipeline Register
always @(posedge clk) begin
    if (!rst_n || flush) begin
        if_id_pc <= 32'h0;
        if_id_instr <= 32'h0000_0013; // NOP
        if_id_valid <= 1'b0;
        if_id_compressed <= 1'b0;
    end else if (!stall) begin
        if_id_pc <= pc_reg;
        if_id_instr <= imem_rdata;
        if_id_valid <= imem_ready;
        if_id_compressed <= (imem_rdata[1:0] != 2'b11);
    end
end

//==============================================================================
// Instruction Decode Stage
//==============================================================================

// Register File
reg [XLEN-1:0] register_file [0:31];

// Decode signals
wire [4:0]  rs1 = if_id_instr[19:15];
wire [4:0]  rs2 = if_id_instr[24:20];
wire [4:0]  rd  = if_id_instr[11:7];
wire [6:0]  opcode = if_id_instr[6:0];
wire [2:0]  funct3 = if_id_instr[14:12];
wire [6:0]  funct7 = if_id_instr[31:25];

// Register Read
wire [XLEN-1:0] rs1_data = (rs1 == 5'b0) ? 32'h0 : register_file[rs1];
wire [XLEN-1:0] rs2_data = (rs2 == 5'b0) ? 32'h0 : register_file[rs2];

// Immediate Generation
wire [XLEN-1:0] imm_i = {{20{if_id_instr[31]}}, if_id_instr[31:20]};
wire [XLEN-1:0] imm_s = {{20{if_id_instr[31]}}, if_id_instr[31:25], if_id_instr[11:7]};
wire [XLEN-1:0] imm_b = {{19{if_id_instr[31]}}, if_id_instr[31], if_id_instr[7], 
                         if_id_instr[30:25], if_id_instr[11:8], 1'b0};
wire [XLEN-1:0] imm_u = {if_id_instr[31:12], 12'h0};
wire [XLEN-1:0] imm_j = {{11{if_id_instr[31]}}, if_id_instr[31], if_id_instr[19:12],
                         if_id_instr[20], if_id_instr[30:21], 1'b0};

reg [XLEN-1:0] immediate;
always @(*) begin
    case (opcode)
        7'b0010011, // I-type
        7'b0000011, // Load
        7'b1100111: immediate = imm_i; // JALR
        7'b0100011: immediate = imm_s; // Store
        7'b1100011: immediate = imm_b; // Branch
        7'b0110111, // LUI
        7'b0010111: immediate = imm_u; // AUIPC
        7'b1101111: immediate = imm_j; // JAL
        default:    immediate = 32'h0;
    endcase
end

// Control Unit
always @(*) begin
    // Default values
    alu_src = 1'b0;
    alu_op = 4'b0000;
    mem_read = 1'b0;
    mem_write = 1'b0;
    reg_write = 1'b0;
    branch = 1'b0;
    jump = 1'b0;
    
    case (opcode)
        7'b0110011: begin // R-type
            reg_write = 1'b1;
            alu_src = 1'b0;
            case (funct3)
                3'b000: alu_op = (funct7[5]) ? 4'b0001 : 4'b0000; // ADD/SUB
                3'b001: alu_op = 4'b0010; // SLL
                3'b010: alu_op = 4'b0011; // SLT
                3'b011: alu_op = 4'b0100; // SLTU
                3'b100: alu_op = 4'b0101; // XOR
                3'b101: alu_op = (funct7[5]) ? 4'b0111 : 4'b0110; // SRA/SRL
                3'b110: alu_op = 4'b1000; // OR
                3'b111: alu_op = 4'b1001; // AND
            endcase
        end
        
        7'b0010011: begin // I-type
            reg_write = 1'b1;
            alu_src = 1'b1;
            case (funct3)
                3'b000: alu_op = 4'b0000; // ADDI
                3'b010: alu_op = 4'b0011; // SLTI
                3'b011: alu_op = 4'b0100; // SLTIU
                3'b100: alu_op = 4'b0101; // XORI
                3'b110: alu_op = 4'b1000; // ORI
                3'b111: alu_op = 4'b1001; // ANDI
                3'b001: alu_op = 4'b0010; // SLLI
                3'b101: alu_op = (if_id_instr[30]) ? 4'b0111 : 4'b0110; // SRAI/SRLI
            endcase
        end
        
        7'b0000011: begin // Load
            reg_write = 1'b1;
            mem_read = 1'b1;
            alu_src = 1'b1;
            alu_op = 4'b0000; // ADD
        end
        
        7'b0100011: begin // Store
            mem_write = 1'b1;
            alu_src = 1'b1;
            alu_op = 4'b0000; // ADD
        end
        
        7'b1100011: begin // Branch
            branch = 1'b1;
            alu_src = 1'b0;
        end
        
        7'b1101111: begin // JAL
            reg_write = 1'b1;
            jump = 1'b1;
        end
        
        7'b1100111: begin // JALR
            reg_write = 1'b1;
            jump = 1'b1;
            alu_src = 1'b1;
        end
        
        7'b0110111: begin // LUI
            reg_write = 1'b1;
        end
        
        7'b0010111: begin // AUIPC
            reg_write = 1'b1;
        end
    endcase
end

// ID/EX Pipeline Register
always @(posedge clk) begin
    if (!rst_n || flush) begin
        id_ex_pc <= 32'h0;
        id_ex_rs1_data <= 32'h0;
        id_ex_rs2_data <= 32'h0;
        id_ex_rs1 <= 5'h0;
        id_ex_rs2 <= 5'h0;
        id_ex_rd <= 5'h0;
        id_ex_imm <= 32'h0;
        id_ex_opcode <= 7'h0;
        id_ex_funct3 <= 3'h0;
        id_ex_funct7 <= 7'h0;
        id_ex_valid <= 1'b0;
        id_ex_alu_src <= 1'b0;
        id_ex_alu_op <= 4'h0;
        id_ex_mem_read <= 1'b0;
        id_ex_mem_write <= 1'b0;
        id_ex_reg_write <= 1'b0;
        id_ex_branch <= 1'b0;
        id_ex_jump <= 1'b0;
    end else if (!stall) begin
        id_ex_pc <= if_id_pc;
        id_ex_rs1_data <= rs1_data;
        id_ex_rs2_data <= rs2_data;
        id_ex_rs1 <= rs1;
        id_ex_rs2 <= rs2;
        id_ex_rd <= rd;
        id_ex_imm <= immediate;
        id_ex_opcode <= opcode;
        id_ex_funct3 <= funct3;
        id_ex_funct7 <= funct7;
        id_ex_valid <= if_id_valid;
        id_ex_alu_src <= alu_src;
        id_ex_alu_op <= alu_op;
        id_ex_mem_read <= mem_read;
        id_ex_mem_write <= mem_write;
        id_ex_reg_write <= reg_write;
        id_ex_branch <= branch;
        id_ex_jump <= jump;
    end
end

//==============================================================================
// Execute Stage
//==============================================================================

// Forwarding Mux
wire [XLEN-1:0] alu_src1 = (forward_a == 2'b10) ? ex_mem_alu_result :
                           (forward_a == 2'b01) ? mem_wb_data :
                           id_ex_rs1_data;

wire [XLEN-1:0] alu_src2_fwd = (forward_b == 2'b10) ? ex_mem_alu_result :
                               (forward_b == 2'b01) ? mem_wb_data :
                               id_ex_rs2_data;

wire [XLEN-1:0] alu_src2 = id_ex_alu_src ? id_ex_imm : alu_src2_fwd;

// ALU
reg [XLEN-1:0] alu_result;
always @(*) begin
    case (id_ex_alu_op)
        4'b0000: alu_result = alu_src1 + alu_src2;           // ADD
        4'b0001: alu_result = alu_src1 - alu_src2;           // SUB
        4'b0010: alu_result = alu_src1 << alu_src2[4:0];     // SLL
        4'b0011: alu_result = ($signed(alu_src1) < $signed(alu_src2)) ? 1 : 0; // SLT
        4'b0100: alu_result = (alu_src1 < alu_src2) ? 1 : 0; // SLTU
        4'b0101: alu_result = alu_src1 ^ alu_src2;           // XOR
        4'b0110: alu_result = alu_src1 >> alu_src2[4:0];     // SRL
        4'b0111: alu_result = $signed(alu_src1) >>> alu_src2[4:0]; // SRA
        4'b1000: alu_result = alu_src1 | alu_src2;           // OR
        4'b1001: alu_result = alu_src1 & alu_src2;           // AND
        default: alu_result = 32'h0;
    endcase
end

// Branch Unit
always @(*) begin
    branch_taken = 1'b0;
    if (id_ex_branch) begin
        case (id_ex_funct3)
            3'b000: branch_taken = (alu_src1 == alu_src2_fwd);  // BEQ
            3'b001: branch_taken = (alu_src1 != alu_src2_fwd);  // BNE
            3'b100: branch_taken = ($signed(alu_src1) < $signed(alu_src2_fwd));  // BLT
            3'b101: branch_taken = ($signed(alu_src1) >= $signed(alu_src2_fwd)); // BGE
            3'b110: branch_taken = (alu_src1 < alu_src2_fwd);   // BLTU
            3'b111: branch_taken = (alu_src1 >= alu_src2_fwd);  // BGEU
        endcase
    end
end

assign branch_target = id_ex_pc + id_ex_imm;
assign branch_mispredict = (id_ex_branch && branch_taken) || id_ex_jump;

// EX/MEM Pipeline Register
always @(posedge clk) begin
    if (!rst_n) begin
        ex_mem_alu_result <= 32'h0;
        ex_mem_rs2_data <= 32'h0;
        ex_mem_rd <= 5'h0;
        ex_mem_pc <= 32'h0;
        ex_mem_valid <= 1'b0;
        ex_mem_mem_read <= 1'b0;
        ex_mem_mem_write <= 1'b0;
        ex_mem_reg_write <= 1'b0;
    end else begin
        ex_mem_alu_result <= alu_result;
        ex_mem_rs2_data <= alu_src2_fwd;
        ex_mem_rd <= id_ex_rd;
        ex_mem_pc <= id_ex_pc;
        ex_mem_valid <= id_ex_valid;
        ex_mem_mem_read <= id_ex_mem_read;
        ex_mem_mem_write <= id_ex_mem_write;
        ex_mem_reg_write <= id_ex_reg_write;
    end
end

//==============================================================================
// Memory Stage
//==============================================================================

// Data Memory Interface
assign dmem_addr = ex_mem_alu_result;
assign dmem_wdata = ex_mem_rs2_data;
assign dmem_we = {4{ex_mem_mem_write}};
assign dmem_req = ex_mem_mem_read || ex_mem_mem_write;

// MEM/WB Pipeline Register
always @(posedge clk) begin
    if (!rst_n) begin
        mem_wb_data <= 32'h0;
        mem_wb_rd <= 5'h0;
        mem_wb_valid <= 1'b0;
        mem_wb_reg_write <= 1'b0;
    end else begin
        mem_wb_data <= ex_mem_mem_read ? dmem_rdata : ex_mem_alu_result;
        mem_wb_rd <= ex_mem_rd;
        mem_wb_valid <= ex_mem_valid;
        mem_wb_reg_write <= ex_mem_reg_write;
    end
end

//==============================================================================
// Write Back Stage
//==============================================================================

// Register Write
always @(posedge clk) begin
    if (mem_wb_reg_write && mem_wb_valid && (mem_wb_rd != 5'h0)) begin
        register_file[mem_wb_rd] <= mem_wb_data;
    end
end

//==============================================================================
// Hazard Detection Unit
//==============================================================================

hazard_detection_unit hdu (
    .id_ex_mem_read(id_ex_mem_read),
    .id_ex_rd(id_ex_rd),
    .if_id_rs1(rs1),
    .if_id_rs2(rs2),
    .stall(stall)
);

assign flush = branch_mispredict;
assign pc_stall = stall || !imem_ready || !dmem_ready;

//==============================================================================
// Forwarding Unit
//==============================================================================

forwarding_unit fwd (
    .ex_mem_reg_write(ex_mem_reg_write),
    .ex_mem_rd(ex_mem_rd),
    .mem_wb_reg_write(mem_wb_reg_write),
    .mem_wb_rd(mem_wb_rd),
    .id_ex_rs1(id_ex_rs1),
    .id_ex_rs2(id_ex_rs2),
    .forward_a(forward_a),
    .forward_b(forward_b)
);

endmodule

//==============================================================================
// Hazard Detection Unit
//==============================================================================

module hazard_detection_unit (
    input  wire        id_ex_mem_read,
    input  wire [4:0]  id_ex_rd,
    input  wire [4:0]  if_id_rs1,
    input  wire [4:0]  if_id_rs2,
    output wire        stall
);

assign stall = id_ex_mem_read && 
               ((id_ex_rd == if_id_rs1) || (id_ex_rd == if_id_rs2)) &&
               (id_ex_rd != 5'h0);

endmodule

//==============================================================================
// Forwarding Unit
//==============================================================================

module forwarding_unit (
    input  wire        ex_mem_reg_write,
    input  wire [4:0]  ex_mem_rd,
    input  wire        mem_wb_reg_write,
    input  wire [4:0]  mem_wb_rd,
    input  wire [4:0]  id_ex_rs1,
    input  wire [4:0]  id_ex_rs2,
    output reg  [1:0]  forward_a,
    output reg  [1:0]  forward_b
);

always @(*) begin
    // Forward A
    if (ex_mem_reg_write && (ex_mem_rd != 5'h0) && (ex_mem_rd == id_ex_rs1))
        forward_a = 2'b10;
    else if (mem_wb_reg_write && (mem_wb_rd != 5'h0) && (mem_wb_rd == id_ex_rs1))
        forward_a = 2'b01;
    else
        forward_a = 2'b00;
    
    // Forward B
    if (ex_mem_reg_write && (ex_mem_rd != 5'h0) && (ex_mem_rd == id_ex_rs2))
        forward_b = 2'b10;
    else if (mem_wb_reg_write && (mem_wb_rd != 5'h0) && (mem_wb_rd == id_ex_rs2))
        forward_b = 2'b01;
    else
        forward_b = 2'b00;
end

endmodule