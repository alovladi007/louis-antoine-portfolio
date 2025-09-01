`timescale 1ns/1ps

module id_stage #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // Control signals
    input  logic             stall,
    input  logic             flush,
    
    // From IF stage
    input  logic [XLEN-1:0]  pc_in,
    input  logic [XLEN-1:0]  instr_in,
    input  logic             valid_in,
    
    // Register file interface
    output logic [4:0]       rf_raddr1,
    output logic [4:0]       rf_raddr2,
    input  logic [XLEN-1:0]  rf_rdata1,
    input  logic [XLEN-1:0]  rf_rdata2,
    
    // To EX stage
    output logic [XLEN-1:0]  pc_out,
    output logic [XLEN-1:0]  rs1_data_out,
    output logic [XLEN-1:0]  rs2_data_out,
    output logic [XLEN-1:0]  imm_out,
    output logic [4:0]       rd_out,
    output logic [4:0]       rs1_out,
    output logic [4:0]       rs2_out,
    output logic [2:0]       funct3_out,
    output logic [6:0]       funct7_out,
    output logic             valid_out,
    
    // Control signals to EX
    output logic [3:0]       alu_op_out,
    output logic             alu_src1_sel_out,  // 0: rs1, 1: PC
    output logic             alu_src2_sel_out,  // 0: rs2, 1: imm
    output logic             mem_read_out,
    output logic             mem_write_out,
    output logic             reg_write_out,
    output logic [1:0]       wb_sel_out,        // 00: ALU, 01: MEM, 10: PC+4
    output logic             branch_out,
    output logic             jal_out,
    output logic             jalr_out,
    output logic             mul_start_out,
    output logic             div_start_out
);

    // Instruction fields
    logic [6:0]  opcode;
    logic [4:0]  rd;
    logic [2:0]  funct3;
    logic [4:0]  rs1;
    logic [4:0]  rs2;
    logic [6:0]  funct7;
    
    // Decode instruction fields
    assign opcode = instr_in[6:0];
    assign rd     = instr_in[11:7];
    assign funct3 = instr_in[14:12];
    assign rs1    = instr_in[19:15];
    assign rs2    = instr_in[24:20];
    assign funct7 = instr_in[31:25];
    
    // Immediate generation
    logic [XLEN-1:0] imm_i, imm_s, imm_b, imm_u, imm_j;
    logic [XLEN-1:0] imm;
    
    always_comb begin
        // I-type immediate
        imm_i = {{20{instr_in[31]}}, instr_in[31:20]};
        
        // S-type immediate
        imm_s = {{20{instr_in[31]}}, instr_in[31:25], instr_in[11:7]};
        
        // B-type immediate
        imm_b = {{19{instr_in[31]}}, instr_in[31], instr_in[7], instr_in[30:25], instr_in[11:8], 1'b0};
        
        // U-type immediate
        imm_u = {instr_in[31:12], 12'b0};
        
        // J-type immediate
        imm_j = {{11{instr_in[31]}}, instr_in[31], instr_in[19:12], instr_in[20], instr_in[30:21], 1'b0};
        
        // Select immediate based on opcode
        case (opcode)
            7'b0010011: imm = imm_i;  // I-type (ADDI, etc.)
            7'b0000011: imm = imm_i;  // Load
            7'b0100011: imm = imm_s;  // Store
            7'b1100011: imm = imm_b;  // Branch
            7'b0110111: imm = imm_u;  // LUI
            7'b0010111: imm = imm_u;  // AUIPC
            7'b1101111: imm = imm_j;  // JAL
            7'b1100111: imm = imm_i;  // JALR
            default:    imm = '0;
        endcase
    end
    
    // Control signal generation
    logic [3:0]  alu_op;
    logic        alu_src1_sel;
    logic        alu_src2_sel;
    logic        mem_read;
    logic        mem_write;
    logic        reg_write;
    logic [1:0]  wb_sel;
    logic        branch;
    logic        jal;
    logic        jalr;
    logic        mul_start;
    logic        div_start;
    
    always_comb begin
        // Default values
        alu_op = 4'b0000;
        alu_src1_sel = 1'b0;
        alu_src2_sel = 1'b0;
        mem_read = 1'b0;
        mem_write = 1'b0;
        reg_write = 1'b0;
        wb_sel = 2'b00;
        branch = 1'b0;
        jal = 1'b0;
        jalr = 1'b0;
        mul_start = 1'b0;
        div_start = 1'b0;
        
        case (opcode)
            7'b0110011: begin  // R-type
                reg_write = 1'b1;
                wb_sel = 2'b00;  // ALU result
                
                if (funct7 == 7'b0000001) begin
                    // M extension
                    case (funct3)
                        3'b000, 3'b001, 3'b010, 3'b011: mul_start = 1'b1;
                        3'b100, 3'b101, 3'b110, 3'b111: div_start = 1'b1;
                    endcase
                end else begin
                    // Base ALU operations
                    case (funct3)
                        3'b000: alu_op = (funct7[5]) ? 4'b0001 : 4'b0000;  // SUB/ADD
                        3'b001: alu_op = 4'b0010;  // SLL
                        3'b010: alu_op = 4'b0011;  // SLT
                        3'b011: alu_op = 4'b0100;  // SLTU
                        3'b100: alu_op = 4'b0101;  // XOR
                        3'b101: alu_op = (funct7[5]) ? 4'b0111 : 4'b0110;  // SRA/SRL
                        3'b110: alu_op = 4'b1000;  // OR
                        3'b111: alu_op = 4'b1001;  // AND
                    endcase
                end
            end
            
            7'b0010011: begin  // I-type
                reg_write = 1'b1;
                alu_src2_sel = 1'b1;  // Use immediate
                wb_sel = 2'b00;  // ALU result
                
                case (funct3)
                    3'b000: alu_op = 4'b0000;  // ADDI
                    3'b001: alu_op = 4'b0010;  // SLLI
                    3'b010: alu_op = 4'b0011;  // SLTI
                    3'b011: alu_op = 4'b0100;  // SLTIU
                    3'b100: alu_op = 4'b0101;  // XORI
                    3'b101: alu_op = (funct7[5]) ? 4'b0111 : 4'b0110;  // SRAI/SRLI
                    3'b110: alu_op = 4'b1000;  // ORI
                    3'b111: alu_op = 4'b1001;  // ANDI
                endcase
            end
            
            7'b0000011: begin  // Load
                reg_write = 1'b1;
                alu_op = 4'b0000;  // ADD for address calculation
                alu_src2_sel = 1'b1;  // Use immediate
                mem_read = 1'b1;
                wb_sel = 2'b01;  // Memory data
            end
            
            7'b0100011: begin  // Store
                alu_op = 4'b0000;  // ADD for address calculation
                alu_src2_sel = 1'b1;  // Use immediate
                mem_write = 1'b1;
            end
            
            7'b1100011: begin  // Branch
                alu_op = 4'b0001;  // SUB for comparison
                branch = 1'b1;
            end
            
            7'b1101111: begin  // JAL
                reg_write = 1'b1;
                wb_sel = 2'b10;  // PC+4
                jal = 1'b1;
            end
            
            7'b1100111: begin  // JALR
                reg_write = 1'b1;
                alu_op = 4'b0000;  // ADD
                alu_src2_sel = 1'b1;  // Use immediate
                wb_sel = 2'b10;  // PC+4
                jalr = 1'b1;
            end
            
            7'b0110111: begin  // LUI
                reg_write = 1'b1;
                alu_op = 4'b0000;  // Pass through
                alu_src1_sel = 1'b1;  // Use zero
                alu_src2_sel = 1'b1;  // Use immediate
                wb_sel = 2'b00;  // ALU result
            end
            
            7'b0010111: begin  // AUIPC
                reg_write = 1'b1;
                alu_op = 4'b0000;  // ADD
                alu_src1_sel = 1'b1;  // Use PC
                alu_src2_sel = 1'b1;  // Use immediate
                wb_sel = 2'b00;  // ALU result
            end
        endcase
    end
    
    // Register file read
    assign rf_raddr1 = rs1;
    assign rf_raddr2 = rs2;
    
    // Pipeline registers
    always_ff @(posedge clk) begin
        if (rst || flush) begin
            pc_out <= '0;
            rs1_data_out <= '0;
            rs2_data_out <= '0;
            imm_out <= '0;
            rd_out <= '0;
            rs1_out <= '0;
            rs2_out <= '0;
            funct3_out <= '0;
            funct7_out <= '0;
            valid_out <= 1'b0;
            alu_op_out <= '0;
            alu_src1_sel_out <= 1'b0;
            alu_src2_sel_out <= 1'b0;
            mem_read_out <= 1'b0;
            mem_write_out <= 1'b0;
            reg_write_out <= 1'b0;
            wb_sel_out <= 2'b00;
            branch_out <= 1'b0;
            jal_out <= 1'b0;
            jalr_out <= 1'b0;
            mul_start_out <= 1'b0;
            div_start_out <= 1'b0;
        end else if (!stall) begin
            pc_out <= pc_in;
            rs1_data_out <= rf_rdata1;
            rs2_data_out <= rf_rdata2;
            imm_out <= imm;
            rd_out <= rd;
            rs1_out <= rs1;
            rs2_out <= rs2;
            funct3_out <= funct3;
            funct7_out <= funct7;
            valid_out <= valid_in;
            alu_op_out <= alu_op;
            alu_src1_sel_out <= alu_src1_sel;
            alu_src2_sel_out <= alu_src2_sel;
            mem_read_out <= mem_read;
            mem_write_out <= mem_write;
            reg_write_out <= reg_write;
            wb_sel_out <= wb_sel;
            branch_out <= branch;
            jal_out <= jal;
            jalr_out <= jalr;
            mul_start_out <= mul_start;
            div_start_out <= div_start;
        end
    end

endmodule