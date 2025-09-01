`timescale 1ns/1ps

module riscv_cpu #(
    parameter XLEN = 32,
    parameter RESET_ADDR = 32'h0000_0000
)(
    input  logic             clk,
    input  logic             rst,
    
    // Instruction memory interface
    output logic [XLEN-1:0]  imem_addr,
    output logic             imem_req,
    input  logic [XLEN-1:0]  imem_rdata,
    input  logic             imem_ready,
    
    // Data memory interface
    output logic [XLEN-1:0]  dmem_addr,
    output logic [XLEN-1:0]  dmem_wdata,
    output logic [3:0]       dmem_we,
    output logic             dmem_req,
    input  logic [XLEN-1:0]  dmem_rdata,
    input  logic             dmem_ready,
    
    // Interrupt interface
    input  logic             timer_irq,
    input  logic             external_irq,
    
    // Debug interface
    output logic [XLEN-1:0]  debug_pc,
    output logic [XLEN-1:0]  debug_instr,
    output logic             debug_valid
);

    // Pipeline stage signals
    // IF/ID interface
    logic [XLEN-1:0] if_pc, if_instr;
    logic            if_valid;
    
    // ID/EX interface
    logic [XLEN-1:0] id_pc, id_rs1_data, id_rs2_data, id_imm;
    logic [4:0]      id_rd, id_rs1, id_rs2;
    logic [2:0]      id_funct3;
    logic [6:0]      id_funct7;
    logic            id_valid;
    logic [3:0]      id_alu_op;
    logic            id_alu_src1_sel, id_alu_src2_sel;
    logic            id_mem_read, id_mem_write, id_reg_write;
    logic [1:0]      id_wb_sel;
    logic            id_branch, id_jal, id_jalr;
    logic            id_mul_start, id_div_start;
    
    // EX/MEM interface
    logic [XLEN-1:0] ex_alu_result, ex_rs2_data;
    logic [4:0]      ex_rd;
    logic [2:0]      ex_funct3;
    logic            ex_valid;
    logic            ex_mem_read, ex_mem_write, ex_reg_write;
    logic [1:0]      ex_wb_sel;
    
    // MEM/WB interface
    logic [XLEN-1:0] mem_alu_result, mem_mem_data;
    logic [4:0]      mem_rd;
    logic            mem_valid;
    logic            mem_reg_write;
    logic [1:0]      mem_wb_sel;
    
    // Register file signals
    logic            rf_we;
    logic [4:0]      rf_waddr;
    logic [XLEN-1:0] rf_wdata;
    logic [4:0]      rf_raddr1, rf_raddr2;
    logic [XLEN-1:0] rf_rdata1, rf_rdata2;
    
    // Hazard control signals
    logic            stall_if, stall_id, stall_ex;
    logic            flush_id, flush_ex;
    logic [1:0]      fwd_rs1_sel, fwd_rs2_sel;
    logic            muldiv_busy;
    
    // Branch signals
    logic            branch_taken;
    logic [XLEN-1:0] branch_target;
    
    // Forwarding data
    logic [XLEN-1:0] fwd_ex_data, fwd_mem_data;
    
    // Register file instance
    regfile #(.XLEN(XLEN)) u_regfile (
        .clk(clk),
        .rst(rst),
        .we(rf_we),
        .waddr(rf_waddr),
        .wdata(rf_wdata),
        .raddr1(rf_raddr1),
        .rdata1(rf_rdata1),
        .raddr2(rf_raddr2),
        .rdata2(rf_rdata2)
    );
    
    // IF stage
    if_stage #(.XLEN(XLEN)) u_if_stage (
        .clk(clk),
        .rst(rst),
        .stall(stall_if),
        .flush(flush_id || branch_taken),
        .branch_taken(branch_taken),
        .branch_target(branch_target),
        .imem_addr(imem_addr),
        .imem_req(imem_req),
        .imem_rdata(imem_rdata),
        .imem_ready(imem_ready),
        .pc_out(if_pc),
        .instr_out(if_instr),
        .valid_out(if_valid)
    );
    
    // ID stage
    id_stage #(.XLEN(XLEN)) u_id_stage (
        .clk(clk),
        .rst(rst),
        .stall(stall_id),
        .flush(flush_id || branch_taken),
        .pc_in(if_pc),
        .instr_in(if_instr),
        .valid_in(if_valid),
        .rf_raddr1(rf_raddr1),
        .rf_raddr2(rf_raddr2),
        .rf_rdata1(rf_rdata1),
        .rf_rdata2(rf_rdata2),
        .pc_out(id_pc),
        .rs1_data_out(id_rs1_data),
        .rs2_data_out(id_rs2_data),
        .imm_out(id_imm),
        .rd_out(id_rd),
        .rs1_out(id_rs1),
        .rs2_out(id_rs2),
        .funct3_out(id_funct3),
        .funct7_out(id_funct7),
        .valid_out(id_valid),
        .alu_op_out(id_alu_op),
        .alu_src1_sel_out(id_alu_src1_sel),
        .alu_src2_sel_out(id_alu_src2_sel),
        .mem_read_out(id_mem_read),
        .mem_write_out(id_mem_write),
        .reg_write_out(id_reg_write),
        .wb_sel_out(id_wb_sel),
        .branch_out(id_branch),
        .jal_out(id_jal),
        .jalr_out(id_jalr),
        .mul_start_out(id_mul_start),
        .div_start_out(id_div_start)
    );
    
    // EX stage
    ex_stage #(.XLEN(XLEN)) u_ex_stage (
        .clk(clk),
        .rst(rst),
        .stall(stall_ex),
        .flush(flush_ex || branch_taken),
        .pc_in(id_pc),
        .rs1_data_in(id_rs1_data),
        .rs2_data_in(id_rs2_data),
        .imm_in(id_imm),
        .rd_in(id_rd),
        .rs1_in(id_rs1),
        .rs2_in(id_rs2),
        .funct3_in(id_funct3),
        .funct7_in(id_funct7),
        .valid_in(id_valid),
        .alu_op_in(id_alu_op),
        .alu_src1_sel_in(id_alu_src1_sel),
        .alu_src2_sel_in(id_alu_src2_sel),
        .mem_read_in(id_mem_read),
        .mem_write_in(id_mem_write),
        .reg_write_in(id_reg_write),
        .wb_sel_in(id_wb_sel),
        .branch_in(id_branch),
        .jal_in(id_jal),
        .jalr_in(id_jalr),
        .mul_start_in(id_mul_start),
        .div_start_in(id_div_start),
        .fwd_rs1_sel(fwd_rs1_sel),
        .fwd_rs2_sel(fwd_rs2_sel),
        .fwd_ex_data(ex_alu_result),
        .fwd_mem_data(mem_alu_result),
        .branch_taken(branch_taken),
        .branch_target(branch_target),
        .alu_result_out(ex_alu_result),
        .rs2_data_out(ex_rs2_data),
        .rd_out(ex_rd),
        .funct3_out(ex_funct3),
        .valid_out(ex_valid),
        .mem_read_out(ex_mem_read),
        .mem_write_out(ex_mem_write),
        .reg_write_out(ex_reg_write),
        .wb_sel_out(ex_wb_sel),
        .muldiv_busy(muldiv_busy)
    );
    
    // MEM stage
    mem_stage #(.XLEN(XLEN)) u_mem_stage (
        .clk(clk),
        .rst(rst),
        .stall(1'b0),  // Memory stage doesn't stall in this simple implementation
        .flush(1'b0),
        .alu_result_in(ex_alu_result),
        .rs2_data_in(ex_rs2_data),
        .rd_in(ex_rd),
        .funct3_in(ex_funct3),
        .valid_in(ex_valid),
        .mem_read_in(ex_mem_read),
        .mem_write_in(ex_mem_write),
        .reg_write_in(ex_reg_write),
        .wb_sel_in(ex_wb_sel),
        .dmem_addr(dmem_addr),
        .dmem_wdata(dmem_wdata),
        .dmem_we(dmem_we),
        .dmem_req(dmem_req),
        .dmem_rdata(dmem_rdata),
        .dmem_ready(dmem_ready),
        .alu_result_out(mem_alu_result),
        .mem_data_out(mem_mem_data),
        .rd_out(mem_rd),
        .valid_out(mem_valid),
        .reg_write_out(mem_reg_write),
        .wb_sel_out(mem_wb_sel)
    );
    
    // WB stage
    wb_stage #(.XLEN(XLEN)) u_wb_stage (
        .clk(clk),
        .rst(rst),
        .alu_result_in(mem_alu_result),
        .mem_data_in(mem_mem_data),
        .rd_in(mem_rd),
        .valid_in(mem_valid),
        .reg_write_in(mem_reg_write),
        .wb_sel_in(mem_wb_sel),
        .rf_we(rf_we),
        .rf_waddr(rf_waddr),
        .rf_wdata(rf_wdata)
    );
    
    // Hazard detection unit
    hazard_unit u_hazard_unit (
        .id_rs1_valid(!id_alu_src1_sel),
        .id_rs2_valid(!id_alu_src2_sel && !id_mem_write),
        .id_rs1(id_rs1),
        .id_rs2(id_rs2),
        .ex_rd_valid(ex_reg_write),
        .ex_rd(ex_rd),
        .ex_mem_read(ex_mem_read),
        .mem_rd_valid(mem_reg_write),
        .mem_rd(mem_rd),
        .wb_rd_valid(rf_we),
        .wb_rd(rf_waddr),
        .muldiv_busy(muldiv_busy),
        .stall_if(stall_if),
        .stall_id(stall_id),
        .stall_ex(stall_ex),
        .flush_id(flush_id),
        .flush_ex(flush_ex),
        .fwd_rs1_sel(fwd_rs1_sel),
        .fwd_rs2_sel(fwd_rs2_sel)
    );
    
    // Debug outputs
    assign debug_pc = if_pc;
    assign debug_instr = if_instr;
    assign debug_valid = if_valid;

endmodule