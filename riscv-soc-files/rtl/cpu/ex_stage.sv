`timescale 1ns/1ps

module ex_stage #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // Control signals
    input  logic             stall,
    input  logic             flush,
    
    // From ID stage
    input  logic [XLEN-1:0]  pc_in,
    input  logic [XLEN-1:0]  rs1_data_in,
    input  logic [XLEN-1:0]  rs2_data_in,
    input  logic [XLEN-1:0]  imm_in,
    input  logic [4:0]       rd_in,
    input  logic [4:0]       rs1_in,
    input  logic [4:0]       rs2_in,
    input  logic [2:0]       funct3_in,
    input  logic [6:0]       funct7_in,
    input  logic             valid_in,
    
    // Control signals from ID
    input  logic [3:0]       alu_op_in,
    input  logic             alu_src1_sel_in,
    input  logic             alu_src2_sel_in,
    input  logic             mem_read_in,
    input  logic             mem_write_in,
    input  logic             reg_write_in,
    input  logic [1:0]       wb_sel_in,
    input  logic             branch_in,
    input  logic             jal_in,
    input  logic             jalr_in,
    input  logic             mul_start_in,
    input  logic             div_start_in,
    
    // Forwarding inputs
    input  logic [1:0]       fwd_rs1_sel,  // 00: ID, 01: EX/MEM, 10: MEM/WB
    input  logic [1:0]       fwd_rs2_sel,
    input  logic [XLEN-1:0]  fwd_ex_data,
    input  logic [XLEN-1:0]  fwd_mem_data,
    
    // Branch resolution
    output logic             branch_taken,
    output logic [XLEN-1:0]  branch_target,
    
    // To MEM stage
    output logic [XLEN-1:0]  alu_result_out,
    output logic [XLEN-1:0]  rs2_data_out,
    output logic [4:0]       rd_out,
    output logic [2:0]       funct3_out,
    output logic             valid_out,
    output logic             mem_read_out,
    output logic             mem_write_out,
    output logic             reg_write_out,
    output logic [1:0]       wb_sel_out,
    
    // MulDiv interface
    output logic             muldiv_busy
);

    // Internal signals
    logic [XLEN-1:0] alu_src1, alu_src2;
    logic [XLEN-1:0] alu_result;
    logic             alu_zero, alu_lt, alu_ltu;
    logic [XLEN-1:0] pc_plus4;
    logic [XLEN-1:0] forwarded_rs1, forwarded_rs2;
    
    // Forwarding muxes
    always_comb begin
        case (fwd_rs1_sel)
            2'b00: forwarded_rs1 = rs1_data_in;
            2'b01: forwarded_rs1 = fwd_ex_data;
            2'b10: forwarded_rs1 = fwd_mem_data;
            default: forwarded_rs1 = rs1_data_in;
        endcase
        
        case (fwd_rs2_sel)
            2'b00: forwarded_rs2 = rs2_data_in;
            2'b01: forwarded_rs2 = fwd_ex_data;
            2'b10: forwarded_rs2 = fwd_mem_data;
            default: forwarded_rs2 = rs2_data_in;
        endcase
    end
    
    // ALU source selection
    assign pc_plus4 = pc_in + 32'd4;
    
    always_comb begin
        // Source 1: RS1 or PC
        if (alu_src1_sel_in) begin
            alu_src1 = pc_in;
        end else begin
            alu_src1 = forwarded_rs1;
        end
        
        // Source 2: RS2 or immediate
        if (alu_src2_sel_in) begin
            alu_src2 = imm_in;
        end else begin
            alu_src2 = forwarded_rs2;
        end
    end
    
    // ALU instance
    alu #(.XLEN(XLEN)) u_alu (
        .a(alu_src1),
        .b(alu_src2),
        .alu_op(alu_op_in),
        .result(alu_result),
        .zero(alu_zero),
        .lt(alu_lt),
        .ltu(alu_ltu)
    );
    
    // MulDiv unit
    logic [XLEN-1:0] muldiv_result;
    logic             muldiv_valid;
    logic [2:0]       muldiv_op;
    
    // Map funct3 to muldiv operation
    assign muldiv_op = funct3_in;
    
    muldiv #(.XLEN(XLEN)) u_muldiv (
        .clk(clk),
        .rst(rst),
        .start(mul_start_in || div_start_in),
        .op(muldiv_op),
        .a(forwarded_rs1),
        .b(forwarded_rs2),
        .result(muldiv_result),
        .busy(muldiv_busy),
        .valid(muldiv_valid)
    );
    
    // Branch resolution
    logic branch_condition;
    
    always_comb begin
        branch_condition = 1'b0;
        
        if (branch_in) begin
            case (funct3_in)
                3'b000: branch_condition = alu_zero;        // BEQ
                3'b001: branch_condition = !alu_zero;       // BNE
                3'b100: branch_condition = alu_lt;          // BLT
                3'b101: branch_condition = !alu_lt;         // BGE
                3'b110: branch_condition = alu_ltu;         // BLTU
                3'b111: branch_condition = !alu_ltu;        // BGEU
                default: branch_condition = 1'b0;
            endcase
        end
    end
    
    // Branch/Jump target calculation
    always_comb begin
        branch_taken = 1'b0;
        branch_target = '0;
        
        if (jal_in) begin
            branch_taken = 1'b1;
            branch_target = pc_in + imm_in;
        end else if (jalr_in) begin
            branch_taken = 1'b1;
            branch_target = (forwarded_rs1 + imm_in) & ~32'h1;  // Clear LSB
        end else if (branch_in && branch_condition) begin
            branch_taken = 1'b1;
            branch_target = pc_in + imm_in;
        end
    end
    
    // Result selection
    logic [XLEN-1:0] ex_result;
    
    always_comb begin
        if (mul_start_in || div_start_in) begin
            ex_result = muldiv_result;
        end else if (wb_sel_in == 2'b10) begin
            ex_result = pc_plus4;  // For JAL/JALR
        end else begin
            ex_result = alu_result;
        end
    end
    
    // Pipeline registers
    always_ff @(posedge clk) begin
        if (rst || flush) begin
            alu_result_out <= '0;
            rs2_data_out <= '0;
            rd_out <= '0;
            funct3_out <= '0;
            valid_out <= 1'b0;
            mem_read_out <= 1'b0;
            mem_write_out <= 1'b0;
            reg_write_out <= 1'b0;
            wb_sel_out <= 2'b00;
        end else if (!stall) begin
            alu_result_out <= ex_result;
            rs2_data_out <= forwarded_rs2;
            rd_out <= rd_in;
            funct3_out <= funct3_in;
            valid_out <= valid_in && !branch_taken;  // Invalidate if branch taken
            mem_read_out <= mem_read_in;
            mem_write_out <= mem_write_in;
            reg_write_out <= reg_write_in;
            wb_sel_out <= wb_sel_in;
        end
    end

endmodule