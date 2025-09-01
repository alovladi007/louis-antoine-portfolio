`timescale 1ns/1ps

module mem_stage #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // Control signals
    input  logic             stall,
    input  logic             flush,
    
    // From EX stage
    input  logic [XLEN-1:0]  alu_result_in,
    input  logic [XLEN-1:0]  rs2_data_in,
    input  logic [4:0]       rd_in,
    input  logic [2:0]       funct3_in,
    input  logic             valid_in,
    input  logic             mem_read_in,
    input  logic             mem_write_in,
    input  logic             reg_write_in,
    input  logic [1:0]       wb_sel_in,
    
    // Data memory interface
    output logic [XLEN-1:0]  dmem_addr,
    output logic [XLEN-1:0]  dmem_wdata,
    output logic [3:0]       dmem_we,
    output logic             dmem_req,
    input  logic [XLEN-1:0]  dmem_rdata,
    input  logic             dmem_ready,
    
    // To WB stage
    output logic [XLEN-1:0]  alu_result_out,
    output logic [XLEN-1:0]  mem_data_out,
    output logic [4:0]       rd_out,
    output logic             valid_out,
    output logic             reg_write_out,
    output logic [1:0]       wb_sel_out
);

    // Memory address and data
    assign dmem_addr = alu_result_in;
    assign dmem_req = (mem_read_in || mem_write_in) && valid_in && !stall;
    
    // Store data alignment and byte enables
    always_comb begin
        dmem_wdata = '0;
        dmem_we = 4'b0000;
        
        if (mem_write_in) begin
            case (funct3_in)
                3'b000: begin  // SB (Store Byte)
                    case (alu_result_in[1:0])
                        2'b00: begin
                            dmem_wdata = {24'h0, rs2_data_in[7:0]};
                            dmem_we = 4'b0001;
                        end
                        2'b01: begin
                            dmem_wdata = {16'h0, rs2_data_in[7:0], 8'h0};
                            dmem_we = 4'b0010;
                        end
                        2'b10: begin
                            dmem_wdata = {8'h0, rs2_data_in[7:0], 16'h0};
                            dmem_we = 4'b0100;
                        end
                        2'b11: begin
                            dmem_wdata = {rs2_data_in[7:0], 24'h0};
                            dmem_we = 4'b1000;
                        end
                    endcase
                end
                
                3'b001: begin  // SH (Store Halfword)
                    case (alu_result_in[1])
                        1'b0: begin
                            dmem_wdata = {16'h0, rs2_data_in[15:0]};
                            dmem_we = 4'b0011;
                        end
                        1'b1: begin
                            dmem_wdata = {rs2_data_in[15:0], 16'h0};
                            dmem_we = 4'b1100;
                        end
                    endcase
                end
                
                3'b010: begin  // SW (Store Word)
                    dmem_wdata = rs2_data_in;
                    dmem_we = 4'b1111;
                end
                
                default: begin
                    dmem_wdata = rs2_data_in;
                    dmem_we = 4'b0000;
                end
            endcase
        end
    end
    
    // Load data alignment
    logic [XLEN-1:0] aligned_mem_data;
    
    always_comb begin
        aligned_mem_data = '0;
        
        if (mem_read_in) begin
            case (funct3_in)
                3'b000: begin  // LB (Load Byte)
                    case (alu_result_in[1:0])
                        2'b00: aligned_mem_data = {{24{dmem_rdata[7]}}, dmem_rdata[7:0]};
                        2'b01: aligned_mem_data = {{24{dmem_rdata[15]}}, dmem_rdata[15:8]};
                        2'b10: aligned_mem_data = {{24{dmem_rdata[23]}}, dmem_rdata[23:16]};
                        2'b11: aligned_mem_data = {{24{dmem_rdata[31]}}, dmem_rdata[31:24]};
                    endcase
                end
                
                3'b001: begin  // LH (Load Halfword)
                    case (alu_result_in[1])
                        1'b0: aligned_mem_data = {{16{dmem_rdata[15]}}, dmem_rdata[15:0]};
                        1'b1: aligned_mem_data = {{16{dmem_rdata[31]}}, dmem_rdata[31:16]};
                    endcase
                end
                
                3'b010: begin  // LW (Load Word)
                    aligned_mem_data = dmem_rdata;
                end
                
                3'b100: begin  // LBU (Load Byte Unsigned)
                    case (alu_result_in[1:0])
                        2'b00: aligned_mem_data = {24'h0, dmem_rdata[7:0]};
                        2'b01: aligned_mem_data = {24'h0, dmem_rdata[15:8]};
                        2'b10: aligned_mem_data = {24'h0, dmem_rdata[23:16]};
                        2'b11: aligned_mem_data = {24'h0, dmem_rdata[31:24]};
                    endcase
                end
                
                3'b101: begin  // LHU (Load Halfword Unsigned)
                    case (alu_result_in[1])
                        1'b0: aligned_mem_data = {16'h0, dmem_rdata[15:0]};
                        1'b1: aligned_mem_data = {16'h0, dmem_rdata[31:16]};
                    endcase
                end
                
                default: aligned_mem_data = dmem_rdata;
            endcase
        end else begin
            aligned_mem_data = dmem_rdata;
        end
    end
    
    // Pipeline registers
    always_ff @(posedge clk) begin
        if (rst || flush) begin
            alu_result_out <= '0;
            mem_data_out <= '0;
            rd_out <= '0;
            valid_out <= 1'b0;
            reg_write_out <= 1'b0;
            wb_sel_out <= 2'b00;
        end else if (!stall) begin
            alu_result_out <= alu_result_in;
            mem_data_out <= aligned_mem_data;
            rd_out <= rd_in;
            valid_out <= valid_in;
            reg_write_out <= reg_write_in;
            wb_sel_out <= wb_sel_in;
        end
    end

endmodule