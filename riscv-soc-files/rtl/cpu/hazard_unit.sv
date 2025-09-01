`timescale 1ns/1ps

module hazard_unit (
    // From ID stage
    input  logic       id_rs1_valid,
    input  logic       id_rs2_valid,
    input  logic [4:0] id_rs1,
    input  logic [4:0] id_rs2,
    
    // From EX stage
    input  logic       ex_rd_valid,
    input  logic [4:0] ex_rd,
    input  logic       ex_mem_read,
    
    // From MEM stage
    input  logic       mem_rd_valid,
    input  logic [4:0] mem_rd,
    
    // From WB stage
    input  logic       wb_rd_valid,
    input  logic [4:0] wb_rd,
    
    // MulDiv busy
    input  logic       muldiv_busy,
    
    // Control outputs
    output logic       stall_if,
    output logic       stall_id,
    output logic       stall_ex,
    output logic       flush_id,
    output logic       flush_ex,
    
    // Forwarding control
    output logic [1:0] fwd_rs1_sel,  // 00: ID, 01: EX/MEM, 10: MEM/WB
    output logic [1:0] fwd_rs2_sel
);

    // Load-use hazard detection
    logic load_use_hazard;
    
    always_comb begin
        load_use_hazard = 1'b0;
        
        if (ex_mem_read && ex_rd_valid && (ex_rd != 5'd0)) begin
            if ((id_rs1_valid && (id_rs1 == ex_rd)) ||
                (id_rs2_valid && (id_rs2 == ex_rd))) begin
                load_use_hazard = 1'b1;
            end
        end
    end
    
    // Forwarding logic for RS1
    always_comb begin
        fwd_rs1_sel = 2'b00;  // Default: no forwarding
        
        if (id_rs1_valid && (id_rs1 != 5'd0)) begin
            // Priority: EX/MEM > MEM/WB
            if (ex_rd_valid && (ex_rd == id_rs1) && !ex_mem_read) begin
                fwd_rs1_sel = 2'b01;  // Forward from EX/MEM
            end else if (mem_rd_valid && (mem_rd == id_rs1)) begin
                fwd_rs1_sel = 2'b10;  // Forward from MEM/WB
            end
        end
    end
    
    // Forwarding logic for RS2
    always_comb begin
        fwd_rs2_sel = 2'b00;  // Default: no forwarding
        
        if (id_rs2_valid && (id_rs2 != 5'd0)) begin
            // Priority: EX/MEM > MEM/WB
            if (ex_rd_valid && (ex_rd == id_rs2) && !ex_mem_read) begin
                fwd_rs2_sel = 2'b01;  // Forward from EX/MEM
            end else if (mem_rd_valid && (mem_rd == id_rs2)) begin
                fwd_rs2_sel = 2'b10;  // Forward from MEM/WB
            end
        end
    end
    
    // Stall and flush control
    assign stall_if = load_use_hazard || muldiv_busy;
    assign stall_id = load_use_hazard || muldiv_busy;
    assign stall_ex = muldiv_busy;
    assign flush_id = 1'b0;  // Will be set by branch predictor
    assign flush_ex = 1'b0;  // Will be set by branch predictor

endmodule