`timescale 1ns/1ps

module bram_init #(
    parameter AW = 14,        // Address width (16KB = 2^14 bytes)
    parameter DW = 32,        // Data width
    parameter INIT_FILE = ""  // Hex file to initialize memory
)(
    input  logic             clk,
    input  logic             en,
    input  logic [3:0]       we,
    input  logic [AW-1:0]    addr,
    input  logic [DW-1:0]    din,
    output logic [DW-1:0]    dout
);

    // Memory array (word-addressed)
    logic [DW-1:0] mem [0:(1<<(AW-2))-1];
    
    // Initialize memory from hex file if provided
    initial begin
        if (INIT_FILE != "") begin
            $display("Loading ROM from %s", INIT_FILE);
            $readmemh(INIT_FILE, mem);
        end else begin
            // Default: Initialize with NOPs
            for (int i = 0; i < (1<<(AW-2)); i++) begin
                mem[i] = 32'h00000013; // NOP (addi x0, x0, 0)
            end
        end
    end
    
    // Word-aligned address
    logic [AW-3:0] word_addr;
    assign word_addr = addr[AW-1:2];
    
    // Synchronous read/write
    always_ff @(posedge clk) begin
        if (en) begin
            // Byte-wise write enable
            if (we[0]) mem[word_addr][7:0]   <= din[7:0];
            if (we[1]) mem[word_addr][15:8]  <= din[15:8];
            if (we[2]) mem[word_addr][23:16] <= din[23:16];
            if (we[3]) mem[word_addr][31:24] <= din[31:24];
            
            // Read (write-first mode)
            dout <= mem[word_addr];
        end
    end

endmodule