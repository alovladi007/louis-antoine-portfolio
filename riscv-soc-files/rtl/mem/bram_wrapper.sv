`timescale 1ns/1ps

module bram_wrapper #(
    parameter ADDR_WIDTH = 15,  // 32KB = 2^15 bytes
    parameter DATA_WIDTH = 32,
    parameter INIT_FILE = ""
)(
    input  logic                     clk,
    input  logic                     rst,
    
    // Port A - Instruction fetch
    input  logic                     ena,
    input  logic [3:0]               wea,
    input  logic [ADDR_WIDTH-1:0]    addra,
    input  logic [DATA_WIDTH-1:0]    dina,
    output logic [DATA_WIDTH-1:0]    douta,
    
    // Port B - Data access
    input  logic                     enb,
    input  logic [3:0]               web,
    input  logic [ADDR_WIDTH-1:0]    addrb,
    input  logic [DATA_WIDTH-1:0]    dinb,
    output logic [DATA_WIDTH-1:0]    doutb
);

    // Memory array
    logic [7:0] mem [0:(1<<ADDR_WIDTH)-1];
    
    // Initialize memory if file provided
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end else begin
            for (int i = 0; i < (1<<ADDR_WIDTH); i++) begin
                mem[i] = 8'h00;
            end
        end
    end
    
    // Port A - Read/Write
    always_ff @(posedge clk) begin
        if (ena) begin
            // Byte-wise write
            if (wea[0]) mem[addra + 0] <= dina[7:0];
            if (wea[1]) mem[addra + 1] <= dina[15:8];
            if (wea[2]) mem[addra + 2] <= dina[23:16];
            if (wea[3]) mem[addra + 3] <= dina[31:24];
            
            // Read (write-first mode)
            douta <= {mem[addra + 3], mem[addra + 2], mem[addra + 1], mem[addra + 0]};
        end
    end
    
    // Port B - Read/Write
    always_ff @(posedge clk) begin
        if (enb) begin
            // Byte-wise write
            if (web[0]) mem[addrb + 0] <= dinb[7:0];
            if (web[1]) mem[addrb + 1] <= dinb[15:8];
            if (web[2]) mem[addrb + 2] <= dinb[23:16];
            if (web[3]) mem[addrb + 3] <= dinb[31:24];
            
            // Read (write-first mode)
            doutb <= {mem[addrb + 3], mem[addrb + 2], mem[addrb + 1], mem[addrb + 0]};
        end
    end

endmodule