// AXI-Lite Micro-Sequencer that prints a string by polling UART STATUS then writing TXDATA
`timescale 1ns/1ps

module axi_seq_hello #(
    parameter AW = 32,
    parameter DW = 32,
    parameter UART_BASE = 32'h4000_0000,
    parameter integer STRLEN = 6
)(
    input  logic             ACLK,
    input  logic             ARESETn,
    
    // AXI-Lite master interface
    output logic [AW-1:0]    M_AWADDR,
    output logic             M_AWVALID,
    input  logic             M_AWREADY,
    output logic [DW-1:0]    M_WDATA,
    output logic [DW/8-1:0]  M_WSTRB,
    output logic             M_WVALID,
    input  logic             M_WREADY,
    input  logic [1:0]       M_BRESP,
    input  logic             M_BVALID,
    output logic             M_BREADY,
    output logic [AW-1:0]    M_ARADDR,
    output logic             M_ARVALID,
    input  logic             M_ARREADY,
    input  logic [DW-1:0]    M_RDATA,
    input  logic [1:0]       M_RRESP,
    input  logic             M_RVALID,
    output logic             M_RREADY
);

    // ROM string storage
    byte rom [0:STRLEN-1];
    initial begin
        rom[0] = "H";
        rom[1] = "E";
        rom[2] = "L";
        rom[3] = "L";
        rom[4] = "O";
        rom[5] = "\n";
    end

    // State machine
    typedef enum logic [2:0] {
        IDLE, 
        POLL_REQ, 
        POLL_WAIT, 
        WRITE_ADDR, 
        WRITE_DATA, 
        WRITE_RESP, 
        NEXT_CHAR, 
        DONE
    } state_t;
    
    state_t state;
    integer idx;

    // Default master signals
    always_comb begin
        M_AWADDR  = '0; 
        M_AWVALID = 1'b0;
        M_WDATA   = '0; 
        M_WSTRB   = 4'h0; 
        M_WVALID  = 1'b0;
        M_BREADY  = 1'b0;
        M_ARADDR  = '0; 
        M_ARVALID = 1'b0;
        M_RREADY  = 1'b0;

        case (state)
            POLL_REQ: begin
                // Read STATUS @ UART_BASE+0x08
                M_ARADDR  = UART_BASE + 32'h8;
                M_ARVALID = 1'b1;
            end
            
            POLL_WAIT: begin
                M_RREADY = 1'b1;
            end
            
            WRITE_ADDR: begin
                // Write to TX_DATA @ UART_BASE+0x00
                M_AWADDR  = UART_BASE + 32'h0;
                M_AWVALID = 1'b1;
            end
            
            WRITE_DATA: begin
                M_WDATA  = {24'h0, rom[idx]};
                M_WSTRB  = 4'h1;
                M_WVALID = 1'b1;
            end
            
            WRITE_RESP: begin
                M_BREADY = 1'b1;
            end
            
            default: ;
        endcase
    end

    // State machine logic
    always_ff @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            state <= IDLE; 
            idx <= 0;
        end else begin
            case (state)
                IDLE: begin
                    state <= POLL_REQ;
                end
                
                POLL_REQ: begin
                    if (M_ARREADY) 
                        state <= POLL_WAIT;
                end
                
                POLL_WAIT: begin
                    if (M_RVALID) begin
                        // STATUS[0] = tx_busy
                        if (M_RDATA[0] == 1'b0) 
                            state <= WRITE_ADDR; 
                        else 
                            state <= POLL_REQ;
                    end
                end
                
                WRITE_ADDR: begin
                    if (M_AWREADY) 
                        state <= WRITE_DATA;
                end
                
                WRITE_DATA: begin
                    if (M_WREADY)  
                        state <= WRITE_RESP;
                end
                
                WRITE_RESP: begin
                    if (M_BVALID)  
                        state <= NEXT_CHAR;
                end
                
                NEXT_CHAR: begin
                    if (idx == STRLEN-1) 
                        state <= DONE;
                    else begin
                        idx <= idx + 1;
                        state <= POLL_REQ;
                    end
                end
                
                DONE: begin
                    state <= DONE;
                end
            endcase
        end
    end

endmodule