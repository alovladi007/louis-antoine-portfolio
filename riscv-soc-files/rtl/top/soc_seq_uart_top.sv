// SoC top with AXI-Lite micro-sequencer that prints "HELLO\n" at boot
`timescale 1ns/1ps

module soc_seq_uart_top #(
    parameter AW = 32,
    parameter DW = 32
)(
    input  logic clk_50mhz,
    input  logic rst_btn,
    output logic uart_txd
);

    // Reset synchronization
    logic rst_n;
    logic [3:0] rst_sync;
    
    always_ff @(posedge clk_50mhz or posedge rst_btn) begin
        if (rst_btn) 
            rst_sync <= 4'hF;
        else 
            rst_sync <= {rst_sync[2:0], 1'b0};
    end
    
    assign rst_n = ~rst_sync[3];

    // AXI-Lite master signals from sequencer
    logic [AW-1:0] M_AWADDR; 
    logic M_AWVALID; 
    logic M_AWREADY;
    logic [DW-1:0] M_WDATA;  
    logic [DW/8-1:0] M_WSTRB; 
    logic M_WVALID; 
    logic M_WREADY;
    logic [1:0] M_BRESP;  
    logic M_BVALID; 
    logic M_BREADY;
    logic [AW-1:0] M_ARADDR; 
    logic M_ARVALID; 
    logic M_ARREADY;
    logic [DW-1:0] M_RDATA;  
    logic [1:0] M_RRESP;  
    logic M_RVALID; 
    logic M_RREADY;

    // UART instance (simplified direct connection)
    axi_uart_tx #(
        .AW(AW), 
        .DW(DW), 
        .CLK_HZ(50_000_000), 
        .BAUD(115200)
    ) UART0 (
        .ACLK(clk_50mhz), 
        .ARESETn(rst_n),
        .S_AWADDR(M_AWADDR), 
        .S_AWVALID(M_AWVALID), 
        .S_AWREADY(M_AWREADY),
        .S_WDATA(M_WDATA), 
        .S_WSTRB(M_WSTRB), 
        .S_WVALID(M_WVALID), 
        .S_WREADY(M_WREADY),
        .S_BRESP(M_BRESP), 
        .S_BVALID(M_BVALID), 
        .S_BREADY(M_BREADY),
        .S_ARADDR(M_ARADDR), 
        .S_ARVALID(M_ARVALID), 
        .S_ARREADY(M_ARREADY),
        .S_RDATA(M_RDATA), 
        .S_RRESP(M_RRESP), 
        .S_RVALID(M_RVALID), 
        .S_RREADY(M_RREADY),
        .txd(uart_txd)
    );

    // Micro-sequencer that prints "HELLO\n"
    axi_seq_hello #(
        .AW(AW), 
        .DW(DW), 
        .UART_BASE(32'h4000_0000), 
        .STRLEN(6)
    ) SEQ (
        .ACLK(clk_50mhz), 
        .ARESETn(rst_n),
        .M_AWADDR(M_AWADDR), 
        .M_AWVALID(M_AWVALID), 
        .M_AWREADY(M_AWREADY),
        .M_WDATA(M_WDATA), 
        .M_WSTRB(M_WSTRB), 
        .M_WVALID(M_WVALID), 
        .M_WREADY(M_WREADY),
        .M_BRESP(M_BRESP), 
        .M_BVALID(M_BVALID), 
        .M_BREADY(M_BREADY),
        .M_ARADDR(M_ARADDR), 
        .M_ARVALID(M_ARVALID), 
        .M_ARREADY(M_ARREADY),
        .M_RDATA(M_RDATA), 
        .M_RRESP(M_RRESP), 
        .M_RVALID(M_RVALID), 
        .M_RREADY(M_RREADY)
    );

endmodule