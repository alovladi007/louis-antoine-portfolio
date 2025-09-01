`timescale 1ns/1ps

interface axi_lite_if #(
    parameter AW = 32,
    parameter DW = 32
)(
    input logic ACLK,
    input logic ARESETn
);

    // Write Address Channel
    logic [AW-1:0] AWADDR;
    logic          AWVALID;
    logic          AWREADY;
    
    // Write Data Channel
    logic [DW-1:0]   WDATA;
    logic [DW/8-1:0] WSTRB;
    logic            WVALID;
    logic            WREADY;
    
    // Write Response Channel
    logic [1:0] BRESP;
    logic       BVALID;
    logic       BREADY;
    
    // Read Address Channel
    logic [AW-1:0] ARADDR;
    logic          ARVALID;
    logic          ARREADY;
    
    // Read Data Channel
    logic [DW-1:0] RDATA;
    logic [1:0]    RRESP;
    logic          RVALID;
    logic          RREADY;
    
    // Master modport
    modport master (
        input  ACLK, ARESETn,
        output AWADDR, AWVALID, input AWREADY,
        output WDATA, WSTRB, WVALID, input WREADY,
        input  BRESP, BVALID, output BREADY,
        output ARADDR, ARVALID, input ARREADY,
        input  RDATA, RRESP, RVALID, output RREADY
    );
    
    // Slave modport
    modport slave (
        input  ACLK, ARESETn,
        input  AWADDR, AWVALID, output AWREADY,
        input  WDATA, WSTRB, WVALID, output WREADY,
        output BRESP, BVALID, input BREADY,
        input  ARADDR, ARVALID, output ARREADY,
        output RDATA, RRESP, RVALID, input RREADY
    );

endinterface