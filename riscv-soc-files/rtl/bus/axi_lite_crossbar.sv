`timescale 1ns/1ps

module axi_lite_crossbar #(
    parameter AW = 32,
    parameter DW = 32,
    parameter NUM_SLAVES = 8,
    parameter logic [AW-1:0] BASE_ADDR [NUM_SLAVES] = '{
        32'h4000_0000,  // UART
        32'h4001_0000,  // SPI
        32'h4002_0000,  // I2C
        32'h4003_0000,  // GPIO
        32'h4004_0000,  // Timer/PWM
        32'h2000_0000,  // CLINT
        32'h2001_0000,  // PLIC
        32'h1000_0000   // SRAM
    },
    parameter logic [AW-1:0] ADDR_MASK [NUM_SLAVES] = '{
        32'hFFFF_F000,  // 4KB regions
        32'hFFFF_F000,
        32'hFFFF_F000,
        32'hFFFF_F000,
        32'hFFFF_F000,
        32'hFFFF_F000,
        32'hFFF0_0000,  // 64KB for PLIC
        32'hFFF0_0000   // 128KB for SRAM
    }
)(
    input logic ACLK,
    input logic ARESETn,
    
    // Master interface
    axi_lite_if.slave m_axi,
    
    // Slave interfaces
    axi_lite_if.master s_axi[NUM_SLAVES]
);

    // Decode function
    function automatic int decode_addr(input logic [AW-1:0] addr);
        for (int i = 0; i < NUM_SLAVES; i++) begin
            if ((addr & ADDR_MASK[i]) == BASE_ADDR[i]) begin
                return i;
            end
        end
        return -1;  // No match
    endfunction
    
    // Write channel routing
    logic [NUM_SLAVES-1:0] aw_select;
    logic [NUM_SLAVES-1:0] w_select;
    logic [NUM_SLAVES-1:0] b_select;
    int aw_slave_idx;
    int w_slave_idx;
    int b_slave_idx;
    
    // Read channel routing
    logic [NUM_SLAVES-1:0] ar_select;
    logic [NUM_SLAVES-1:0] r_select;
    int ar_slave_idx;
    int r_slave_idx;
    
    // Write address channel decoding
    always_ff @(posedge ACLK) begin
        if (!ARESETn) begin
            aw_slave_idx <= -1;
            aw_select <= '0;
        end else if (m_axi.AWVALID && m_axi.AWREADY) begin
            aw_slave_idx <= decode_addr(m_axi.AWADDR);
            for (int i = 0; i < NUM_SLAVES; i++) begin
                aw_select[i] <= (decode_addr(m_axi.AWADDR) == i);
            end
        end
    end
    
    // Write data channel follows write address
    always_ff @(posedge ACLK) begin
        if (!ARESETn) begin
            w_select <= '0;
        end else begin
            w_select <= aw_select;
        end
    end
    
    // Write response channel
    always_ff @(posedge ACLK) begin
        if (!ARESETn) begin
            b_select <= '0;
        end else begin
            b_select <= w_select;
        end
    end
    
    // Read address channel decoding
    always_ff @(posedge ACLK) begin
        if (!ARESETn) begin
            ar_slave_idx <= -1;
            ar_select <= '0;
        end else if (m_axi.ARVALID && m_axi.ARREADY) begin
            ar_slave_idx <= decode_addr(m_axi.ARADDR);
            for (int i = 0; i < NUM_SLAVES; i++) begin
                ar_select[i] <= (decode_addr(m_axi.ARADDR) == i);
            end
        end
    end
    
    // Read data channel follows read address
    always_ff @(posedge ACLK) begin
        if (!ARESETn) begin
            r_select <= '0;
        end else begin
            r_select <= ar_select;
        end
    end
    
    // Connect master to slaves
    always_comb begin
        // Default values
        m_axi.AWREADY = 1'b0;
        m_axi.WREADY = 1'b0;
        m_axi.BRESP = 2'b00;
        m_axi.BVALID = 1'b0;
        m_axi.ARREADY = 1'b0;
        m_axi.RDATA = '0;
        m_axi.RRESP = 2'b00;
        m_axi.RVALID = 1'b0;
        
        for (int i = 0; i < NUM_SLAVES; i++) begin
            // Write address channel
            s_axi[i].AWADDR = m_axi.AWADDR;
            s_axi[i].AWVALID = m_axi.AWVALID && aw_select[i];
            
            // Write data channel
            s_axi[i].WDATA = m_axi.WDATA;
            s_axi[i].WSTRB = m_axi.WSTRB;
            s_axi[i].WVALID = m_axi.WVALID && w_select[i];
            
            // Write response channel
            s_axi[i].BREADY = m_axi.BREADY && b_select[i];
            
            // Read address channel
            s_axi[i].ARADDR = m_axi.ARADDR;
            s_axi[i].ARVALID = m_axi.ARVALID && ar_select[i];
            
            // Read data channel
            s_axi[i].RREADY = m_axi.RREADY && r_select[i];
            
            // Collect responses
            if (aw_select[i]) m_axi.AWREADY |= s_axi[i].AWREADY;
            if (w_select[i]) m_axi.WREADY |= s_axi[i].WREADY;
            if (b_select[i]) begin
                m_axi.BRESP = s_axi[i].BRESP;
                m_axi.BVALID = s_axi[i].BVALID;
            end
            if (ar_select[i]) m_axi.ARREADY |= s_axi[i].ARREADY;
            if (r_select[i]) begin
                m_axi.RDATA = s_axi[i].RDATA;
                m_axi.RRESP = s_axi[i].RRESP;
                m_axi.RVALID = s_axi[i].RVALID;
            end
        end
    end

endmodule