`timescale 1ns/1ps

module soc_top #(
    parameter CLK_FREQ = 50_000_000,
    parameter BAUD_RATE = 115200,
    parameter SRAM_SIZE = 128*1024,  // 128KB
    parameter BOOT_ADDR = 32'h0000_0000
)(
    input  logic clk_in,
    input  logic rst_in,
    
    // UART
    output logic uart_tx,
    input  logic uart_rx,
    
    // SPI
    output logic spi_sclk,
    output logic spi_mosi,
    input  logic spi_miso,
    output logic [3:0] spi_cs_n,
    
    // GPIO
    input  logic [15:0] gpio_in,
    output logic [15:0] gpio_out,
    output logic [15:0] gpio_oe,
    
    // PWM
    output logic [3:0] pwm_out,
    
    // Debug LEDs
    output logic [7:0] led,
    
    // JTAG (optional)
    input  logic tck,
    input  logic tms,
    input  logic tdi,
    output logic tdo
);

    // Internal signals
    logic clk;
    logic rst;
    logic locked;
    
    // Clock and reset management
    clk_rst_gen #(
        .CLK_FREQ(CLK_FREQ)
    ) u_clk_rst (
        .clk_in(clk_in),
        .rst_in(rst_in),
        .clk_out(clk),
        .rst_out(rst),
        .locked(locked)
    );
    
    // CPU instruction memory interface
    logic [31:0] imem_addr;
    logic        imem_req;
    logic [31:0] imem_rdata;
    logic        imem_ready;
    
    // CPU data memory interface
    logic [31:0] dmem_addr;
    logic [31:0] dmem_wdata;
    logic [3:0]  dmem_we;
    logic        dmem_req;
    logic [31:0] dmem_rdata;
    logic        dmem_ready;
    
    // AXI-Lite interfaces
    axi_lite_if #(.AW(32), .DW(32)) cpu_axi (.ACLK(clk), .ARESETn(!rst));
    axi_lite_if #(.AW(32), .DW(32)) periph_axi[8] (.ACLK(clk), .ARESETn(!rst));
    
    // Interrupt signals
    logic timer_irq;
    logic uart_irq;
    logic spi_irq;
    logic gpio_irq;
    logic external_irq;
    
    // RISC-V CPU Core
    riscv_cpu #(
        .XLEN(32),
        .RESET_ADDR(BOOT_ADDR)
    ) u_cpu (
        .clk(clk),
        .rst(rst),
        .imem_addr(imem_addr),
        .imem_req(imem_req),
        .imem_rdata(imem_rdata),
        .imem_ready(imem_ready),
        .dmem_addr(dmem_addr),
        .dmem_wdata(dmem_wdata),
        .dmem_we(dmem_we),
        .dmem_req(dmem_req),
        .dmem_rdata(dmem_rdata),
        .dmem_ready(dmem_ready),
        .timer_irq(timer_irq),
        .external_irq(external_irq),
        .debug_pc(),
        .debug_instr(),
        .debug_valid()
    );
    
    // Memory subsystem (simplified - direct connection to BRAM)
    logic [14:0] bram_addra, bram_addrb;
    logic [31:0] bram_dina, bram_dinb;
    logic [31:0] bram_douta, bram_doutb;
    logic [3:0]  bram_wea, bram_web;
    logic        bram_ena, bram_enb;
    
    // Instruction memory interface
    assign bram_addra = imem_addr[16:2];  // Word addressing
    assign bram_dina = 32'h0;
    assign bram_wea = 4'b0000;  // Read-only for instruction port
    assign bram_ena = imem_req;
    assign imem_rdata = bram_douta;
    assign imem_ready = imem_req;  // Single-cycle memory
    
    // Data memory interface (simplified - direct to BRAM for addresses < 0x20000)
    logic is_bram_access;
    assign is_bram_access = (dmem_addr < 32'h0002_0000);
    
    assign bram_addrb = dmem_addr[16:2];  // Word addressing
    assign bram_dinb = dmem_wdata;
    assign bram_web = is_bram_access ? dmem_we : 4'b0000;
    assign bram_enb = dmem_req && is_bram_access;
    
    // BRAM instance (128KB)
    bram_wrapper #(
        .ADDR_WIDTH(15),  // 32KB addressable as words
        .DATA_WIDTH(32),
        .INIT_FILE("")    // Can specify initialization file
    ) u_bram (
        .clk(clk),
        .rst(rst),
        .ena(bram_ena),
        .wea(bram_wea),
        .addra(bram_addra),
        .dina(bram_dina),
        .douta(bram_douta),
        .enb(bram_enb),
        .web(bram_web),
        .addrb(bram_addrb),
        .dinb(bram_dinb),
        .doutb(bram_doutb)
    );
    
    // CPU to AXI-Lite bridge (simplified)
    cpu_to_axi_bridge u_bridge (
        .clk(clk),
        .rst(rst),
        .dmem_addr(dmem_addr),
        .dmem_wdata(dmem_wdata),
        .dmem_we(dmem_we),
        .dmem_req(dmem_req && !is_bram_access),
        .dmem_rdata(dmem_rdata),
        .dmem_ready(dmem_ready),
        .bram_rdata(bram_doutb),
        .bram_ready(is_bram_access),
        .m_axi(cpu_axi)
    );
    
    // AXI-Lite Crossbar
    axi_lite_crossbar #(
        .AW(32),
        .DW(32),
        .NUM_SLAVES(8)
    ) u_xbar (
        .ACLK(clk),
        .ARESETn(!rst),
        .m_axi(cpu_axi),
        .s_axi(periph_axi)
    );
    
    // UART Peripheral
    uart #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(BAUD_RATE)
    ) u_uart (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[0]),
        .uart_tx(uart_tx),
        .uart_rx(uart_rx),
        .irq(uart_irq)
    );
    
    // SPI Peripheral
    spi #(
        .CLK_FREQ(CLK_FREQ),
        .SPI_FREQ(10_000_000)
    ) u_spi (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[1]),
        .spi_sclk(spi_sclk),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        .irq(spi_irq)
    );
    
    // I2C Peripheral (stub for now)
    axi_lite_stub u_i2c (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[2])
    );
    
    // GPIO Peripheral
    gpio #(
        .NUM_GPIO(16)
    ) u_gpio (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[3]),
        .gpio_in(gpio_in),
        .gpio_out(gpio_out),
        .gpio_oe(gpio_oe),
        .irq(gpio_irq)
    );
    
    // Timer/PWM Peripheral
    timer #(
        .CLK_FREQ(CLK_FREQ)
    ) u_timer (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[4]),
        .pwm_out(pwm_out),
        .irq(timer_irq)
    );
    
    // CLINT (stub for now)
    axi_lite_stub u_clint (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[5])
    );
    
    // PLIC (stub for now)
    axi_lite_stub u_plic (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[6])
    );
    
    // Reserved slot
    axi_lite_stub u_reserved (
        .clk(clk),
        .rst(rst),
        .s_axi(periph_axi[7])
    );
    
    // Interrupt aggregation
    assign external_irq = uart_irq | spi_irq | gpio_irq;
    
    // Debug LEDs
    assign led[0] = !rst;
    assign led[1] = locked;
    assign led[2] = uart_irq;
    assign led[3] = spi_irq;
    assign led[4] = gpio_irq;
    assign led[5] = timer_irq;
    assign led[6] = imem_req;
    assign led[7] = dmem_req;

endmodule

// Clock and reset generator
module clk_rst_gen #(
    parameter CLK_FREQ = 50_000_000
)(
    input  logic clk_in,
    input  logic rst_in,
    output logic clk_out,
    output logic rst_out,
    output logic locked
);

    // For simulation and simple implementation, pass through
    // In real implementation, use PLL/MMCM
    assign clk_out = clk_in;
    assign locked = !rst_in;
    
    // Synchronous reset generation
    logic [3:0] rst_sync;
    
    always_ff @(posedge clk_in) begin
        if (rst_in) begin
            rst_sync <= 4'hF;
        end else begin
            rst_sync <= {rst_sync[2:0], 1'b0};
        end
    end
    
    assign rst_out = rst_sync[3];

endmodule

// CPU to AXI-Lite bridge
module cpu_to_axi_bridge (
    input  logic        clk,
    input  logic        rst,
    
    // CPU data memory interface
    input  logic [31:0] dmem_addr,
    input  logic [31:0] dmem_wdata,
    input  logic [3:0]  dmem_we,
    input  logic        dmem_req,
    output logic [31:0] dmem_rdata,
    output logic        dmem_ready,
    
    // BRAM interface
    input  logic [31:0] bram_rdata,
    input  logic        bram_ready,
    
    // AXI-Lite master interface
    axi_lite_if.master m_axi
);

    typedef enum logic [2:0] {IDLE, AW_W, B_WAIT, AR, R_WAIT} state_t;
    state_t state;
    
    logic [31:0] rdata_reg;
    logic is_write;
    
    assign is_write = |dmem_we;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            rdata_reg <= '0;
            dmem_ready <= 1'b0;
            
            m_axi.AWADDR <= '0;
            m_axi.AWVALID <= 1'b0;
            m_axi.WDATA <= '0;
            m_axi.WSTRB <= '0;
            m_axi.WVALID <= 1'b0;
            m_axi.BREADY <= 1'b0;
            m_axi.ARADDR <= '0;
            m_axi.ARVALID <= 1'b0;
            m_axi.RREADY <= 1'b0;
        end else begin
            dmem_ready <= 1'b0;
            
            case (state)
                IDLE: begin
                    if (dmem_req) begin
                        if (bram_ready) begin
                            // BRAM access
                            dmem_rdata <= bram_rdata;
                            dmem_ready <= 1'b1;
                        end else if (is_write) begin
                            // AXI write
                            m_axi.AWADDR <= dmem_addr;
                            m_axi.AWVALID <= 1'b1;
                            m_axi.WDATA <= dmem_wdata;
                            m_axi.WSTRB <= dmem_we;
                            m_axi.WVALID <= 1'b1;
                            state <= AW_W;
                        end else begin
                            // AXI read
                            m_axi.ARADDR <= dmem_addr;
                            m_axi.ARVALID <= 1'b1;
                            state <= AR;
                        end
                    end
                end
                
                AW_W: begin
                    if (m_axi.AWREADY) m_axi.AWVALID <= 1'b0;
                    if (m_axi.WREADY) m_axi.WVALID <= 1'b0;
                    
                    if (!m_axi.AWVALID && !m_axi.WVALID) begin
                        m_axi.BREADY <= 1'b1;
                        state <= B_WAIT;
                    end
                end
                
                B_WAIT: begin
                    if (m_axi.BVALID) begin
                        m_axi.BREADY <= 1'b0;
                        dmem_ready <= 1'b1;
                        state <= IDLE;
                    end
                end
                
                AR: begin
                    if (m_axi.ARREADY) begin
                        m_axi.ARVALID <= 1'b0;
                        m_axi.RREADY <= 1'b1;
                        state <= R_WAIT;
                    end
                end
                
                R_WAIT: begin
                    if (m_axi.RVALID) begin
                        rdata_reg <= m_axi.RDATA;
                        dmem_rdata <= m_axi.RDATA;
                        m_axi.RREADY <= 1'b0;
                        dmem_ready <= 1'b1;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule

// AXI-Lite stub for unimplemented peripherals
module axi_lite_stub (
    input logic clk,
    input logic rst,
    axi_lite_if.slave s_axi
);

    always_ff @(posedge clk) begin
        if (rst) begin
            s_axi.AWREADY <= 1'b0;
            s_axi.WREADY <= 1'b0;
            s_axi.BVALID <= 1'b0;
            s_axi.BRESP <= 2'b00;
            s_axi.ARREADY <= 1'b0;
            s_axi.RVALID <= 1'b0;
            s_axi.RDATA <= '0;
            s_axi.RRESP <= 2'b00;
        end else begin
            // Accept all transactions but return zeros
            s_axi.AWREADY <= s_axi.AWVALID && !s_axi.AWREADY;
            s_axi.WREADY <= s_axi.WVALID && !s_axi.WREADY;
            s_axi.BVALID <= s_axi.WREADY && s_axi.WVALID && !s_axi.BVALID;
            if (s_axi.BREADY && s_axi.BVALID) s_axi.BVALID <= 1'b0;
            
            s_axi.ARREADY <= s_axi.ARVALID && !s_axi.ARREADY;
            s_axi.RVALID <= s_axi.ARREADY && !s_axi.RVALID;
            s_axi.RDATA <= 32'h0;
            if (s_axi.RREADY && s_axi.RVALID) s_axi.RVALID <= 1'b0;
        end
    end

endmodule