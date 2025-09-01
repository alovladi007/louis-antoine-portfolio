`timescale 1ns/1ps

module spi #(
    parameter AW = 32,
    parameter DW = 32,
    parameter CLK_FREQ = 50_000_000,
    parameter SPI_FREQ = 10_000_000
)(
    input  logic clk,
    input  logic rst,
    
    // AXI-Lite interface
    axi_lite_if.slave s_axi,
    
    // SPI pins
    output logic spi_sclk,
    output logic spi_mosi,
    input  logic spi_miso,
    output logic [3:0] spi_cs_n,
    
    // Interrupt
    output logic irq
);

    // Register map
    // 0x00: Control Register [0]=Enable, [1]=CPOL, [2]=CPHA, [3]=LSB_First, [7:4]=CS_Select
    // 0x04: Status Register [0]=Busy, [1]=TX_Empty, [2]=RX_Full
    // 0x08: TX Data Register
    // 0x0C: RX Data Register
    // 0x10: Clock Divider Register
    // 0x14: Interrupt Enable Register
    
    localparam ADDR_CONTROL  = 3'h0;
    localparam ADDR_STATUS   = 3'h1;
    localparam ADDR_TX_DATA  = 3'h2;
    localparam ADDR_RX_DATA  = 3'h3;
    localparam ADDR_CLK_DIV  = 3'h4;
    localparam ADDR_IRQ_EN   = 3'h5;
    
    // Internal registers
    logic [7:0]  control_reg;
    logic [2:0]  status_reg;
    logic [7:0]  tx_data_reg;
    logic [7:0]  rx_data_reg;
    logic [15:0] clk_div_reg;
    logic [1:0]  irq_en_reg;
    
    // SPI core signals
    logic spi_start;
    logic spi_busy;
    logic spi_done;
    logic [7:0] spi_tx_data;
    logic [7:0] spi_rx_data;
    
    // Default clock divisor
    localparam DEFAULT_DIV = CLK_FREQ / (2 * SPI_FREQ);
    
    // AXI-Lite write FSM
    typedef enum logic [1:0] {W_IDLE, W_DATA, W_RESP} write_state_t;
    write_state_t wstate;
    logic [AW-1:0] waddr;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            wstate <= W_IDLE;
            waddr <= '0;
            control_reg <= 8'h01;  // Enable, Mode 0, MSB first, CS0
            tx_data_reg <= '0;
            clk_div_reg <= DEFAULT_DIV[15:0];
            irq_en_reg <= 2'b00;
            spi_start <= 1'b0;
            s_axi.AWREADY <= 1'b0;
            s_axi.WREADY <= 1'b0;
            s_axi.BVALID <= 1'b0;
            s_axi.BRESP <= 2'b00;
        end else begin
            spi_start <= 1'b0;  // Single cycle pulse
            
            case (wstate)
                W_IDLE: begin
                    s_axi.AWREADY <= 1'b1;
                    if (s_axi.AWVALID) begin
                        waddr <= s_axi.AWADDR;
                        s_axi.AWREADY <= 1'b0;
                        wstate <= W_DATA;
                    end
                end
                
                W_DATA: begin
                    s_axi.WREADY <= 1'b1;
                    if (s_axi.WVALID) begin
                        case (waddr[4:2])
                            ADDR_CONTROL: control_reg <= s_axi.WDATA[7:0];
                            ADDR_TX_DATA: begin
                                if (!spi_busy && control_reg[0]) begin
                                    tx_data_reg <= s_axi.WDATA[7:0];
                                    spi_start <= 1'b1;
                                end
                            end
                            ADDR_CLK_DIV: clk_div_reg <= s_axi.WDATA[15:0];
                            ADDR_IRQ_EN: irq_en_reg <= s_axi.WDATA[1:0];
                        endcase
                        s_axi.WREADY <= 1'b0;
                        s_axi.BVALID <= 1'b1;
                        s_axi.BRESP <= 2'b00;
                        wstate <= W_RESP;
                    end
                end
                
                W_RESP: begin
                    if (s_axi.BREADY) begin
                        s_axi.BVALID <= 1'b0;
                        wstate <= W_IDLE;
                    end
                end
            endcase
        end
    end
    
    // AXI-Lite read FSM
    typedef enum logic [1:0] {R_IDLE, R_DATA} read_state_t;
    read_state_t rstate;
    logic [AW-1:0] raddr;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            rstate <= R_IDLE;
            raddr <= '0;
            s_axi.ARREADY <= 1'b0;
            s_axi.RVALID <= 1'b0;
            s_axi.RDATA <= '0;
            s_axi.RRESP <= 2'b00;
        end else begin
            case (rstate)
                R_IDLE: begin
                    s_axi.ARREADY <= 1'b1;
                    if (s_axi.ARVALID) begin
                        raddr <= s_axi.ARADDR;
                        s_axi.ARREADY <= 1'b0;
                        rstate <= R_DATA;
                    end
                end
                
                R_DATA: begin
                    case (raddr[4:2])
                        ADDR_CONTROL: s_axi.RDATA <= {24'h0, control_reg};
                        ADDR_STATUS: s_axi.RDATA <= {29'h0, status_reg};
                        ADDR_TX_DATA: s_axi.RDATA <= {24'h0, tx_data_reg};
                        ADDR_RX_DATA: s_axi.RDATA <= {24'h0, rx_data_reg};
                        ADDR_CLK_DIV: s_axi.RDATA <= {16'h0, clk_div_reg};
                        ADDR_IRQ_EN: s_axi.RDATA <= {30'h0, irq_en_reg};
                        default: s_axi.RDATA <= 32'h0;
                    endcase
                    s_axi.RRESP <= 2'b00;
                    s_axi.RVALID <= 1'b1;
                    if (s_axi.RREADY) begin
                        s_axi.RVALID <= 1'b0;
                        rstate <= R_IDLE;
                    end
                end
            endcase
        end
    end
    
    // RX data buffer
    always_ff @(posedge clk) begin
        if (rst) begin
            rx_data_reg <= '0;
        end else if (spi_done) begin
            rx_data_reg <= spi_rx_data;
        end
    end
    
    // Status register
    assign status_reg = {
        (rx_data_reg != '0),  // RX Full
        !spi_busy,            // TX Empty
        spi_busy              // Busy
    };
    
    // SPI Master Core
    spi_master u_spi_master (
        .clk(clk),
        .rst(rst),
        .start(spi_start),
        .cpol(control_reg[1]),
        .cpha(control_reg[2]),
        .lsb_first(control_reg[3]),
        .clk_div(clk_div_reg),
        .tx_data(tx_data_reg),
        .rx_data(spi_rx_data),
        .sclk(spi_sclk),
        .mosi(spi_mosi),
        .miso(spi_miso),
        .busy(spi_busy),
        .done(spi_done)
    );
    
    // Chip select generation
    always_comb begin
        spi_cs_n = 4'hF;  // All deselected by default
        if (control_reg[0] && (spi_busy || spi_start)) begin
            spi_cs_n[control_reg[5:4]] = 1'b0;  // Active low
        end
    end
    
    // Interrupt generation
    assign irq = (irq_en_reg[0] && spi_done) || (irq_en_reg[1] && (rx_data_reg != '0));

endmodule

// SPI Master Core
module spi_master (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic        cpol,
    input  logic        cpha,
    input  logic        lsb_first,
    input  logic [15:0] clk_div,
    input  logic [7:0]  tx_data,
    output logic [7:0]  rx_data,
    output logic        sclk,
    output logic        mosi,
    input  logic        miso,
    output logic        busy,
    output logic        done
);

    typedef enum logic [1:0] {IDLE, TRANSFER, DONE} state_t;
    state_t state;
    
    logic [15:0] clk_counter;
    logic [2:0]  bit_counter;
    logic [7:0]  tx_shift_reg;
    logic [7:0]  rx_shift_reg;
    logic        sclk_reg;
    logic        sample_edge;
    logic        shift_edge;
    
    // Clock phase and polarity handling
    always_comb begin
        if (cpha) begin
            sample_edge = (sclk_reg != cpol);
            shift_edge = (sclk_reg == cpol);
        end else begin
            sample_edge = (sclk_reg == cpol);
            shift_edge = (sclk_reg != cpol);
        end
    end
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            sclk_reg <= cpol;
            mosi <= 1'b0;
            rx_data <= '0;
            clk_counter <= '0;
            bit_counter <= '0;
            tx_shift_reg <= '0;
            rx_shift_reg <= '0;
        end else begin
            done <= 1'b0;
            
            case (state)
                IDLE: begin
                    sclk_reg <= cpol;
                    busy <= 1'b0;
                    if (start) begin
                        tx_shift_reg <= lsb_first ? tx_data : {tx_data[6:0], tx_data[7]};
                        rx_shift_reg <= '0;
                        clk_counter <= clk_div;
                        bit_counter <= 3'd0;
                        busy <= 1'b1;
                        state <= TRANSFER;
                        if (!cpha) begin
                            mosi <= lsb_first ? tx_data[0] : tx_data[7];
                        end
                    end
                end
                
                TRANSFER: begin
                    if (clk_counter == 0) begin
                        clk_counter <= clk_div;
                        sclk_reg <= ~sclk_reg;
                        
                        if (sample_edge) begin
                            // Sample MISO
                            if (lsb_first) begin
                                rx_shift_reg <= {miso, rx_shift_reg[7:1]};
                            end else begin
                                rx_shift_reg <= {rx_shift_reg[6:0], miso};
                            end
                        end
                        
                        if (shift_edge) begin
                            if (bit_counter == 3'd7) begin
                                state <= DONE;
                            end else begin
                                bit_counter <= bit_counter + 1;
                                // Shift and output next bit
                                if (lsb_first) begin
                                    tx_shift_reg <= {1'b0, tx_shift_reg[7:1]};
                                    mosi <= tx_shift_reg[1];
                                end else begin
                                    tx_shift_reg <= {tx_shift_reg[6:0], 1'b0};
                                    mosi <= tx_shift_reg[6];
                                end
                            end
                        end
                    end else begin
                        clk_counter <= clk_counter - 1;
                    end
                end
                
                DONE: begin
                    rx_data <= rx_shift_reg;
                    done <= 1'b1;
                    sclk_reg <= cpol;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign sclk = sclk_reg;

endmodule