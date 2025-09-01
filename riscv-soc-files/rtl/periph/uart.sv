`timescale 1ns/1ps

module uart #(
    parameter AW = 32,
    parameter DW = 32,
    parameter CLK_FREQ = 50_000_000,
    parameter BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rst,
    
    // AXI-Lite interface
    axi_lite_if.slave s_axi,
    
    // UART pins
    output logic uart_tx,
    input  logic uart_rx,
    
    // Interrupt
    output logic irq
);

    // Register map
    // 0x00: TX Data Register (write to transmit)
    // 0x04: RX Data Register (read to receive)
    // 0x08: Status Register [0]=TX busy, [1]=RX ready, [2]=TX full, [3]=RX full
    // 0x0C: Control Register [0]=TX enable, [1]=RX enable, [2]=TX IRQ enable, [3]=RX IRQ enable
    // 0x10: Baud Divisor Register
    
    localparam ADDR_TX_DATA   = 3'h0;
    localparam ADDR_RX_DATA   = 3'h1;
    localparam ADDR_STATUS    = 3'h2;
    localparam ADDR_CONTROL   = 3'h3;
    localparam ADDR_BAUD_DIV  = 3'h4;
    
    // Internal registers
    logic [7:0]  tx_data_reg;
    logic [7:0]  rx_data_reg;
    logic [3:0]  status_reg;
    logic [3:0]  control_reg;
    logic [15:0] baud_div_reg;
    
    // TX/RX logic signals
    logic tx_start, tx_busy, tx_done;
    logic rx_ready, rx_valid;
    logic [7:0] rx_data;
    
    // Default baud divisor
    localparam DEFAULT_DIV = CLK_FREQ / BAUD_RATE;
    
    // AXI-Lite write FSM
    typedef enum logic [1:0] {W_IDLE, W_ADDR, W_DATA, W_RESP} write_state_t;
    write_state_t wstate;
    logic [AW-1:0] waddr;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            wstate <= W_IDLE;
            waddr <= '0;
            tx_data_reg <= '0;
            control_reg <= 4'b0011;  // TX and RX enabled by default
            baud_div_reg <= DEFAULT_DIV[15:0];
            tx_start <= 1'b0;
            s_axi.AWREADY <= 1'b0;
            s_axi.WREADY <= 1'b0;
            s_axi.BVALID <= 1'b0;
            s_axi.BRESP <= 2'b00;
        end else begin
            tx_start <= 1'b0;  // Single cycle pulse
            
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
                            ADDR_TX_DATA: begin
                                if (!tx_busy && control_reg[0]) begin
                                    tx_data_reg <= s_axi.WDATA[7:0];
                                    tx_start <= 1'b1;
                                end
                            end
                            ADDR_CONTROL: control_reg <= s_axi.WDATA[3:0];
                            ADDR_BAUD_DIV: baud_div_reg <= s_axi.WDATA[15:0];
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
                
                default: wstate <= W_IDLE;
            endcase
        end
    end
    
    // AXI-Lite read FSM
    typedef enum logic [1:0] {R_IDLE, R_ADDR, R_DATA} read_state_t;
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
                        ADDR_TX_DATA: s_axi.RDATA <= {24'h0, tx_data_reg};
                        ADDR_RX_DATA: s_axi.RDATA <= {24'h0, rx_data_reg};
                        ADDR_STATUS: s_axi.RDATA <= {28'h0, status_reg};
                        ADDR_CONTROL: s_axi.RDATA <= {28'h0, control_reg};
                        ADDR_BAUD_DIV: s_axi.RDATA <= {16'h0, baud_div_reg};
                        default: s_axi.RDATA <= 32'h0;
                    endcase
                    s_axi.RRESP <= 2'b00;
                    s_axi.RVALID <= 1'b1;
                    if (s_axi.RREADY) begin
                        s_axi.RVALID <= 1'b0;
                        // Clear RX ready flag on read
                        if (raddr[4:2] == ADDR_RX_DATA) begin
                            rx_data_reg <= '0;
                        end
                        rstate <= R_IDLE;
                    end
                end
                
                default: rstate <= R_IDLE;
            endcase
        end
    end
    
    // TX module
    uart_tx u_uart_tx (
        .clk(clk),
        .rst(rst),
        .tx_data(tx_data_reg),
        .tx_start(tx_start),
        .baud_div(baud_div_reg),
        .tx(uart_tx),
        .tx_busy(tx_busy),
        .tx_done(tx_done)
    );
    
    // RX module
    uart_rx u_uart_rx (
        .clk(clk),
        .rst(rst),
        .rx(uart_rx),
        .baud_div(baud_div_reg),
        .rx_data(rx_data),
        .rx_valid(rx_valid)
    );
    
    // RX data buffer
    always_ff @(posedge clk) begin
        if (rst) begin
            rx_data_reg <= '0;
            rx_ready <= 1'b0;
        end else if (rx_valid && control_reg[1]) begin
            rx_data_reg <= rx_data;
            rx_ready <= 1'b1;
        end else if (rstate == R_DATA && raddr[4:2] == ADDR_RX_DATA && s_axi.RREADY) begin
            rx_ready <= 1'b0;
        end
    end
    
    // Status register
    assign status_reg = {
        rx_ready,     // Bit 3: RX data available
        tx_busy,      // Bit 2: TX FIFO full (using busy for simplicity)
        rx_ready,     // Bit 1: RX ready
        tx_busy       // Bit 0: TX busy
    };
    
    // Interrupt generation
    assign irq = (control_reg[2] && tx_done) || (control_reg[3] && rx_valid);

endmodule

// UART Transmitter
module uart_tx (
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  tx_data,
    input  logic        tx_start,
    input  logic [15:0] baud_div,
    output logic        tx,
    output logic        tx_busy,
    output logic        tx_done
);

    typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
    state_t state;
    
    logic [15:0] baud_counter;
    logic [2:0]  bit_counter;
    logic [7:0]  shift_reg;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            tx <= 1'b1;
            tx_busy <= 1'b0;
            tx_done <= 1'b0;
            baud_counter <= '0;
            bit_counter <= '0;
            shift_reg <= '0;
        end else begin
            tx_done <= 1'b0;
            
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    tx_busy <= 1'b0;
                    if (tx_start) begin
                        shift_reg <= tx_data;
                        baud_counter <= baud_div - 1;
                        tx_busy <= 1'b1;
                        state <= START;
                    end
                end
                
                START: begin
                    tx <= 1'b0;  // Start bit
                    if (baud_counter == 0) begin
                        baud_counter <= baud_div - 1;
                        bit_counter <= 3'd0;
                        state <= DATA;
                    end else begin
                        baud_counter <= baud_counter - 1;
                    end
                end
                
                DATA: begin
                    tx <= shift_reg[0];
                    if (baud_counter == 0) begin
                        baud_counter <= baud_div - 1;
                        shift_reg <= {1'b0, shift_reg[7:1]};
                        if (bit_counter == 3'd7) begin
                            state <= STOP;
                        end else begin
                            bit_counter <= bit_counter + 1;
                        end
                    end else begin
                        baud_counter <= baud_counter - 1;
                    end
                end
                
                STOP: begin
                    tx <= 1'b1;  // Stop bit
                    if (baud_counter == 0) begin
                        tx_done <= 1'b1;
                        state <= IDLE;
                    end else begin
                        baud_counter <= baud_counter - 1;
                    end
                end
            endcase
        end
    end

endmodule

// UART Receiver
module uart_rx (
    input  logic        clk,
    input  logic        rst,
    input  logic        rx,
    input  logic [15:0] baud_div,
    output logic [7:0]  rx_data,
    output logic        rx_valid
);

    typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
    state_t state;
    
    logic [15:0] baud_counter;
    logic [2:0]  bit_counter;
    logic [7:0]  shift_reg;
    logic        rx_sync, rx_sync2;
    
    // Synchronize RX input
    always_ff @(posedge clk) begin
        if (rst) begin
            rx_sync <= 1'b1;
            rx_sync2 <= 1'b1;
        end else begin
            rx_sync <= rx;
            rx_sync2 <= rx_sync;
        end
    end
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            rx_valid <= 1'b0;
            rx_data <= '0;
            baud_counter <= '0;
            bit_counter <= '0;
            shift_reg <= '0;
        end else begin
            rx_valid <= 1'b0;
            
            case (state)
                IDLE: begin
                    if (!rx_sync2) begin  // Start bit detected
                        baud_counter <= baud_div / 2 - 1;  // Sample at mid-bit
                        state <= START;
                    end
                end
                
                START: begin
                    if (baud_counter == 0) begin
                        if (!rx_sync2) begin  // Verify start bit
                            baud_counter <= baud_div - 1;
                            bit_counter <= 3'd0;
                            state <= DATA;
                        end else begin
                            state <= IDLE;  // False start
                        end
                    end else begin
                        baud_counter <= baud_counter - 1;
                    end
                end
                
                DATA: begin
                    if (baud_counter == 0) begin
                        baud_counter <= baud_div - 1;
                        shift_reg <= {rx_sync2, shift_reg[7:1]};
                        if (bit_counter == 3'd7) begin
                            state <= STOP;
                        end else begin
                            bit_counter <= bit_counter + 1;
                        end
                    end else begin
                        baud_counter <= baud_counter - 1;
                    end
                end
                
                STOP: begin
                    if (baud_counter == 0) begin
                        if (rx_sync2) begin  // Valid stop bit
                            rx_data <= shift_reg;
                            rx_valid <= 1'b1;
                        end
                        state <= IDLE;
                    end else begin
                        baud_counter <= baud_counter - 1;
                    end
                end
            endcase
        end
    end

endmodule