`timescale 1ns/1ps

module soc_seq_uart_tb;
    
    logic clk = 0;
    logic rst_btn = 1;
    logic uart_txd;
    
    // 50 MHz clock
    always #10 clk = ~clk; 
    
    // DUT instantiation
    soc_seq_uart_top DUT (
        .clk_50mhz(clk), 
        .rst_btn(rst_btn), 
        .uart_txd(uart_txd)
    );
    
    // UART monitor to capture transmitted characters
    logic [7:0] uart_rx_data;
    logic uart_rx_valid;
    
    uart_monitor #(
        .CLK_FREQ(50_000_000),
        .BAUD_RATE(115200)
    ) u_monitor (
        .clk(clk),
        .rst(rst_btn),
        .rx(uart_txd),
        .data(uart_rx_data),
        .valid(uart_rx_valid)
    );
    
    // Test sequence
    initial begin
        $dumpfile("soc_seq_uart_tb.vcd");
        $dumpvars(0, soc_seq_uart_tb);
        
        $display("Starting micro-sequencer test...");
        
        // Hold reset for a while
        repeat (10) @(posedge clk);
        rst_btn = 0;
        
        // Monitor UART output
        fork
            begin
                string received = "";
                while (received != "HELLO\n") begin
                    @(posedge uart_rx_valid);
                    received = {received, string'(uart_rx_data)};
                    $display("UART RX: %c (0x%02x)", uart_rx_data, uart_rx_data);
                end
                $display("SUCCESS: Received complete message: %s", received);
            end
        join_none
        
        // Run for enough cycles to transmit "HELLO\n"
        repeat (1000000) @(posedge clk);
        
        $display("Test completed!");
        $finish;
    end

endmodule

// Simple UART monitor for testbench
module uart_monitor #(
    parameter CLK_FREQ = 50_000_000,
    parameter BAUD_RATE = 115200
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       rx,
    output logic [7:0] data,
    output logic       valid
);

    localparam BIT_PERIOD = CLK_FREQ / BAUD_RATE;
    
    typedef enum {IDLE, START, DATA, STOP} state_t;
    state_t state;
    
    logic [$clog2(BIT_PERIOD)-1:0] counter;
    logic [2:0] bit_idx;
    logic [7:0] shift_reg;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            counter <= 0;
            bit_idx <= 0;
            shift_reg <= 0;
            data <= 0;
            valid <= 0;
        end else begin
            valid <= 0;
            
            case (state)
                IDLE: begin
                    if (!rx) begin  // Start bit detected
                        counter <= BIT_PERIOD / 2;
                        state <= START;
                    end
                end
                
                START: begin
                    if (counter == 0) begin
                        if (!rx) begin  // Verify start bit
                            counter <= BIT_PERIOD;
                            bit_idx <= 0;
                            state <= DATA;
                        end else begin
                            state <= IDLE;
                        end
                    end else begin
                        counter <= counter - 1;
                    end
                end
                
                DATA: begin
                    if (counter == 0) begin
                        shift_reg[bit_idx] <= rx;
                        counter <= BIT_PERIOD;
                        
                        if (bit_idx == 7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        counter <= counter - 1;
                    end
                end
                
                STOP: begin
                    if (counter == 0) begin
                        if (rx) begin  // Valid stop bit
                            data <= shift_reg;
                            valid <= 1;
                        end
                        state <= IDLE;
                    end else begin
                        counter <= counter - 1;
                    end
                end
            endcase
        end
    end

endmodule