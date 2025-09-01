`timescale 1ns/1ps

module soc_tb;

    // Clock and reset
    logic clk;
    logic rst;
    
    // UART
    logic uart_tx;
    logic uart_rx;
    
    // SPI
    logic spi_sclk;
    logic spi_mosi;
    logic spi_miso;
    logic [3:0] spi_cs_n;
    
    // GPIO
    logic [15:0] gpio_in;
    logic [15:0] gpio_out;
    logic [15:0] gpio_oe;
    
    // PWM
    logic [3:0] pwm_out;
    
    // Debug
    logic [7:0] led;
    
    // JTAG
    logic tck, tms, tdi, tdo;
    
    // DUT instance
    soc_top #(
        .CLK_FREQ(50_000_000),
        .BAUD_RATE(115200),
        .SRAM_SIZE(128*1024),
        .BOOT_ADDR(32'h0000_0000)
    ) dut (
        .clk_in(clk),
        .rst_in(rst),
        .uart_tx(uart_tx),
        .uart_rx(uart_rx),
        .spi_sclk(spi_sclk),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        .gpio_in(gpio_in),
        .gpio_out(gpio_out),
        .gpio_oe(gpio_oe),
        .pwm_out(pwm_out),
        .led(led),
        .tck(tck),
        .tms(tms),
        .tdi(tdi),
        .tdo(tdo)
    );
    
    // Clock generation (50 MHz)
    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end
    
    // UART receiver for monitoring
    logic [7:0] uart_rx_data;
    logic uart_rx_valid;
    
    uart_monitor u_uart_mon (
        .clk(clk),
        .rst(rst),
        .rx(uart_tx),  // Monitor DUT's TX
        .data(uart_rx_data),
        .valid(uart_rx_valid)
    );
    
    // Test vectors for CPU
    initial begin
        $dumpfile("soc_tb.vcd");
        $dumpvars(0, soc_tb);
        
        // Initialize signals
        rst = 1;
        uart_rx = 1;
        spi_miso = 0;
        gpio_in = 16'h0000;
        tck = 0;
        tms = 0;
        tdi = 0;
        
        // Load test program into memory
        load_test_program();
        
        // Release reset
        #100;
        rst = 0;
        
        // Run test
        #10000;
        
        // Test GPIO
        test_gpio();
        
        // Test UART
        test_uart();
        
        // Test SPI
        test_spi();
        
        // Test Timer/PWM
        test_timer();
        
        // Run for more cycles
        #100000;
        
        $display("Test completed!");
        $finish;
    end
    
    // Load test program
    task load_test_program();
        // Simple test program that writes to UART
        // This would be loaded from a hex file in real implementation
        
        // NOP sled at reset vector
        dut.u_bram.mem[0] = 8'h13; dut.u_bram.mem[1] = 8'h00; dut.u_bram.mem[2] = 8'h00; dut.u_bram.mem[3] = 8'h00;
        
        // Load immediate values and write to UART
        // lui t0, 0x40000  ; UART base address
        dut.u_bram.mem[4] = 8'hB7; dut.u_bram.mem[5] = 8'h02; dut.u_bram.mem[6] = 8'h00; dut.u_bram.mem[7] = 8'h40;
        
        // addi t1, zero, 'H'  ; ASCII 'H' = 0x48
        dut.u_bram.mem[8] = 8'h13; dut.u_bram.mem[9] = 8'h03; dut.u_bram.mem[10] = 8'h80; dut.u_bram.mem[11] = 8'h04;
        
        // sw t1, 0(t0)  ; Write to UART TX register
        dut.u_bram.mem[12] = 8'h23; dut.u_bram.mem[13] = 8'h20; dut.u_bram.mem[14] = 8'h62; dut.u_bram.mem[15] = 8'h00;
        
        // Infinite loop
        // loop: j loop
        dut.u_bram.mem[16] = 8'h6F; dut.u_bram.mem[17] = 8'h00; dut.u_bram.mem[18] = 8'h00; dut.u_bram.mem[19] = 8'h00;
        
        $display("Test program loaded into memory");
    endtask
    
    // Test GPIO
    task test_gpio();
        $display("Testing GPIO...");
        
        // Set some input pins
        gpio_in = 16'hA5A5;
        #1000;
        
        // Check if outputs respond
        if (gpio_out != 16'h0000) begin
            $display("GPIO output changed: %h", gpio_out);
        end
    endtask
    
    // Test UART
    task test_uart();
        $display("Testing UART...");
        
        // Wait for any UART transmission
        @(posedge uart_rx_valid);
        $display("UART received: %c (0x%h)", uart_rx_data, uart_rx_data);
    endtask
    
    // Test SPI
    task test_spi();
        $display("Testing SPI...");
        
        // Provide some MISO data
        spi_miso = 1;
        #1000;
        spi_miso = 0;
        #1000;
        
        // Check for SPI activity
        if (spi_cs_n != 4'hF) begin
            $display("SPI chip select activated: %b", spi_cs_n);
        end
    endtask
    
    // Test Timer
    task test_timer();
        $display("Testing Timer/PWM...");
        
        // Wait and check for PWM outputs
        #10000;
        if (pwm_out != 4'h0) begin
            $display("PWM outputs active: %b", pwm_out);
        end
    endtask

endmodule

// Simple UART monitor for testbench
module uart_monitor (
    input  logic clk,
    input  logic rst,
    input  logic rx,
    output logic [7:0] data,
    output logic valid
);

    parameter CLK_FREQ = 50_000_000;
    parameter BAUD_RATE = 115200;
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