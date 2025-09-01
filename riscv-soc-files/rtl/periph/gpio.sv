`timescale 1ns/1ps

module gpio #(
    parameter AW = 32,
    parameter DW = 32,
    parameter NUM_GPIO = 32
)(
    input  logic clk,
    input  logic rst,
    
    // AXI-Lite interface
    axi_lite_if.slave s_axi,
    
    // GPIO pins
    input  logic [NUM_GPIO-1:0] gpio_in,
    output logic [NUM_GPIO-1:0] gpio_out,
    output logic [NUM_GPIO-1:0] gpio_oe,  // Output enable
    
    // Interrupt
    output logic irq
);

    // Register map
    // 0x00: Data Input Register (read-only)
    // 0x04: Data Output Register
    // 0x08: Direction Register (0=input, 1=output)
    // 0x0C: Interrupt Enable Register
    // 0x10: Interrupt Status Register
    // 0x14: Interrupt Type Register (0=level, 1=edge)
    // 0x18: Interrupt Polarity Register (0=low/falling, 1=high/rising)
    // 0x1C: Interrupt Clear Register (write 1 to clear)
    
    localparam ADDR_DATA_IN    = 3'h0;
    localparam ADDR_DATA_OUT   = 3'h1;
    localparam ADDR_DIR        = 3'h2;
    localparam ADDR_IRQ_EN     = 3'h3;
    localparam ADDR_IRQ_STATUS = 3'h4;
    localparam ADDR_IRQ_TYPE   = 3'h5;
    localparam ADDR_IRQ_POL    = 3'h6;
    localparam ADDR_IRQ_CLEAR  = 3'h7;
    
    // Internal registers
    logic [NUM_GPIO-1:0] data_out_reg;
    logic [NUM_GPIO-1:0] dir_reg;
    logic [NUM_GPIO-1:0] irq_en_reg;
    logic [NUM_GPIO-1:0] irq_status_reg;
    logic [NUM_GPIO-1:0] irq_type_reg;
    logic [NUM_GPIO-1:0] irq_pol_reg;
    
    // Synchronized input
    logic [NUM_GPIO-1:0] gpio_in_sync;
    logic [NUM_GPIO-1:0] gpio_in_prev;
    
    // AXI-Lite write FSM
    typedef enum logic [1:0] {W_IDLE, W_DATA, W_RESP} write_state_t;
    write_state_t wstate;
    logic [AW-1:0] waddr;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            wstate <= W_IDLE;
            waddr <= '0;
            data_out_reg <= '0;
            dir_reg <= '0;
            irq_en_reg <= '0;
            irq_type_reg <= '0;
            irq_pol_reg <= '0;
            s_axi.AWREADY <= 1'b0;
            s_axi.WREADY <= 1'b0;
            s_axi.BVALID <= 1'b0;
            s_axi.BRESP <= 2'b00;
        end else begin
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
                            ADDR_DATA_OUT: data_out_reg <= s_axi.WDATA[NUM_GPIO-1:0];
                            ADDR_DIR: dir_reg <= s_axi.WDATA[NUM_GPIO-1:0];
                            ADDR_IRQ_EN: irq_en_reg <= s_axi.WDATA[NUM_GPIO-1:0];
                            ADDR_IRQ_TYPE: irq_type_reg <= s_axi.WDATA[NUM_GPIO-1:0];
                            ADDR_IRQ_POL: irq_pol_reg <= s_axi.WDATA[NUM_GPIO-1:0];
                            ADDR_IRQ_CLEAR: begin
                                // Clear interrupt status bits where write data is 1
                                irq_status_reg <= irq_status_reg & ~s_axi.WDATA[NUM_GPIO-1:0];
                            end
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
                        ADDR_DATA_IN: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, gpio_in_sync};
                        ADDR_DATA_OUT: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, data_out_reg};
                        ADDR_DIR: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, dir_reg};
                        ADDR_IRQ_EN: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, irq_en_reg};
                        ADDR_IRQ_STATUS: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, irq_status_reg};
                        ADDR_IRQ_TYPE: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, irq_type_reg};
                        ADDR_IRQ_POL: s_axi.RDATA <= {{(32-NUM_GPIO){1'b0}}, irq_pol_reg};
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
    
    // Input synchronization
    always_ff @(posedge clk) begin
        if (rst) begin
            gpio_in_sync <= '0;
            gpio_in_prev <= '0;
        end else begin
            gpio_in_sync <= gpio_in;
            gpio_in_prev <= gpio_in_sync;
        end
    end
    
    // Interrupt detection
    always_ff @(posedge clk) begin
        if (rst) begin
            irq_status_reg <= '0;
        end else begin
            for (int i = 0; i < NUM_GPIO; i++) begin
                if (irq_en_reg[i]) begin
                    if (irq_type_reg[i]) begin
                        // Edge detection
                        if (irq_pol_reg[i]) begin
                            // Rising edge
                            if (gpio_in_sync[i] && !gpio_in_prev[i]) begin
                                irq_status_reg[i] <= 1'b1;
                            end
                        end else begin
                            // Falling edge
                            if (!gpio_in_sync[i] && gpio_in_prev[i]) begin
                                irq_status_reg[i] <= 1'b1;
                            end
                        end
                    end else begin
                        // Level detection
                        if (gpio_in_sync[i] == irq_pol_reg[i]) begin
                            irq_status_reg[i] <= 1'b1;
                        end
                    end
                end
            end
            
            // Clear on write to clear register (handled in write FSM)
        end
    end
    
    // GPIO output
    assign gpio_out = data_out_reg;
    assign gpio_oe = dir_reg;
    
    // Interrupt generation
    assign irq = |(irq_status_reg & irq_en_reg);

endmodule