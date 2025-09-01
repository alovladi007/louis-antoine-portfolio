`timescale 1ns/1ps

module timer #(
    parameter AW = 32,
    parameter DW = 32,
    parameter CLK_FREQ = 50_000_000
)(
    input  logic clk,
    input  logic rst,
    
    // AXI-Lite interface
    axi_lite_if.slave s_axi,
    
    // PWM outputs
    output logic [3:0] pwm_out,
    
    // Interrupt
    output logic irq
);

    // Register map
    // 0x00: Control Register [0]=Enable, [1]=Auto-reload, [2]=IRQ Enable, [3]=PWM0_En, [4]=PWM1_En, [5]=PWM2_En, [6]=PWM3_En
    // 0x04: Status Register [0]=Timer overflow
    // 0x08: Counter Value (32-bit)
    // 0x0C: Reload Value (32-bit)
    // 0x10: Compare Value (32-bit)
    // 0x14: Prescaler (16-bit)
    // 0x18: PWM0 Duty Cycle (16-bit)
    // 0x1C: PWM1 Duty Cycle (16-bit)
    // 0x20: PWM2 Duty Cycle (16-bit)
    // 0x24: PWM3 Duty Cycle (16-bit)
    // 0x28: PWM Period (16-bit)
    
    localparam ADDR_CONTROL   = 4'h0;
    localparam ADDR_STATUS    = 4'h1;
    localparam ADDR_COUNTER   = 4'h2;
    localparam ADDR_RELOAD    = 4'h3;
    localparam ADDR_COMPARE   = 4'h4;
    localparam ADDR_PRESCALER = 4'h5;
    localparam ADDR_PWM0_DUTY = 4'h6;
    localparam ADDR_PWM1_DUTY = 4'h7;
    localparam ADDR_PWM2_DUTY = 4'h8;
    localparam ADDR_PWM3_DUTY = 4'h9;
    localparam ADDR_PWM_PERIOD = 4'hA;
    
    // Internal registers
    logic [7:0]  control_reg;
    logic        status_reg;
    logic [31:0] counter_reg;
    logic [31:0] reload_reg;
    logic [31:0] compare_reg;
    logic [15:0] prescaler_reg;
    logic [15:0] prescale_counter;
    logic [15:0] pwm_duty [4];
    logic [15:0] pwm_period_reg;
    logic [15:0] pwm_counter;
    
    // Timer tick
    logic timer_tick;
    
    // AXI-Lite write FSM
    typedef enum logic [1:0] {W_IDLE, W_DATA, W_RESP} write_state_t;
    write_state_t wstate;
    logic [AW-1:0] waddr;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            wstate <= W_IDLE;
            waddr <= '0;
            control_reg <= 8'h00;
            reload_reg <= 32'hFFFF_FFFF;
            compare_reg <= 32'hFFFF_FFFF;
            prescaler_reg <= 16'h0000;
            pwm_duty[0] <= 16'h8000;
            pwm_duty[1] <= 16'h8000;
            pwm_duty[2] <= 16'h8000;
            pwm_duty[3] <= 16'h8000;
            pwm_period_reg <= 16'hFFFF;
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
                        case (waddr[5:2])
                            ADDR_CONTROL: control_reg <= s_axi.WDATA[7:0];
                            ADDR_STATUS: status_reg <= status_reg & ~s_axi.WDATA[0];  // Clear on write
                            ADDR_COUNTER: counter_reg <= s_axi.WDATA;
                            ADDR_RELOAD: reload_reg <= s_axi.WDATA;
                            ADDR_COMPARE: compare_reg <= s_axi.WDATA;
                            ADDR_PRESCALER: prescaler_reg <= s_axi.WDATA[15:0];
                            ADDR_PWM0_DUTY: pwm_duty[0] <= s_axi.WDATA[15:0];
                            ADDR_PWM1_DUTY: pwm_duty[1] <= s_axi.WDATA[15:0];
                            ADDR_PWM2_DUTY: pwm_duty[2] <= s_axi.WDATA[15:0];
                            ADDR_PWM3_DUTY: pwm_duty[3] <= s_axi.WDATA[15:0];
                            ADDR_PWM_PERIOD: pwm_period_reg <= s_axi.WDATA[15:0];
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
                    case (raddr[5:2])
                        ADDR_CONTROL: s_axi.RDATA <= {24'h0, control_reg};
                        ADDR_STATUS: s_axi.RDATA <= {31'h0, status_reg};
                        ADDR_COUNTER: s_axi.RDATA <= counter_reg;
                        ADDR_RELOAD: s_axi.RDATA <= reload_reg;
                        ADDR_COMPARE: s_axi.RDATA <= compare_reg;
                        ADDR_PRESCALER: s_axi.RDATA <= {16'h0, prescaler_reg};
                        ADDR_PWM0_DUTY: s_axi.RDATA <= {16'h0, pwm_duty[0]};
                        ADDR_PWM1_DUTY: s_axi.RDATA <= {16'h0, pwm_duty[1]};
                        ADDR_PWM2_DUTY: s_axi.RDATA <= {16'h0, pwm_duty[2]};
                        ADDR_PWM3_DUTY: s_axi.RDATA <= {16'h0, pwm_duty[3]};
                        ADDR_PWM_PERIOD: s_axi.RDATA <= {16'h0, pwm_period_reg};
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
    
    // Prescaler
    always_ff @(posedge clk) begin
        if (rst) begin
            prescale_counter <= '0;
            timer_tick <= 1'b0;
        end else begin
            timer_tick <= 1'b0;
            if (control_reg[0]) begin  // Timer enabled
                if (prescale_counter >= prescaler_reg) begin
                    prescale_counter <= '0;
                    timer_tick <= 1'b1;
                end else begin
                    prescale_counter <= prescale_counter + 1;
                end
            end else begin
                prescale_counter <= '0;
            end
        end
    end
    
    // Timer counter
    always_ff @(posedge clk) begin
        if (rst) begin
            counter_reg <= '0;
            status_reg <= 1'b0;
        end else if (control_reg[0]) begin  // Timer enabled
            if (timer_tick) begin
                if (counter_reg >= compare_reg) begin
                    status_reg <= 1'b1;  // Set overflow flag
                    if (control_reg[1]) begin  // Auto-reload
                        counter_reg <= reload_reg;
                    end else begin
                        counter_reg <= '0;
                    end
                end else begin
                    counter_reg <= counter_reg + 1;
                end
            end
        end
    end
    
    // PWM generation
    always_ff @(posedge clk) begin
        if (rst) begin
            pwm_counter <= '0;
            pwm_out <= 4'b0000;
        end else begin
            if (pwm_counter >= pwm_period_reg) begin
                pwm_counter <= '0;
            end else begin
                pwm_counter <= pwm_counter + 1;
            end
            
            // Generate PWM outputs
            for (int i = 0; i < 4; i++) begin
                if (control_reg[3+i]) begin  // PWM enabled
                    pwm_out[i] <= (pwm_counter < pwm_duty[i]);
                end else begin
                    pwm_out[i] <= 1'b0;
                end
            end
        end
    end
    
    // Interrupt generation
    assign irq = control_reg[2] && status_reg;

endmodule