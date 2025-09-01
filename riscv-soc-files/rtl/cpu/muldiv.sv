`timescale 1ns/1ps

module muldiv #(
    parameter XLEN = 32
)(
    input  logic             clk,
    input  logic             rst,
    
    // Control signals
    input  logic             start,
    input  logic [2:0]       op,     // 000:MUL, 001:MULH, 010:MULHSU, 011:MULHU, 100:DIV, 101:DIVU, 110:REM, 111:REMU
    input  logic [XLEN-1:0]  a,
    input  logic [XLEN-1:0]  b,
    
    // Results
    output logic [XLEN-1:0]  result,
    output logic             busy,
    output logic             valid
);

    // Operation encoding
    localparam OP_MUL    = 3'b000;
    localparam OP_MULH   = 3'b001;
    localparam OP_MULHSU = 3'b010;
    localparam OP_MULHU  = 3'b011;
    localparam OP_DIV    = 3'b100;
    localparam OP_DIVU   = 3'b101;
    localparam OP_REM    = 3'b110;
    localparam OP_REMU   = 3'b111;
    
    // Internal signals
    logic [63:0] mul_result;
    logic [31:0] div_quotient;
    logic [31:0] div_remainder;
    logic [5:0]  div_counter;
    logic        div_busy;
    logic        is_mul_op;
    logic        is_div_op;
    
    // Decode operation type
    assign is_mul_op = (op[2] == 1'b0);
    assign is_div_op = (op[2] == 1'b1);
    
    // Multiplication (single-cycle using DSP blocks)
    always_comb begin
        case (op[1:0])
            2'b00: mul_result = $signed(a) * $signed(b);           // MUL/MULH
            2'b01: mul_result = $signed(a) * $signed(b);           // MULH
            2'b10: mul_result = $signed(a) * $unsigned(b);         // MULHSU
            2'b11: mul_result = $unsigned(a) * $unsigned(b);       // MULHU
        endcase
    end
    
    // Division unit (simple iterative divider)
    logic [31:0] dividend_reg;
    logic [31:0] divisor_reg;
    logic [31:0] quotient_reg;
    logic [31:0] remainder_reg;
    logic        div_sign;
    logic        div_unsigned;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            div_busy <= 1'b0;
            div_counter <= '0;
            dividend_reg <= '0;
            divisor_reg <= '0;
            quotient_reg <= '0;
            remainder_reg <= '0;
            div_sign <= 1'b0;
            div_unsigned <= 1'b0;
        end else begin
            if (start && is_div_op && !div_busy) begin
                div_busy <= 1'b1;
                div_counter <= 6'd32;
                div_unsigned <= op[0];
                
                // Handle signed division
                if (!op[0] && ((op == OP_DIV) || (op == OP_REM))) begin
                    div_sign <= a[31] ^ b[31];
                    dividend_reg <= a[31] ? -a : a;
                    divisor_reg <= b[31] ? -b : b;
                end else begin
                    div_sign <= 1'b0;
                    dividend_reg <= a;
                    divisor_reg <= b;
                end
                
                quotient_reg <= '0;
                remainder_reg <= '0;
                
            end else if (div_busy) begin
                if (div_counter > 0) begin
                    // Shift and subtract algorithm
                    remainder_reg <= {remainder_reg[30:0], dividend_reg[31]};
                    dividend_reg <= {dividend_reg[30:0], 1'b0};
                    
                    if (remainder_reg >= divisor_reg) begin
                        remainder_reg <= remainder_reg - divisor_reg;
                        quotient_reg <= {quotient_reg[30:0], 1'b1};
                    end else begin
                        quotient_reg <= {quotient_reg[30:0], 1'b0};
                    end
                    
                    div_counter <= div_counter - 1;
                end else begin
                    div_busy <= 1'b0;
                end
            end
        end
    end
    
    // Result selection
    always_comb begin
        if (is_mul_op) begin
            if (op == OP_MUL) begin
                result = mul_result[31:0];
            end else begin
                result = mul_result[63:32];
            end
        end else begin
            // Division results
            if ((op == OP_DIV) || (op == OP_DIVU)) begin
                result = (div_sign && !div_unsigned) ? -quotient_reg : quotient_reg;
            end else begin
                result = (a[31] && !div_unsigned) ? -remainder_reg : remainder_reg;
            end
        end
    end
    
    // Control signals
    assign busy = div_busy;
    assign valid = is_mul_op ? start : (div_busy && (div_counter == 0));

endmodule