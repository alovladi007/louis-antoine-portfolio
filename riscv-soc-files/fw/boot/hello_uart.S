# hello_uart.S - RISC-V (RV32I) boot program to print "HELLO\n" via UART
# 
# Memory Map:
#   UART_BASE: 0x40000000
#   UART_TX:   0x40000000 (TX Data Register)
#   UART_RX:   0x40000004 (RX Data Register)  
#   UART_ST:   0x40000008 (Status Register, bit 0 = tx_busy)

.section .text.init
.global _start

# UART MMIO addresses
.equ UART_BASE, 0x40000000
.equ UART_TX,   0x40000000
.equ UART_RX,   0x40000004
.equ UART_ST,   0x40000008

_start:
    # Load address of message
    la   t0, hello_msg      # t0 = address of message
    
print_loop:
    # Load next character
    lb   t1, 0(t0)          # t1 = *t0 (load byte)
    beq  t1, x0, done       # if null terminator, we're done
    
wait_uart:
    # Poll UART status until ready
    li   t2, UART_ST        # t2 = UART status register address
    lw   t3, 0(t2)          # t3 = status register value
    andi t3, t3, 1          # check tx_busy bit
    bnez t3, wait_uart      # if busy, keep waiting
    
    # Transmit character
    li   t2, UART_TX        # t2 = UART TX register address
    sw   t1, 0(t2)          # write character to UART
    
    # Move to next character
    addi t0, t0, 1          # increment string pointer
    j    print_loop         # continue loop
    
done:
    # Infinite loop when done
    j    done

# Data section with message
.section .rodata
.align 4
hello_msg:
    .string "HELLO\n"
    .byte 0  # Null terminator