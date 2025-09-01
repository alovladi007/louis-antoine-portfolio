# RISC-V Boot Code (crt0.S)
# Sets up stack, initializes data sections, and jumps to main

.section .text.init
.global _start
.global _exit

_start:
    # Set global pointer
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    
    # Set stack pointer to end of RAM
    la sp, _stack_top
    
    # Clear BSS section
    la t0, _bss_start
    la t1, _bss_end
    
clear_bss:
    beq t0, t1, done_clear
    sw zero, 0(t0)
    addi t0, t0, 4
    j clear_bss
    
done_clear:
    # Copy initialized data from ROM to RAM
    la t0, _data_start
    la t1, _data_end
    la t2, _data_rom
    
copy_data:
    beq t0, t1, done_copy
    lw t3, 0(t2)
    sw t3, 0(t0)
    addi t0, t0, 4
    addi t2, t2, 4
    j copy_data
    
done_copy:
    # Initialize trap vector
    la t0, trap_handler
    csrw mtvec, t0
    
    # Enable interrupts
    li t0, 0x88      # MIE and MPIE bits
    csrs mstatus, t0
    li t0, 0x888     # Enable M-mode timer, software, and external interrupts
    csrw mie, t0
    
    # Call main
    call main
    
    # If main returns, go to exit
    j _exit

_exit:
    # Disable interrupts
    csrw mie, zero
    
    # Infinite loop
1:
    wfi
    j 1b

# Default trap handler
.align 4
trap_handler:
    # Save context
    addi sp, sp, -128
    sw ra, 0(sp)
    sw t0, 4(sp)
    sw t1, 8(sp)
    sw t2, 12(sp)
    sw t3, 16(sp)
    sw t4, 20(sp)
    sw t5, 24(sp)
    sw t6, 28(sp)
    sw a0, 32(sp)
    sw a1, 36(sp)
    sw a2, 40(sp)
    sw a3, 44(sp)
    sw a4, 48(sp)
    sw a5, 52(sp)
    sw a6, 56(sp)
    sw a7, 60(sp)
    
    # Call interrupt handler
    csrr a0, mcause
    csrr a1, mepc
    call interrupt_handler
    
    # Restore context
    lw ra, 0(sp)
    lw t0, 4(sp)
    lw t1, 8(sp)
    lw t2, 12(sp)
    lw t3, 16(sp)
    lw t4, 20(sp)
    lw t5, 24(sp)
    lw t6, 28(sp)
    lw a0, 32(sp)
    lw a1, 36(sp)
    lw a2, 40(sp)
    lw a3, 44(sp)
    lw a4, 48(sp)
    lw a5, 52(sp)
    lw a6, 56(sp)
    lw a7, 60(sp)
    addi sp, sp, 128
    
    mret

# Weak default interrupt handler
.weak interrupt_handler
interrupt_handler:
    ret