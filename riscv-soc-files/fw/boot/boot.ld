/* boot.ld - Linker script for RISC-V boot ROM */

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    ROM (rx)  : ORIGIN = 0x00000000, LENGTH = 16K
    RAM (rwx) : ORIGIN = 0x10000000, LENGTH = 4K
}

SECTIONS
{
    /* Code section */
    .text : {
        *(.text.init)
        *(.text*)
        . = ALIGN(4);
    } > ROM
    
    /* Read-only data */
    .rodata : {
        *(.rodata*)
        . = ALIGN(4);
    } > ROM
    
    /* Data section (if any) */
    .data : {
        *(.data*)
        . = ALIGN(4);
    } > ROM
    
    /* BSS section */
    .bss : {
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
    } > RAM
    
    /* Stack */
    .stack : {
        . = ALIGN(16);
        _stack_bottom = .;
        . = . + 0x1000;  /* 4KB stack */
        _stack_top = .;
    } > RAM
}