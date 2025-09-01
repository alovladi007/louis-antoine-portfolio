/* RISC-V SoC Linker Script */

MEMORY
{
    ROM (rx)  : ORIGIN = 0x00000000, LENGTH = 16K
    RAM (rwx) : ORIGIN = 0x10000000, LENGTH = 128K
}

ENTRY(_start)

SECTIONS
{
    /* Code section in ROM */
    .text : {
        *(.text.init)
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    } > ROM
    
    /* Initialized data (copied from ROM to RAM) */
    .data : AT(_etext) {
        _data_start = .;
        *(.data*)
        . = ALIGN(4);
        _data_end = .;
    } > RAM
    
    _data_rom = LOADADDR(.data);
    
    /* BSS section in RAM */
    .bss : {
        _bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > RAM
    
    /* Stack */
    .stack : {
        . = ALIGN(16);
        _stack_bottom = .;
        . = . + 0x4000;  /* 16KB stack */
        _stack_top = .;
    } > RAM
    
    /* Global pointer */
    __global_pointer$ = MIN(_data_start + 0x800,
                            MAX(_data_start, _bss_end - 0x800));
}