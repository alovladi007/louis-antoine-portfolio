# Makefile for building RISC-V boot ROM
# Requires riscv64-unknown-elf toolchain

# Cross-compiler prefix
CROSS_COMPILE ?= riscv64-unknown-elf-

# Tools
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump

# Compiler/Assembler flags
ASFLAGS = -march=rv32i -mabi=ilp32
LDFLAGS = -T boot.ld -nostdlib -static

# Targets
all: boot.hex boot.bin boot.dump

# Assemble boot code
boot.o: hello_uart.S
	$(AS) $(ASFLAGS) -o $@ $<

# Link to ELF
boot.elf: boot.o boot.ld
	$(LD) $(LDFLAGS) -o $@ boot.o

# Generate hex file for Verilog $readmemh
boot.hex: boot.elf
	$(OBJCOPY) -O verilog $< $@
	@echo "Generated boot.hex for simulation"

# Generate binary file
boot.bin: boot.elf
	$(OBJCOPY) -O binary $< $@
	@echo "Generated boot.bin"

# Generate disassembly for debugging
boot.dump: boot.elf
	$(OBJDUMP) -d $< > $@
	@echo "Generated boot.dump (disassembly)"

# Clean build artifacts
clean:
	rm -f *.o *.elf *.hex *.bin *.dump

# Show memory usage
size: boot.elf
	$(CROSS_COMPILE)size $<

.PHONY: all clean size