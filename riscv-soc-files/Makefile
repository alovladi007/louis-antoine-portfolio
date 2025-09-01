# RISC-V SoC Project Makefile

# Tool configuration
CROSS_COMPILE ?= riscv64-unknown-elf-
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump

# Directories
RTL_DIR = rtl
SIM_DIR = sim
FW_DIR = fw
BUILD_DIR = build

# Compiler flags
CFLAGS = -march=rv32im -mabi=ilp32 -O2 -ffreestanding -nostdlib
ASFLAGS = -march=rv32im -mabi=ilp32
LDFLAGS = -T $(FW_DIR)/boot/link.ld -nostdlib

# Source files
FW_SRCS = $(FW_DIR)/boot/crt0.S \
          $(FW_DIR)/drivers/uart.c \
          $(FW_DIR)/drivers/gpio.c \
          $(FW_DIR)/drivers/timer.c \
          $(FW_DIR)/apps/hello.c

FW_OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(patsubst %.S,$(BUILD_DIR)/%.o,$(notdir $(FW_SRCS))))

# RTL files for simulation
RTL_SRCS = $(shell find $(RTL_DIR) -name "*.sv")

# Default target
all: firmware simulation

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Firmware targets
firmware: $(BUILD_DIR)/firmware.hex

$(BUILD_DIR)/%.o: $(FW_DIR)/boot/%.S | $(BUILD_DIR)
	$(AS) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(FW_DIR)/drivers/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(FW_DIR)/apps/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(FW_DIR)/drivers -c $< -o $@

$(BUILD_DIR)/firmware.elf: $(FW_OBJS)
	$(LD) $(LDFLAGS) $^ -o $@
	$(OBJDUMP) -d $@ > $(BUILD_DIR)/firmware.dis

$(BUILD_DIR)/firmware.bin: $(BUILD_DIR)/firmware.elf
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/firmware.hex: $(BUILD_DIR)/firmware.elf
	$(OBJCOPY) -O verilog $< $@

# Simulation targets
simulation: $(BUILD_DIR)/soc_sim

$(BUILD_DIR)/soc_sim: $(RTL_SRCS) $(SIM_DIR)/tb/soc_tb.sv | $(BUILD_DIR)
	iverilog -g2012 -o $@ \
		-I$(RTL_DIR) \
		$(RTL_SRCS) \
		$(SIM_DIR)/tb/soc_tb.sv

run_sim: $(BUILD_DIR)/soc_sim $(BUILD_DIR)/firmware.hex
	cd $(BUILD_DIR) && vvp soc_sim

# Micro-sequencer test
test_sequencer: | $(BUILD_DIR)
	@echo "Testing micro-sequencer..."
	iverilog -g2012 -o $(BUILD_DIR)/sim_seq \
		-I$(RTL_DIR) \
		$(RTL_DIR)/periph/uart.sv \
		$(RTL_DIR)/bus/axi_lite_if.sv \
		$(RTL_DIR)/micro/axi_seq_hello.sv \
		$(RTL_DIR)/top/soc_seq_uart_top.sv \
		$(SIM_DIR)/tb/soc_seq_uart_tb.sv
	cd $(BUILD_DIR) && vvp sim_seq

# Boot ROM
bootrom:
	$(MAKE) -C $(FW_DIR)/boot

# Synthesis targets (Vivado)
synthesis: $(RTL_SRCS)
	cd scripts && vivado -mode batch -source build_vivado.tcl

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "RISC-V SoC Build System"
	@echo "======================="
	@echo "Targets:"
	@echo "  all        - Build firmware and simulation"
	@echo "  firmware   - Build firmware binary and hex files"
	@echo "  simulation - Build simulation executable"
	@echo "  run_sim    - Run simulation"
	@echo "  synthesis  - Run FPGA synthesis (requires Vivado)"
	@echo "  clean      - Clean build artifacts"
	@echo "  help       - Show this help message"

.PHONY: all firmware simulation run_sim synthesis clean help