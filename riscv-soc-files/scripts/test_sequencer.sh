#!/bin/bash

# Test script for the micro-sequencer that prints "HELLO"

echo "==================================="
echo "RISC-V SoC Micro-Sequencer Test"
echo "==================================="

# Create build directory
mkdir -p build

# Compile the micro-sequencer SoC
echo "Compiling micro-sequencer SoC..."
iverilog -g2012 -o build/sim_seq \
    -I rtl \
    rtl/periph/uart.sv \
    rtl/bus/axi_lite_if.sv \
    rtl/micro/axi_seq_hello.sv \
    rtl/top/soc_seq_uart_top.sv \
    sim/tb/soc_seq_uart_tb.sv

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Running simulation..."
cd build && vvp sim_seq

echo ""
echo "Simulation complete!"
echo "View waveforms with: gtkwave build/soc_seq_uart_tb.vcd"