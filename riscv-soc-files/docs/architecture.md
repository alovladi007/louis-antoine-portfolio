# RISC-V SoC Architecture Documentation

## Overview

This document describes the architecture of a complete RISC-V System-on-Chip (SoC) implementation targeting FPGA platforms. The SoC features a 5-stage pipelined RV32IM processor core, AXI-Lite interconnect, and a comprehensive set of peripherals.

## CPU Core Architecture

### Pipeline Stages

The CPU implements a classic 5-stage pipeline:

1. **IF (Instruction Fetch)**
   - Fetches instructions from memory
   - Manages program counter (PC)
   - Handles branch target calculation

2. **ID (Instruction Decode)**
   - Decodes RISC-V instructions
   - Reads register file
   - Generates immediate values
   - Produces control signals

3. **EX (Execute)**
   - Performs ALU operations
   - Executes multiply/divide operations
   - Resolves branches
   - Handles data forwarding

4. **MEM (Memory Access)**
   - Performs load/store operations
   - Handles data alignment
   - Manages memory interface

5. **WB (Write Back)**
   - Writes results back to register file
   - Selects between ALU, memory, or PC+4

### Hazard Handling

- **Data Hazards**: Resolved through forwarding and stalling
- **Control Hazards**: Handled with branch prediction and pipeline flushing
- **Load-Use Hazards**: Managed with pipeline stalls

### ISA Support

- **RV32I**: Base integer instruction set
- **M Extension**: Hardware multiply/divide
- **CSR**: Control and status registers
- **Exceptions**: Basic exception handling

## Memory Architecture

### Memory Map

| Address Range | Size | Description |
|--------------|------|-------------|
| 0x0000_0000 - 0x0000_3FFF | 16KB | Boot ROM |
| 0x1000_0000 - 0x1001_FFFF | 128KB | Main SRAM |
| 0x2000_0000 - 0x2000_0FFF | 4KB | CLINT |
| 0x2001_0000 - 0x2001_FFFF | 64KB | PLIC |
| 0x4000_0000 - 0x4000_0FFF | 4KB | UART |
| 0x4001_0000 - 0x4001_0FFF | 4KB | SPI |
| 0x4002_0000 - 0x4002_0FFF | 4KB | I2C |
| 0x4003_0000 - 0x4003_0FFF | 4KB | GPIO |
| 0x4004_0000 - 0x4004_0FFF | 4KB | Timer/PWM |

### Memory Interface

- Separate instruction and data ports
- Single-cycle BRAM access for on-chip memory
- AXI-Lite interface for peripheral access

## Bus Architecture

### AXI-Lite Interconnect

- 1×N crossbar topology
- Address-based routing
- Single outstanding transaction support
- 32-bit address and data width

### Bus Protocol

- Standard AXI-Lite handshaking
- Separate read and write channels
- Response channel for write acknowledgment

## Peripheral Subsystem

### UART
- 16550-compatible interface
- Programmable baud rate
- TX/RX FIFOs
- Interrupt support

### SPI Master
- Modes 0-3 support
- Programmable clock rate
- 4 chip select lines
- 8-bit data width

### GPIO
- 16 configurable I/O pins
- Individual direction control
- Interrupt capability (edge/level)
- Pull-up/pull-down configuration

### Timer/PWM
- 32-bit counter
- Programmable prescaler
- Auto-reload capability
- 4 PWM channels
- Interrupt generation

### Interrupt Controller
- CLINT for timer and software interrupts
- PLIC for external interrupts
- Priority-based arbitration
- Nested interrupt support

## Performance Characteristics

### Target Metrics
- **Frequency**: 50-100 MHz on Artix-7
- **CPI**: ~1.4 (typical)
- **Dhrystone**: ≥0.7 DMIPS/MHz
- **CoreMark**: ≥1.5 CM/MHz

### Resource Utilization (Artix-7)
- **LUTs**: ~3000-4000
- **Registers**: ~2000-3000
- **BRAM**: 32-64 blocks
- **DSP**: 4-8 slices

## Debug Support

### Debug Features
- JTAG interface
- Hardware breakpoints
- Single-step execution
- Register and memory inspection

### Debug Registers
- Debug control and status
- Debug PC
- Debug scratch registers

## Boot Process

1. Reset vector at 0x0000_0000
2. Initialize stack pointer
3. Clear BSS section
4. Copy data section from ROM to RAM
5. Initialize trap vector
6. Enable interrupts
7. Jump to main()

## Power Management

### Clock Gating
- Peripheral-level clock gating
- Dynamic clock scaling support

### Power Domains
- Core power domain
- Peripheral power domain
- Always-on domain for critical functions

## Verification Strategy

### Unit Testing
- Individual module testbenches
- Functional coverage tracking
- Assertion-based verification

### Integration Testing
- Full SoC simulation
- ISA compliance tests
- Benchmark execution

### Hardware Validation
- FPGA prototyping
- Real-world application testing
- Performance characterization