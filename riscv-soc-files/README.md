# RISC-V SoC on FPGA - Complete Implementation

## Overview
Complete RV32IM RISC-V System-on-Chip implementation targeting Artix-7/Cyclone V FPGAs with:
- 5-stage pipelined CPU core with hazard detection and forwarding
- AXI4/AXI4-Lite interconnect fabric
- Full peripheral suite (UART, SPI, I2C, GPIO, Timer, PWM)
- Interrupt controllers (CLINT, PLIC)
- 128KB on-chip SRAM
- Debug support

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                        SoC Top                         │
│                                                        │
│  ┌─────────┐   AXI4   ┌─────────────┐                 │
│  │ Clock & │──────────▶ BRAM Ctrl   │──┐               │
│  │ Reset   │          └─────────────┘  │  128KB SRAM  │
│  └─────────┘                           ▼               │
│                                    ┌─────────────┐     │
│  ┌─────────┐  IF/ID/EX/MEM/WB     │  TCM/BRAM   │     │
│  │ RISC-V  │══════════════════════│ (Instr+Data)│     │
│  │ CPU     │                      └─────────────┘     │
│  └─────────┘                            ▲              │
│        ▲          AXI-Lite              │              │
│        │  ┌─────────────────────────────┴──────┐      │
│        │  │    AXI-Lite Interconnect           │      │
│        │  └───┬────────┬───────┬───────┬──────┘      │
│        ▼      ▼        ▼       ▼       ▼              │
│   ┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────────┐   │
│   │  PLIC  │ │ UART │ │  SPI │ │  I2C │ │  GPIO  │   │
│   └────────┘ └──────┘ └──────┘ └──────┘ └────────┘   │
│        ▲         ▲        ▲       ▲         ▲         │
│        │         │        │       │         │         │
│   ┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────────┐   │
│   │  CLINT │ │TIMER │ │  PWM │ │  JTAG│ │  DBG   │   │
│   └────────┘ └──────┘ └──────┘ └──────┘ └────────┘   │
└────────────────────────────────────────────────────────┘
```

## Performance Targets
- **Frequency**: 50-100 MHz on mid-range FPGA
- **Dhrystone**: ≥ 0.7 DMIPS/MHz
- **CoreMark**: ≥ 1.5 CM/MHz
- **UART**: 115200 bps
- **SPI**: ≥ 10 MHz
- **I²C**: 100/400 kHz

## Memory Map
| Region | Base Address | Size | Description |
|--------|-------------|------|-------------|
| Boot ROM | 0x0000_0000 | 16 KB | Reset vector @ 0x0000_0000 |
| SRAM | 0x1000_0000 | 128 KB | Code + Data |
| CLINT | 0x2000_0000 | 4 KB | Timer/Software interrupts |
| PLIC | 0x2001_0000 | 64 KB | External interrupt controller |
| UART | 0x4000_0000 | 4 KB | 16550-like registers |
| SPI | 0x4001_0000 | 4 KB | Mode 0/3, up to 25 MHz |
| I²C | 0x4002_0000 | 4 KB | 100/400 kHz |
| GPIO | 0x4003_0000 | 4 KB | IN/OUT/Direction |
| TIMER/PWM | 0x4004_0000 | 4 KB | Periodic + PWM channels |
| Debug | 0x5FFF_0000 | 64 KB | RISC-V Debug Module |

## Directory Structure
```
riscv-soc-project/
├─ rtl/
│  ├─ cpu/           # 5-stage pipeline, ALU, regfile, CSR
│  ├─ bus/           # AXI4-lite interconnect
│  ├─ mem/           # BRAM wrappers
│  ├─ periph/        # UART, SPI, I2C, GPIO, timers
│  └─ top/           # SoC top-level
├─ sim/
│  ├─ tb/            # SystemVerilog testbenches
│  └─ cocotb/        # Python-based tests
├─ fw/
│  ├─ boot/          # Bootloader and crt0
│  ├─ drivers/       # Peripheral drivers
│  └─ apps/          # Demo applications
├─ scripts/          # Build and synthesis scripts
├─ constr/           # FPGA constraints
└─ docs/             # Documentation
```

## Quick Start

### Simulation
```bash
# Run UART test
make -C sim test_uart

# Run full SoC test
make -C sim test_soc

# Run ISA compliance tests
make -C sim test_isa
```

### Synthesis
```bash
# For Xilinx (Vivado)
cd scripts && vivado -mode batch -source build_vivado.tcl

# For Intel (Quartus)
cd scripts && quartus_sh -t build_quartus.tcl
```

### Firmware
```bash
cd fw/apps
make blinky
make dhrystone
make coremark
```

## Implementation Status
- [x] 5-stage pipeline CPU core
- [x] RV32I base ISA
- [x] M extension (multiply/divide)
- [x] CSR and exceptions
- [x] AXI-Lite interconnect
- [x] UART with interrupts
- [x] SPI master
- [x] I2C master
- [x] GPIO with interrupts
- [x] Timer and PWM
- [x] CLINT (Core Local Interruptor)
- [x] PLIC (Platform Level Interrupt Controller)
- [x] Debug module
- [x] Dhrystone/CoreMark benchmarks

## Advanced Features

### Micro-Sequencer
The project includes an AXI-Lite micro-sequencer that can autonomously print messages via UART without CPU intervention. This demonstrates:
- Hardware state machines
- AXI-Lite master implementation
- UART polling and control

To test the micro-sequencer:
```bash
make test_sequencer
```

### Boot ROM
A complete boot ROM implementation in RISC-V assembly that:
- Initializes the system
- Prints "HELLO\n" via UART
- Demonstrates MMIO programming

To build the boot ROM (requires RISC-V toolchain):
```bash
make bootrom
```

### Memory Initialization
The BRAM modules support initialization from hex files, allowing pre-loaded programs for immediate execution upon reset.

## License
MIT License - See LICENSE file for details

## Author
Louis Antoine - 2025