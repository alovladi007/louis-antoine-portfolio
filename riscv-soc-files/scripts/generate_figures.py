#!/usr/bin/env python3
"""
Generate figures and a markdown "scientific paper" summarizing the RISC-V SoC work.
The figures include: UART waveform, CPI model curves, throughput vs frequency, 
and transmission timing for "HELLO\n".
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import datetime
import zipfile

# Create output directory
outdir = Path("../docs/figures")
outdir.mkdir(parents=True, exist_ok=True)

# ---- Figure 1: UART waveform for one byte ('H' = 0x48) ----
CLK_HZ = 50_000_000
BAUD = 115200
div = CLK_HZ / BAUD  # ideal divider
bit_time = 1.0 / BAUD

# Construct a waveform: start(0), 8 data bits LSB first, stop(1)
byte = 0x48  # 'H'
bits = [0] + [(byte >> i) & 1 for i in range(8)] + [1]  # start, data0..7, stop

# Sample the waveform at, say, 20x over-sampling of baud
oversamp = 20
dt = bit_time / oversamp
samples_per_bit = oversamp
t = np.arange(0, bit_time * len(bits), dt)
wave = np.zeros_like(t)
for i, b in enumerate(bits):
    wave[i * samples_per_bit : (i + 1) * samples_per_bit] = b

plt.figure(figsize=(10, 4))
plt.step(t * 1e3, wave, where="post", linewidth=2, color='#2E86AB')
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("UART TXD level", fontsize=12)
plt.title("UART TX Waveform for 'H' (0x48) @115200 baud", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
fig1_path = outdir / "fig_uart_waveform.png"
plt.tight_layout()
plt.savefig(fig1_path, dpi=150)
plt.close()

# ---- Figure 2: CPI model vs mispredict rate, with different load-use stall rates ----
mispredict_rates = np.linspace(0, 0.3, 100)  # 0..30%
branch_penalty = 2.0
stall_rates = [0.05, 0.10, 0.20]
colors = ['#2E86AB', '#A23B72', '#F18F01']

plt.figure(figsize=(10, 6))
for idx, f_lu in enumerate(stall_rates):
    cpi = 1.0 + f_lu * 1.0 + mispredict_rates * branch_penalty
    plt.plot(mispredict_rates * 100, cpi, label=f"load-use={f_lu:.2f}", 
             linewidth=2, color=colors[idx])

plt.xlabel("Branch mispredict rate (%)", fontsize=12)
plt.ylabel("CPI (cycles per instruction)", fontsize=12)
plt.title("CPI vs. Branch Mispredict Rate (Penalty=2 cycles)", fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
fig2_path = outdir / "fig_cpi_vs_mispredict.png"
plt.tight_layout()
plt.savefig(fig2_path, dpi=150)
plt.close()

# ---- Figure 3: Throughput (MIPS) vs frequency for CPI=1.4 ----
freqs = np.array([25, 50, 75, 100, 125, 150]) * 1e6  # Hz
CPI = 1.4
ips = freqs / CPI
mips = ips / 1e6

plt.figure(figsize=(10, 6))
plt.plot(freqs / 1e6, mips, marker="o", markersize=8, linewidth=2, 
         color='#2E86AB', markerfacecolor='#F18F01', markeredgewidth=2)
plt.xlabel("Clock frequency (MHz)", fontsize=12)
plt.ylabel("Throughput (MIPS)", fontsize=12)
plt.title("Throughput vs. Frequency (CPI=1.4)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
# Add value labels on points
for f, m in zip(freqs/1e6, mips):
    plt.annotate(f'{m:.1f}', (f, m), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)
fig3_path = outdir / "fig_throughput_vs_freq.png"
plt.tight_layout()
plt.savefig(fig3_path, dpi=150)
plt.close()

# ---- Figure 4: Message transmission timing for "HELLO\n" ----
message = b"HELLO\n"
bits_per_char = 1 + 8 + 1  # start + data + stop
total_bits = bits_per_char * len(message)
total_time = total_bits * bit_time  # seconds

# Construct a bar-like plot: cumulative transmitted bits over time
time_axis = np.linspace(0, total_time, total_bits + 1)
cum_bits = np.arange(0, total_bits + 1)

plt.figure(figsize=(10, 6))
plt.step(time_axis * 1e3, cum_bits, where="post", linewidth=2, color='#A23B72')
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Cumulative bits sent", fontsize=12)
plt.title(f"TX Progress for 'HELLO\\n' @ {BAUD} baud", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
# Add annotations for each character boundary
for i, char in enumerate(message):
    char_time = (i + 1) * bits_per_char * bit_time * 1e3
    char_bits = (i + 1) * bits_per_char
    if i < len(message) - 1:
        plt.axvline(x=char_time, color='gray', linestyle='--', alpha=0.5)
        plt.text(char_time - 0.02, char_bits - 3, chr(char), fontsize=10, ha='right')
fig4_path = outdir / "fig_tx_timing.png"
plt.tight_layout()
plt.savefig(fig4_path, dpi=150)
plt.close()

# ---- Compose a Markdown paper ----
today = datetime.date.today().strftime("%B %d, %Y")
paper_md = f"""
# A Minimal RV32I RISC-V SoC with AXI-Lite Peripherals and UART Console: Design, Simulation, and Performance Modeling

**Author:** Louis Vladimir Antoine  
**Date:** {today}

---

## Abstract
We present the design and simulation of a minimal **RV32I** RISC-V system-on-chip (SoC) implemented for FPGA targets.
The SoC integrates a 5-stage–inspired multi-cycle core (implemented as an FSM in this revision), a dual-port on-chip ROM,
and an **AXI-Lite** interconnect to a **memory-mapped UART** peripheral. We describe the microarchitecture,
boot ROM, and verification approach; provide a micro-sequencer alternative for bring-up; and report analytical performance
models for CPI, throughput, and I/O timing. Reproducible artifacts (SystemVerilog RTL, testbenches, and a RISC-V boot ROM)
are included.

---

## 1. Introduction
RISC-V's open instruction set enables rapid prototyping of full systems on commodity FPGAs. This work demonstrates a compact
SoC with a minimal RV32I core, AXI-Lite peripheral fabric, and a UART console capable of printing text from both (i) a
firmware boot ROM and (ii) a simple **AXI-Lite micro-sequencer** used for early bring-up. The design targets 50–100 MHz
on mid-range FPGAs and prioritizes clarity and verifiability over peak performance.

Contributions:
1. A clean, educational SoC scaffold with **AXI-Lite UART** and **dual-port ROM**.
2. A minimal **RV32I core** that supports ADDI/LUI/AUIPC, LB/LW/SW, BEQ/BNE, and JAL sufficient to run a UART-printing boot ROM.
3. A **micro-sequencer** that prints without a CPU—useful for lab bring-up.
4. Analytical models and simulations for UART timing, CPI sensitivity, and throughput.

---

## 2. System Architecture
**Top-level** modules include: (i) a minimal RV32I core, (ii) a dual-port ROM (`bram2p_wrapper`) for instruction and read-only
data fetches, (iii) an **AXI-Lite crossbar** (`axi_lite_xbar`), and (iv) a **UART** implemented as `uart_tx` wrapped by
`axi_uart_tx` (TXDATA/STATUS/DIV registers). The UART base address is `0x4000_0000`.

**Memory map (excerpt):**
- `0x0000_0000`–… : Boot ROM (.text, .rodata)  
- `0x4000_0000` : UART TXDATA (W)  
- `0x4000_0004` : UART STATUS (R) `[0]=tx_busy`  
- `0x4000_0008` : UART DIV (R/W)

---

## 3. CPU Microarchitecture (Minimal RV32I)
The core is a compact multi-cycle FSM capable of running simple firmware. Supported opcodes in this revision are:
- **OP-IMM**: `ADDI`  
- **LUI**, **AUIPC**  
- **LOAD**: `LB`, `LW` (ROM/rodata and MMIO)  
- **STORE**: `SW` (MMIO)  
- **BRANCH**: `BEQ`, `BNE`  
- **JAL**

The pipeline can be extended to a full 5-stage implementation (IF/ID/EX/MEM/WB) with forwarding and load-use hazard handling.
AXI-Lite master ports are used for memory-mapped I/O (UART) accesses, while instruction and .rodata fetches come from the
dual-port ROM.

---

## 4. Firmware and Bring-Up
A **boot ROM** (`fw/boot/hello_uart.S`) prints `HELLO\\n` by polling `UART STATUS` and writing `UART TXDATA`. For earliest
bring-up without a CPU, an **AXI-Lite micro-sequencer** (`axi_seq_hello.sv`) generates the same traffic.

---

## 5. Methodology
We provide SystemVerilog testbenches for three scenarios:
1. **UART unit test** (`uart_tx_tb.sv`)  
2. **Micro-sequencer SoC test** (`soc_seq_uart_tb.sv`)  
3. **CPU-driven SoC test** (`soc_cpu_uart_tb.sv`)

Analytical performance is evaluated using standard first-order models of CPI and throughput, and UART serial timing derived
from the baud rate. The figures below are generated programmatically.

---

## 6. Results

### 6.1 UART Serial Timing
Figure 1 shows the TXD logic level over time for the character `'H'` at 115200 baud. The start bit is logic 0, followed by 8
LSB-first data bits, and a stop bit at logic 1.

![Figure 1](figures/fig_uart_waveform.png)

**Bit time** at 115200 baud is {1.0/BAUD:.6f} s ({(1.0/BAUD)*1e3:.3f} ms). A full 10-bit frame takes about {10*(1.0/BAUD)*1e3:.3f} ms.

### 6.2 CPI Sensitivity to Branch Misprediction and Load-Use Stalls
We model CPI as `CPI = 1 + f_load-use × 1 + f_mispredict × P`, with `P=2` cycles. Figure 2 shows CPI growth with mispredict
rate for several load-use stall fractions.

![Figure 2](figures/fig_cpi_vs_mispredict.png)

The model highlights the importance of hazard mitigation (forwarding/bypass and better branch handling) even in small cores.

### 6.3 Throughput vs Frequency
Assuming `CPI = 1.4`, Figure 3 plots MIPS vs clock frequency. In a mid-range FPGA, 50–100 MHz are readily achievable for
compact cores, translating to **~35–70 MIPS** in this configuration.

![Figure 3](figures/fig_throughput_vs_freq.png)

### 6.4 End-to-End UART "HELLO\\n" Transmission Time
For a 6-character string and a 10-bit frame per character, total bits = 60. Figure 4 shows cumulative bits vs time.
Total TX time is approximately **{total_time*1e3:.2f} ms** at 115200 baud.

![Figure 4](figures/fig_tx_timing.png)

---

## 7. Discussion
The design is intentionally minimal to maximize clarity and verifiability:
- AXI-Lite provides a clean separation between the core and peripherals.
- A micro-sequencer enables **board bring-up** without depending on CPU correctness.
- The CPU supports enough instructions to run a meaningful ROM while staying compact.

**Limitations:** No caches/TLB, no CSR/interrupts in this revision, limited ISA coverage, and no data SRAM. These are natural
next steps that will measurably improve CPI and functionality.

---

## 8. Future Work
1. Expand ISA coverage (full OP-IMM/OP, byte/halfword loads/stores, CSRs).  
2. Implement a lightweight **D-cache** or dedicated SRAM TCM with a proper LSU.  
3. Add **RISC-V Debug Module** for OpenOCD/GDB bring-up.  
4. Integrate a **branch predictor** and measure mispredict rates vs CPI.  
5. Synthesize to FPGA, record Fmax, LUT/FF/BRAM/DSP utilization, and power.  
6. Run **CoreMark**/Dhrystone and compare with open microcores at similar resource budgets.

---

## 9. Reproducibility and Artifacts
- RTL: `rtl/` (CPU, bus, memory, peripherals, tops)  
- Firmware: `fw/boot/` (RISC-V ASM, linker, Makefile)  
- Testbenches: `sim/tb/`  
- Docs: `docs/`

**Simulation quickstart (Icarus Verilog):**  
UART unit test:
```bash
iverilog -g2012 -o sim_out rtl/periph/uart_tx.sv sim/tb/uart_tx_tb.sv && vvp sim_out
```

Micro-sequencer SoC:
```bash
iverilog -g2012 -o sim_seq \\
  rtl/periph/uart_tx.sv \\
  rtl/periph/axi_uart_tx.sv \\
  rtl/bus/axi_lite_xbar.sv \\
  rtl/micro/axi_seq_hello.sv \\
  rtl/top/soc_seq_uart_top.sv \\
  sim/tb/soc_seq_uart_tb.sv && vvp sim_seq
```

CPU + ROM:
```bash
cd fw/boot && make && cd ../..
iverilog -g2012 -o sim_cpu \\
  rtl/mem/bram2p_wrapper.sv \\
  rtl/periph/uart_tx.sv \\
  rtl/periph/axi_uart_tx.sv \\
  rtl/bus/axi_lite_xbar.sv \\
  rtl/cpu/rv32_core_min.sv \\
  rtl/top/soc_cpu_top.sv \\
  sim/tb/soc_cpu_uart_tb.sv && vvp sim_cpu
```

---

## 10. Conclusion
We built a compact, educational RV32I SoC with a memory-mapped UART and demonstrated both CPU-driven and micro-sequencer-driven
console output. The provided analytical models and figures quantify UART timing and performance sensitivities and form a baseline
for future architectural enhancements. This artifact is designed to be extended toward a full 5-stage, interrupt-capable core
with caches and debug support.

---
"""

paper_path = outdir.parent / "riscv_soc_paper.md"
paper_path.write_text(paper_md)

# Also package the paper + figures into a zip
zip_path = outdir.parent / "riscv_soc_paper_assets.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(paper_path, arcname="riscv_soc_paper.md")
    z.write(fig1_path, arcname=fig1_path.name)
    z.write(fig2_path, arcname=fig2_path.name)
    z.write(fig3_path, arcname=fig3_path.name)
    z.write(fig4_path, arcname=fig4_path.name)

print(f"Generated paper: {paper_path}")
print(f"Generated figures in: {outdir}")
print(f"Created archive: {zip_path}")
print("\nFigures generated:")
print(f"  - {fig1_path.name}: UART waveform")
print(f"  - {fig2_path.name}: CPI vs mispredict rate")
print(f"  - {fig3_path.name}: Throughput vs frequency")
print(f"  - {fig4_path.name}: TX timing for HELLO")