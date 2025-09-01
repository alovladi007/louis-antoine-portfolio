# Getting Started Guide

## Prerequisites

### Required Tools

1. **RISC-V Toolchain**
   ```bash
   # Download prebuilt toolchain
   wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.10.18/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14.tar.gz
   tar -xzf riscv64-unknown-elf-toolchain-*.tar.gz
   export PATH=$PATH:$PWD/riscv64-unknown-elf-toolchain/bin
   ```

2. **Simulation Tools**
   ```bash
   # Install Icarus Verilog
   sudo apt-get install iverilog gtkwave
   
   # Or install Verilator for faster simulation
   sudo apt-get install verilator
   ```

3. **FPGA Tools** (for synthesis)
   - Xilinx Vivado (for Artix-7/Zynq)
   - Intel Quartus (for Cyclone/Arria)

## Quick Start

### 1. Clone and Build

```bash
# Clone the repository
git clone <repository-url>
cd riscv-soc-project

# Build firmware
make firmware

# Build and run simulation
make simulation
make run_sim
```

### 2. View Waveforms

```bash
# Open waveform viewer
gtkwave build/soc_tb.vcd
```

### 3. Synthesize for FPGA

```bash
# For Xilinx FPGAs
make synthesis

# Program FPGA
vivado -mode tcl -source scripts/program_fpga.tcl
```

## Firmware Development

### Writing Your First Program

1. Create a new C file in `fw/apps/`:

```c
#include "../drivers/uart.h"
#include "../drivers/gpio.h"

int main(void) {
    uart_init(115200);
    uart_puts("Hello, RISC-V!\n");
    
    // Blink LED
    gpio_init();
    gpio_set_direction(0x01);  // Pin 0 as output
    
    while (1) {
        gpio_toggle_pin(0);
        for (volatile int i = 0; i < 1000000; i++);
    }
    
    return 0;
}
```

2. Build and load:

```bash
make firmware
make run_sim
```

### Using Peripherals

#### UART Communication
```c
// Initialize UART
uart_init(115200);

// Send data
uart_puts("Hello World\n");
uart_putc('A');

// Receive data
char c = uart_getc();
```

#### GPIO Control
```c
// Set pin direction (1=output, 0=input)
gpio_set_direction(0xFF);  // Lower 8 pins as outputs

// Write to pins
gpio_write(0x55);

// Read from pins
uint32_t value = gpio_read();

// Control individual pins
gpio_set_pin(3);
gpio_clear_pin(3);
gpio_toggle_pin(3);
```

#### Timer and PWM
```c
// Initialize timer for 1kHz tick
timer_init(1000);
timer_start();

// Set up PWM
pwm_init(0, 1000, 500);  // Channel 0, period=1000, duty=500 (50%)
pwm_enable(0);
```

#### SPI Communication
```c
// Initialize SPI
spi_init(1000000);  // 1MHz clock

// Transfer data
uint8_t rx_data = spi_transfer(0x42);

// Multi-byte transfer
uint8_t tx_buf[] = {0x01, 0x02, 0x03};
uint8_t rx_buf[3];
spi_transfer_buffer(tx_buf, rx_buf, 3);
```

## Simulation and Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test
make test TEST=uart

# Run with coverage
make coverage
```

### Debugging

1. **Using GDB**:
```bash
riscv64-unknown-elf-gdb build/firmware.elf
(gdb) target remote :3333
(gdb) load
(gdb) break main
(gdb) continue
```

2. **Using Waveforms**:
- Add signals to waveform viewer
- Look for pipeline stalls
- Check memory transactions
- Verify peripheral operations

## FPGA Deployment

### Board Support

Currently supported boards:
- Digilent Basys3
- Digilent Arty A7
- Terasic DE10-Nano

### Programming Steps

1. **Generate Bitstream**:
```bash
cd scripts
vivado -mode batch -source build_vivado.tcl
```

2. **Program FPGA**:
```bash
# Connect board via USB
# Open Vivado Hardware Manager
# Program with generated bitstream
```

3. **Connect Serial Terminal**:
```bash
# Find serial port
ls /dev/ttyUSB*

# Connect at 115200 baud
screen /dev/ttyUSB0 115200
```

## Performance Tuning

### Optimization Tips

1. **CPU Performance**:
   - Enable branch prediction
   - Optimize memory layout
   - Use hardware multiply/divide

2. **Memory Performance**:
   - Align data structures
   - Use DMA for bulk transfers
   - Optimize cache usage

3. **Power Optimization**:
   - Use clock gating
   - Reduce switching activity
   - Optimize voltage scaling

## Troubleshooting

### Common Issues

1. **Simulation hangs**:
   - Check for infinite loops
   - Verify reset sequence
   - Check clock generation

2. **UART not working**:
   - Verify baud rate settings
   - Check pin assignments
   - Test with loopback

3. **FPGA timing failures**:
   - Reduce clock frequency
   - Add pipeline stages
   - Optimize critical paths

## Next Steps

- Explore example applications in `fw/apps/`
- Read the [Architecture Documentation](architecture.md)
- Check [Performance Guide](performance.md)
- Join the community discussions