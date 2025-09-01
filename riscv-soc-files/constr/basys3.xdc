# Basys3 Board Constraints File

# Clock signal
set_property PACKAGE_PIN W5 [get_ports clk_in]
set_property IOSTANDARD LVCMOS33 [get_ports clk_in]
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} -add [get_ports clk_in]

# Reset button (BTNC)
set_property PACKAGE_PIN U18 [get_ports rst_in]
set_property IOSTANDARD LVCMOS33 [get_ports rst_in]

# UART
set_property PACKAGE_PIN B18 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]
set_property PACKAGE_PIN A18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]

# LEDs
set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]
set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]
set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]
set_property PACKAGE_PIN V19 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]
set_property PACKAGE_PIN W18 [get_ports {led[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[4]}]
set_property PACKAGE_PIN U15 [get_ports {led[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[5]}]
set_property PACKAGE_PIN U14 [get_ports {led[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[6]}]
set_property PACKAGE_PIN V14 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[7]}]

# Switches (GPIO inputs)
set_property PACKAGE_PIN V17 [get_ports {gpio_in[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[0]}]
set_property PACKAGE_PIN V16 [get_ports {gpio_in[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[1]}]
set_property PACKAGE_PIN W16 [get_ports {gpio_in[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[2]}]
set_property PACKAGE_PIN W17 [get_ports {gpio_in[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[3]}]
set_property PACKAGE_PIN W15 [get_ports {gpio_in[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[4]}]
set_property PACKAGE_PIN V15 [get_ports {gpio_in[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[5]}]
set_property PACKAGE_PIN W14 [get_ports {gpio_in[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[6]}]
set_property PACKAGE_PIN W13 [get_ports {gpio_in[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {gpio_in[7]}]

# Pmod Header JA (SPI)
set_property PACKAGE_PIN J1 [get_ports spi_cs_n[0]]
set_property IOSTANDARD LVCMOS33 [get_ports spi_cs_n[0]]
set_property PACKAGE_PIN L2 [get_ports spi_mosi]
set_property IOSTANDARD LVCMOS33 [get_ports spi_mosi]
set_property PACKAGE_PIN J2 [get_ports spi_miso]
set_property IOSTANDARD LVCMOS33 [get_ports spi_miso]
set_property PACKAGE_PIN G2 [get_ports spi_sclk]
set_property IOSTANDARD LVCMOS33 [get_ports spi_sclk]

# Pmod Header JB (PWM outputs)
set_property PACKAGE_PIN A14 [get_ports {pwm_out[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pwm_out[0]}]
set_property PACKAGE_PIN A16 [get_ports {pwm_out[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pwm_out[1]}]
set_property PACKAGE_PIN B15 [get_ports {pwm_out[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pwm_out[2]}]
set_property PACKAGE_PIN B16 [get_ports {pwm_out[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pwm_out[3]}]

# Timing constraints
create_generated_clock -name spi_sclk -source [get_pins dut/u_spi/spi_sclk_reg/C] -divide_by 2 [get_ports spi_sclk]

# False paths for async signals
set_false_path -from [get_ports rst_in]
set_false_path -from [get_ports {gpio_in[*]}]
set_false_path -to [get_ports {led[*]}]

# Configuration
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]