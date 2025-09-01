# Vivado synthesis script for RISC-V SoC
# Target: Artix-7 (Basys3/Arty A7)

# Create project
create_project riscv_soc ./build/vivado -part xc7a35tcpg236-1 -force

# Add RTL sources
add_files -fileset sources_1 [glob ../rtl/cpu/*.sv]
add_files -fileset sources_1 [glob ../rtl/bus/*.sv]
add_files -fileset sources_1 [glob ../rtl/mem/*.sv]
add_files -fileset sources_1 [glob ../rtl/periph/*.sv]
add_files -fileset sources_1 [glob ../rtl/top/*.sv]

# Set top module
set_property top soc_top [current_fileset]

# Add constraints
add_files -fileset constrs_1 ../constr/basys3.xdc

# Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Run implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Report utilization and timing
open_run impl_1
report_utilization -file ./build/vivado/utilization.rpt
report_timing_summary -file ./build/vivado/timing.rpt

puts "Build complete! Bitstream available at ./build/vivado/riscv_soc.runs/impl_1/soc_top.bit"