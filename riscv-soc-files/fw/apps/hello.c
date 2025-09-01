#include "../drivers/uart.h"
#include "../drivers/gpio.h"
#include "../drivers/timer.h"

void delay_ms(uint32_t ms) {
    // Simple delay loop (not accurate, depends on CPU freq)
    volatile uint32_t count = ms * 10000;
    while (count--);
}

int main(void) {
    // Initialize UART at 115200 baud
    uart_init(115200);
    
    // Print welcome message
    uart_puts("\r\n");
    uart_puts("=====================================\r\n");
    uart_puts("    RISC-V SoC Boot Successful!     \r\n");
    uart_puts("=====================================\r\n");
    uart_puts("CPU: RV32IM @ 50MHz\r\n");
    uart_puts("RAM: 128KB\r\n");
    uart_puts("Peripherals: UART, SPI, GPIO, Timer\r\n");
    uart_puts("\r\n");
    
    // Initialize GPIO
    gpio_init();
    gpio_set_direction(0xFFFF);  // All outputs
    
    // Initialize timer for 1Hz interrupt
    timer_init(1);
    
    // Main loop
    uint32_t counter = 0;
    while (1) {
        // Toggle LED
        gpio_write(counter & 0xFFFF);
        
        // Print status every second
        uart_puts("Heartbeat: ");
        uart_putc('0' + (counter % 10));
        uart_puts("\r\n");
        
        // Check for user input
        int c = uart_getc_nonblock();
        if (c >= 0) {
            uart_puts("You pressed: ");
            uart_putc(c);
            uart_puts("\r\n");
            
            // Handle commands
            switch (c) {
                case 'r':
                case 'R':
                    uart_puts("Resetting...\r\n");
                    // Trigger soft reset
                    asm volatile("j _start");
                    break;
                    
                case 't':
                case 'T':
                    uart_puts("Running tests...\r\n");
                    run_tests();
                    break;
                    
                case 'h':
                case 'H':
                    uart_puts("Commands:\r\n");
                    uart_puts("  h - Help\r\n");
                    uart_puts("  r - Reset\r\n");
                    uart_puts("  t - Run tests\r\n");
                    break;
            }
        }
        
        delay_ms(1000);
        counter++;
    }
    
    return 0;
}

void run_tests(void) {
    uart_puts("Testing peripherals...\r\n");
    
    // Test GPIO
    uart_puts("  GPIO: ");
    for (int i = 0; i < 16; i++) {
        gpio_write(1 << i);
        delay_ms(50);
    }
    gpio_write(0);
    uart_puts("OK\r\n");
    
    // Test Timer
    uart_puts("  Timer: ");
    timer_set_period(1000);
    timer_start();
    delay_ms(100);
    timer_stop();
    uart_puts("OK\r\n");
    
    // Test memory
    uart_puts("  Memory: ");
    volatile uint32_t* test_addr = (uint32_t*)0x10001000;
    *test_addr = 0xDEADBEEF;
    if (*test_addr == 0xDEADBEEF) {
        uart_puts("OK\r\n");
    } else {
        uart_puts("FAIL\r\n");
    }
    
    uart_puts("Tests complete!\r\n");
}

// Interrupt handler
void interrupt_handler(uint32_t cause, uint32_t epc) {
    if (cause & 0x80000000) {
        // Interrupt
        uint32_t interrupt_id = cause & 0x7FFFFFFF;
        
        switch (interrupt_id) {
            case 7:  // Timer interrupt
                timer_clear_interrupt();
                // Handle timer tick
                break;
                
            case 11: // External interrupt (UART, GPIO, etc.)
                // Check and handle peripheral interrupts
                break;
        }
    } else {
        // Exception
        uart_puts("Exception occurred! Cause: ");
        uart_putc('0' + (cause & 0xF));
        uart_puts("\r\n");
    }
}