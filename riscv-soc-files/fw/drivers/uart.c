#include "uart.h"
#include <stdint.h>

// UART register offsets
#define UART_TX_DATA   0x00
#define UART_RX_DATA   0x04
#define UART_STATUS    0x08
#define UART_CONTROL   0x0C
#define UART_BAUD_DIV  0x10

// Status bits
#define UART_TX_BUSY   (1 << 0)
#define UART_RX_READY  (1 << 1)

// Control bits
#define UART_TX_EN     (1 << 0)
#define UART_RX_EN     (1 << 1)
#define UART_TX_IRQ_EN (1 << 2)
#define UART_RX_IRQ_EN (1 << 3)

static volatile uint32_t* const UART = (uint32_t*)UART_BASE;

void uart_init(uint32_t baud_rate) {
    // Calculate baud divisor
    uint32_t divisor = SYSTEM_CLK / baud_rate;
    
    // Set baud rate
    UART[UART_BAUD_DIV/4] = divisor;
    
    // Enable TX and RX
    UART[UART_CONTROL/4] = UART_TX_EN | UART_RX_EN;
}

void uart_putc(char c) {
    // Wait until TX is not busy
    while (UART[UART_STATUS/4] & UART_TX_BUSY);
    
    // Send character
    UART[UART_TX_DATA/4] = c;
}

char uart_getc(void) {
    // Wait until RX has data
    while (!(UART[UART_STATUS/4] & UART_RX_READY));
    
    // Read and return character
    return UART[UART_RX_DATA/4] & 0xFF;
}

int uart_getc_nonblock(void) {
    if (UART[UART_STATUS/4] & UART_RX_READY) {
        return UART[UART_RX_DATA/4] & 0xFF;
    }
    return -1;
}

void uart_puts(const char* str) {
    while (*str) {
        if (*str == '\n') {
            uart_putc('\r');
        }
        uart_putc(*str++);
    }
}

void uart_printf(const char* format, ...) {
    // Simplified printf - just handles %d, %x, %s, %c
    const char* p = format;
    char buf[32];
    
    while (*p) {
        if (*p == '%') {
            p++;
            switch (*p) {
                case 'd': {
                    // Decimal integer - simplified
                    uart_puts("[int]");
                    break;
                }
                case 'x': {
                    // Hexadecimal - simplified
                    uart_puts("[hex]");
                    break;
                }
                case 's': {
                    // String - simplified
                    uart_puts("[str]");
                    break;
                }
                case 'c': {
                    // Character - simplified
                    uart_putc('[');
                    uart_putc('c');
                    uart_putc(']');
                    break;
                }
                default:
                    uart_putc(*p);
            }
        } else {
            if (*p == '\n') {
                uart_putc('\r');
            }
            uart_putc(*p);
        }
        p++;
    }
}