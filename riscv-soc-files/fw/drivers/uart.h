#ifndef UART_H
#define UART_H

#include <stdint.h>

// Base addresses
#define UART_BASE    0x40000000
#define SYSTEM_CLK   50000000

// Function prototypes
void uart_init(uint32_t baud_rate);
void uart_putc(char c);
char uart_getc(void);
int uart_getc_nonblock(void);
void uart_puts(const char* str);
void uart_printf(const char* format, ...);

#endif // UART_H