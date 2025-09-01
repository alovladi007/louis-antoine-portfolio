#ifndef GPIO_H
#define GPIO_H

#include <stdint.h>

#define GPIO_BASE 0x40030000

void gpio_init(void);
void gpio_set_direction(uint32_t mask);
void gpio_write(uint32_t value);
uint32_t gpio_read(void);
void gpio_set_pin(uint32_t pin);
void gpio_clear_pin(uint32_t pin);
void gpio_toggle_pin(uint32_t pin);

#endif // GPIO_H