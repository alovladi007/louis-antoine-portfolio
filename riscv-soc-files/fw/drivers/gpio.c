#include "gpio.h"

#define GPIO_DATA_IN    0x00
#define GPIO_DATA_OUT   0x04
#define GPIO_DIR        0x08
#define GPIO_IRQ_EN     0x0C
#define GPIO_IRQ_STATUS 0x10
#define GPIO_IRQ_TYPE   0x14
#define GPIO_IRQ_POL    0x18
#define GPIO_IRQ_CLEAR  0x1C

static volatile uint32_t* const GPIO = (uint32_t*)GPIO_BASE;

void gpio_init(void) {
    // Clear all outputs and set as inputs by default
    GPIO[GPIO_DATA_OUT/4] = 0;
    GPIO[GPIO_DIR/4] = 0;
    GPIO[GPIO_IRQ_EN/4] = 0;
}

void gpio_set_direction(uint32_t mask) {
    GPIO[GPIO_DIR/4] = mask;
}

void gpio_write(uint32_t value) {
    GPIO[GPIO_DATA_OUT/4] = value;
}

uint32_t gpio_read(void) {
    return GPIO[GPIO_DATA_IN/4];
}

void gpio_set_pin(uint32_t pin) {
    if (pin < 32) {
        GPIO[GPIO_DATA_OUT/4] |= (1 << pin);
    }
}

void gpio_clear_pin(uint32_t pin) {
    if (pin < 32) {
        GPIO[GPIO_DATA_OUT/4] &= ~(1 << pin);
    }
}

void gpio_toggle_pin(uint32_t pin) {
    if (pin < 32) {
        GPIO[GPIO_DATA_OUT/4] ^= (1 << pin);
    }
}