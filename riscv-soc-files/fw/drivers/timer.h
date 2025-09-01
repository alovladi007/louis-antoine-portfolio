#ifndef TIMER_H
#define TIMER_H

#include <stdint.h>

#define TIMER_BASE 0x40040000

void timer_init(uint32_t freq_hz);
void timer_start(void);
void timer_stop(void);
void timer_set_period(uint32_t period);
uint32_t timer_get_count(void);
void timer_clear_interrupt(void);
void timer_enable_interrupt(void);
void timer_disable_interrupt(void);

// PWM functions
void pwm_init(uint32_t channel, uint32_t period, uint32_t duty);
void pwm_set_duty(uint32_t channel, uint32_t duty);
void pwm_enable(uint32_t channel);
void pwm_disable(uint32_t channel);

#endif // TIMER_H