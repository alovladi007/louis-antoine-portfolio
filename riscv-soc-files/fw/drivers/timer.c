#include "timer.h"

#define TIMER_CONTROL   0x00
#define TIMER_STATUS    0x04
#define TIMER_COUNTER   0x08
#define TIMER_RELOAD    0x0C
#define TIMER_COMPARE   0x10
#define TIMER_PRESCALER 0x14
#define TIMER_PWM0_DUTY 0x18
#define TIMER_PWM1_DUTY 0x1C
#define TIMER_PWM2_DUTY 0x20
#define TIMER_PWM3_DUTY 0x24
#define TIMER_PWM_PERIOD 0x28

#define TIMER_EN        (1 << 0)
#define TIMER_AUTO_RLD  (1 << 1)
#define TIMER_IRQ_EN    (1 << 2)
#define TIMER_PWM0_EN   (1 << 3)
#define TIMER_PWM1_EN   (1 << 4)
#define TIMER_PWM2_EN   (1 << 5)
#define TIMER_PWM3_EN   (1 << 6)

static volatile uint32_t* const TIMER = (uint32_t*)TIMER_BASE;

void timer_init(uint32_t freq_hz) {
    // Calculate prescaler for desired frequency
    uint32_t prescaler = (50000000 / freq_hz) - 1;
    
    TIMER[TIMER_PRESCALER/4] = prescaler;
    TIMER[TIMER_RELOAD/4] = 0;
    TIMER[TIMER_COMPARE/4] = 0xFFFFFFFF;
    TIMER[TIMER_CONTROL/4] = TIMER_AUTO_RLD;
}

void timer_start(void) {
    TIMER[TIMER_CONTROL/4] |= TIMER_EN;
}

void timer_stop(void) {
    TIMER[TIMER_CONTROL/4] &= ~TIMER_EN;
}

void timer_set_period(uint32_t period) {
    TIMER[TIMER_COMPARE/4] = period;
}

uint32_t timer_get_count(void) {
    return TIMER[TIMER_COUNTER/4];
}

void timer_clear_interrupt(void) {
    TIMER[TIMER_STATUS/4] = 1;  // Clear overflow flag
}

void timer_enable_interrupt(void) {
    TIMER[TIMER_CONTROL/4] |= TIMER_IRQ_EN;
}

void timer_disable_interrupt(void) {
    TIMER[TIMER_CONTROL/4] &= ~TIMER_IRQ_EN;
}

void pwm_init(uint32_t channel, uint32_t period, uint32_t duty) {
    if (channel > 3) return;
    
    TIMER[TIMER_PWM_PERIOD/4] = period;
    TIMER[(TIMER_PWM0_DUTY + channel * 4)/4] = duty;
}

void pwm_set_duty(uint32_t channel, uint32_t duty) {
    if (channel > 3) return;
    
    TIMER[(TIMER_PWM0_DUTY + channel * 4)/4] = duty;
}

void pwm_enable(uint32_t channel) {
    if (channel > 3) return;
    
    TIMER[TIMER_CONTROL/4] |= (TIMER_PWM0_EN << channel);
}

void pwm_disable(uint32_t channel) {
    if (channel > 3) return;
    
    TIMER[TIMER_CONTROL/4] &= ~(TIMER_PWM0_EN << channel);
}