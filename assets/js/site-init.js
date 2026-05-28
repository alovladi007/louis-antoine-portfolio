// Homepage initialization scripts.
//
// Extracted from inline <script> blocks in index.html so the parser doesn't
// have to evaluate ~16 KB of JS in-stream. Loaded with `defer` from index.html
// so this file executes after the HTML is parsed but before DOMContentLoaded
// fires — the DOMContentLoaded wrappers below are preserved as belt-and-
// suspenders, not because they're strictly required.
//
// Three independent feature blocks:
//   1. About-section tabs + auto-sliding carousel
//   2. Newsletter form submission via Web3Forms
//   3. Light analytics event tracking (project clicks, contact form, scroll
//      depth) wired to gtag + Microsoft Clarity if present
//
// Inline blocks NOT extracted (kept inline in index.html on purpose):
//   - YouTube IFrame player init (binds onYouTubeIframeAPIReady by global name)
//   - Google Analytics gtag.js stub
//   - Microsoft Clarity stub
// (Tawk.to widget was previously inline here; removed alongside the
//  positioning sweep because it shipped with B2B "Customer Support"
//  copy that didn't fit and overlapped the custom AI Assistant bubble.)
//
// =============================================================================
// 1. About Me section: tabs + auto-sliding carousel
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing About Me section with auto-sliding carousel...');

    // Clean up any play button symbols left over from the music control UI
    setTimeout(() => {
        document.querySelectorAll('*').forEach(el => {
            const text = el.textContent || el.innerText || '';
            if (text === '▶️' || text === '⏸️' || text === '▶' || text === '⏸') {
                el.style.display = 'none';
                el.remove();
            }
        });
    }, 100);

    // Carousel state — current slide index per section
    const carouselState = {
        intro: 0,
        work: 0,
        mission: 0
    };

    const slideCounts = {
        intro: 2,
        work: 3,
        mission: 2
    };

    // Initialize intro section
    const introSection = document.getElementById('intro-section');
    if (introSection) {
        introSection.style.display = 'block';
        introSection.classList.add('active');

        // Show first slide in all carousels
        document.querySelectorAll('.carousel-slide').forEach((slide) => {
            if (slide.parentElement.querySelector('.carousel-slide') === slide) {
                slide.classList.add('active');
            }
        });

        // Show first indicator in all carousels
        document.querySelectorAll('.carousel-indicators').forEach(container => {
            const firstIndicator = container.querySelector('.indicator');
            if (firstIndicator) {
                firstIndicator.classList.add('active');
            }
        });
    }

    // Simple tab switcher (exposed globally — called by onclick attrs in markup)
    window.switchAboutTab = function(sectionName) {
        console.log('Switching to:', sectionName);

        document.querySelectorAll('.about-section').forEach(section => {
            section.style.display = 'none';
            section.classList.remove('active');
        });

        const targetSection = document.getElementById(sectionName + '-section');
        if (targetSection) {
            targetSection.style.display = 'block';
            setTimeout(() => {
                targetSection.classList.add('active');
            }, 10);
        }

        document.querySelectorAll('.about-tab-3d').forEach(tab => {
            tab.classList.remove('active');
        });
        const activeTab = document.querySelector(`[data-section="${sectionName}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }
    };

    // Carousel navigation (exposed globally)
    window.navigateSlide = function(section, direction) {
        const slides = document.querySelectorAll(`#${section}-section .carousel-slide`);
        const indicators = document.querySelectorAll(`#${section}-section .indicator`);

        if (slides.length === 0) return;

        carouselState[section] += direction;
        if (carouselState[section] >= slides.length) {
            carouselState[section] = 0;
        } else if (carouselState[section] < 0) {
            carouselState[section] = slides.length - 1;
        }

        slides.forEach((slide, index) => {
            slide.classList.toggle('active', index === carouselState[section]);
        });

        indicators.forEach((indicator, index) => {
            indicator.classList.toggle('active', index === carouselState[section]);
        });
    };

    // Go to specific slide (exposed globally — used by indicator dots)
    window.goToSlide = function(section, slideIndex) {
        const slides = document.querySelectorAll(`#${section}-section .carousel-slide`);
        const indicators = document.querySelectorAll(`#${section}-section .indicator`);

        carouselState[section] = slideIndex - 1;

        slides.forEach((slide, index) => {
            slide.classList.toggle('active', index === carouselState[section]);
        });

        indicators.forEach((indicator, index) => {
            indicator.classList.toggle('active', index === carouselState[section]);
        });
    };

    // Auto-advance carousels every 4 seconds
    let autoPlayInterval = null;

    function startAutoPlay() {
        if (autoPlayInterval) {
            clearInterval(autoPlayInterval);
        }

        autoPlayInterval = setInterval(() => {
            const activeSection = document.querySelector('.about-section.active');
            if (activeSection) {
                const sectionId = activeSection.id.replace('-section', '');
                const slides = document.querySelectorAll(`#${sectionId}-section .carousel-slide`);

                if (slides && slides.length > 1) {
                    console.log(`Auto-advancing ${sectionId} carousel`);
                    window.navigateSlide(sectionId, 1);
                }
            }
        }, 4000);
    }

    setTimeout(() => {
        startAutoPlay();
        console.log('Carousel auto-play started');
    }, 500);

    // Restart auto-play when switching tabs
    const originalSwitchTab = window.switchAboutTab;
    window.switchAboutTab = function(sectionName) {
        originalSwitchTab(sectionName);
        startAutoPlay();
    };

    // Fallback: ensure carousel works even if initial setup failed
    setTimeout(() => {
        const activeSection = document.querySelector('.about-section.active');
        if (activeSection && !autoPlayInterval) {
            console.log('Fallback: Starting auto-play');
            startAutoPlay();
        }
    }, 2000);

    // Pause on hover
    document.querySelectorAll('.carousel-container').forEach(container => {
        container.addEventListener('mouseenter', () => {
            if (autoPlayInterval) {
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
            }
        });

        container.addEventListener('mouseleave', () => {
            startAutoPlay();
        });
    });
});

// =============================================================================
// 2. Newsletter form (Web3Forms)
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {
    const newsletterForm = document.getElementById('newsletterForm');
    const newsletterStatus = document.getElementById('newsletterStatus');

    if (newsletterForm) {
        newsletterForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const email = this.querySelector('input[name="email"]').value;
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;

            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Subscribing...';
            submitBtn.disabled = true;

            try {
                const response = await fetch('https://api.web3forms.com/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        access_key: '62d88081-cbe8-46c0-ab28-328ca1ad61b1',
                        email: email,
                        subject: 'New Newsletter Subscription',
                        message: 'New subscriber: ' + email
                    })
                });

                if (response.ok) {
                    newsletterStatus.style.display = 'block';
                    newsletterStatus.style.color = '#10b981';
                    newsletterStatus.innerHTML = '<i class="fas fa-check-circle"></i> Thank you for subscribing!';
                    this.reset();
                } else {
                    throw new Error('Failed');
                }
            } catch (error) {
                newsletterStatus.style.display = 'block';
                newsletterStatus.style.color = '#ef4444';
                newsletterStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Something went wrong. Please try again.';
            }

            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;

            setTimeout(() => {
                newsletterStatus.style.display = 'none';
            }, 5000);
        });
    }
});

// =============================================================================
// 3. Analytics event tracking (gtag + Microsoft Clarity)
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {
    // Track project link clicks
    document.querySelectorAll('.project-link').forEach(link => {
        link.addEventListener('click', function() {
            const projectName = this.closest('.project-card').querySelector('h3').textContent;
            if (typeof gtag !== 'undefined') {
                gtag('event', 'project_view', {
                    'project_name': projectName
                });
            }
            if (typeof clarity !== 'undefined') {
                clarity('set', 'project_view', projectName);
            }
        });
    });

    // Contact form submission via Web3Forms
    const contactForm = document.querySelector('.contact-form');
    const formStatus = document.getElementById('formStatus');

    if (contactForm) {
        contactForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = this.querySelector('.submit-btn');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
            submitBtn.disabled = true;

            try {
                const formData = new FormData(this);
                const response = await fetch('https://api.web3forms.com/submit', {
                    method: 'POST',
                    body: formData
                });

                console.log('Response status:', response.status);
                console.log('Response ok:', response.ok);

                const data = await response.json();
                console.log('Response data:', data);

                if (response.ok || data.success) {
                    formStatus.style.display = 'block';
                    formStatus.style.backgroundColor = '#10b981';
                    formStatus.style.color = 'white';
                    formStatus.innerHTML = '<i class="fas fa-check-circle"></i> Thank you! Your message has been sent successfully.';
                    this.reset();

                    if (typeof gtag !== 'undefined') {
                        gtag('event', 'contact_form_submit', {
                            'status': 'success'
                        });
                    }

                    setTimeout(() => {
                        formStatus.style.display = 'none';
                    }, 5000);
                } else {
                    throw new Error(data.message || 'Form submission failed');
                }
            } catch (error) {
                console.error('Form submission error:', error);
                formStatus.style.display = 'block';
                formStatus.style.backgroundColor = '#10b981';
                formStatus.style.color = 'white';
                formStatus.innerHTML = '<i class="fas fa-check-circle"></i> Your message has been sent! I will get back to you soon.';
                this.reset();

                if (typeof gtag !== 'undefined') {
                    gtag('event', 'contact_form_submit', {
                        'status': 'sent'
                    });
                }

                setTimeout(() => {
                    formStatus.style.display = 'none';
                }, 5000);
            } finally {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }

    // Track social link clicks
    document.querySelectorAll('.social-link').forEach(link => {
        link.addEventListener('click', function() {
            const platform = this.querySelector('i').className.includes('linkedin') ? 'LinkedIn' :
                           this.querySelector('i').className.includes('github') ? 'GitHub' :
                           this.querySelector('i').className.includes('youtube') ? 'YouTube' : 'Email';
            if (typeof gtag !== 'undefined') {
                gtag('event', 'social_click', {
                    'platform': platform
                });
            }
        });
    });

    // Track scroll depth at 25/50/75/100 milestones
    let maxScroll = 0;
    window.addEventListener('scroll', function() {
        const scrollPercent = Math.round((window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100);
        if (scrollPercent > maxScroll) {
            maxScroll = scrollPercent;
            if (maxScroll === 25 || maxScroll === 50 || maxScroll === 75 || maxScroll === 100) {
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'scroll_depth', {
                        'percent': maxScroll
                    });
                }
            }
        }
    });
});
