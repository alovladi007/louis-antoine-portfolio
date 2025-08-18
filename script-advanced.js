// Advanced Portfolio Features
document.addEventListener('DOMContentLoaded', function() {
    console.log('Advanced features loaded');
    
    // Initialize AOS (Animate On Scroll)
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 1000,
            easing: 'ease-in-out',
            once: false,
            mirror: true
        });
    }
    
    // Add data-aos attributes to existing elements for animations
    document.querySelectorAll('.project-card').forEach((card, index) => {
        card.setAttribute('data-aos', 'fade-up');
        card.setAttribute('data-aos-delay', index * 100);
    });
    
    document.querySelectorAll('.certification-card').forEach((card, index) => {
        card.setAttribute('data-aos', 'fade-up');
        card.setAttribute('data-aos-delay', index * 100);
    });
    
    // Enhanced typing effect for hero title using Typed.js
    if (typeof Typed !== 'undefined') {
        const heroTitle = document.querySelector('.hero-title');
        if (heroTitle) {
            const originalText = heroTitle.textContent;
            heroTitle.textContent = '';
            new Typed('.hero-title', {
                strings: [originalText],
                typeSpeed: 100,
                showCursor: true,
                cursorChar: '|',
                onComplete: function(self) {
                    setTimeout(() => {
                        self.cursor.style.display = 'none';
                    }, 1000);
                }
            });
        }
    }
    
    // GSAP Animations for hero section
    if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
        
        // Parallax effect for hero background
        gsap.to('.hero-bg-elements', {
            yPercent: -50,
            ease: 'none',
            scrollTrigger: {
                trigger: '.hero',
                start: 'top top',
                end: 'bottom top',
                scrub: true
            }
        });
        
        // Animate floating shapes
        gsap.to('.floating-shape', {
            y: '+=30',
            rotation: 360,
            duration: 10,
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut',
            stagger: {
                each: 2,
                from: 'random'
            }
        });
        
        // Animate gradient orbs
        gsap.to('.gradient-orb', {
            scale: 1.2,
            opacity: 0.7,
            duration: 4,
            repeat: -1,
            yoyo: true,
            ease: 'power1.inOut',
            stagger: 1
        });
    }
    
    // 3D Tilt effect on project cards
    const cards = document.querySelectorAll('.project-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
            card.style.transition = 'transform 0.1s ease';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
            card.style.transition = 'transform 0.3s ease';
        });
    });
    
    // Enhanced skill progress bars animation
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const skillObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const skillBars = entry.target.querySelectorAll('.skill-progress');
                skillBars.forEach((bar, index) => {
                    setTimeout(() => {
                        const width = bar.getAttribute('data-width');
                        bar.style.width = width;
                        bar.style.transition = 'width 1.5s cubic-bezier(0.4, 0, 0.2, 1)';
                        
                        // Add percentage counter animation
                        const percentage = parseInt(width);
                        let current = 0;
                        const increment = percentage / 50;
                        const counter = setInterval(() => {
                            current += increment;
                            if (current >= percentage) {
                                current = percentage;
                                clearInterval(counter);
                            }
                            if (bar.nextElementSibling && bar.nextElementSibling.classList.contains('skill-percentage')) {
                                bar.nextElementSibling.textContent = Math.floor(current) + '%';
                            }
                        }, 30);
                    }, index * 100);
                });
            }
        });
    }, observerOptions);
    
    const skillsSection = document.querySelector('.skills');
    if (skillsSection) {
        skillObserver.observe(skillsSection);
    }
    
    // Smooth reveal for sections
    const sections = document.querySelectorAll('section');
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('section-visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    sections.forEach(section => {
        section.classList.add('section-hidden');
        sectionObserver.observe(section);
    });
    
    // Enhanced navbar with hide/show on scroll
    let lastScroll = 0;
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
        
        if (currentScroll > lastScroll && currentScroll > 500) {
            navbar.classList.add('navbar-hidden');
        } else {
            navbar.classList.remove('navbar-hidden');
        }
        
        lastScroll = currentScroll;
    });
    
    // Add loading animation to images
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.classList.add('loaded');
        });
        if (img.complete) {
            img.classList.add('loaded');
        }
    });
    
    // Enhanced hover effects for buttons
    const buttons = document.querySelectorAll('.cta-button, .btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Add magnetic effect to nav links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            
            this.style.transform = `translate(${x * 0.3}px, ${y * 0.3}px)`;
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'translate(0, 0)';
        });
    });
    
    // Initialize Particles.js if available
    if (typeof particlesJS !== 'undefined' && !document.getElementById('particles-js')) {
        // Create particles container in hero section
        const heroSection = document.querySelector('.hero');
        if (heroSection) {
            const particlesDiv = document.createElement('div');
            particlesDiv.id = 'particles-js';
            particlesDiv.style.position = 'absolute';
            particlesDiv.style.width = '100%';
            particlesDiv.style.height = '100%';
            particlesDiv.style.top = '0';
            particlesDiv.style.left = '0';
            particlesDiv.style.zIndex = '1';
            particlesDiv.style.pointerEvents = 'none';
            heroSection.insertBefore(particlesDiv, heroSection.firstChild);
            
            particlesJS('particles-js', {
                particles: {
                    number: { value: 50, density: { enable: true, value_area: 800 } },
                    color: { value: '#667eea' },
                    shape: { type: 'circle' },
                    opacity: { value: 0.3, random: true },
                    size: { value: 3, random: true },
                    line_linked: {
                        enable: true,
                        distance: 150,
                        color: '#667eea',
                        opacity: 0.2,
                        width: 1
                    },
                    move: {
                        enable: true,
                        speed: 2,
                        direction: 'none',
                        random: false,
                        straight: false,
                        out_mode: 'out',
                        bounce: false
                    }
                },
                interactivity: {
                    detect_on: 'canvas',
                    events: {
                        onhover: { enable: true, mode: 'grab' },
                        onclick: { enable: true, mode: 'push' },
                        resize: true
                    },
                    modes: {
                        grab: { distance: 140, line_linked: { opacity: 0.5 } },
                        push: { particles_nb: 4 }
                    }
                }
            });
        }
    }
});

// Add necessary CSS for new effects
const style = document.createElement('style');
style.textContent = `
    /* Section animations */
    .section-hidden {
        opacity: 0;
        transform: translateY(30px);
    }
    
    .section-visible {
        opacity: 1;
        transform: translateY(0);
        transition: opacity 0.8s ease, transform 0.8s ease;
    }
    
    /* Navbar enhancements */
    .navbar-scrolled {
        backdrop-filter: blur(20px) !important;
        background: rgba(10, 10, 10, 0.98) !important;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }
    
    .navbar-hidden {
        transform: translateY(-100%);
        transition: transform 0.3s ease;
    }
    
    /* Image loading */
    img {
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    
    img.loaded {
        opacity: 1;
    }
    
    /* Button ripple effect */
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: scale(0);
        animation: ripple-animation 0.6s ease-out;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    /* Magnetic nav links */
    .nav-link {
        transition: transform 0.2s ease;
    }
    
    /* Enhanced project card hover */
    .project-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .project-card:hover {
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Typed.js cursor */
    .typed-cursor {
        opacity: 1;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Particles container */
    #particles-js {
        opacity: 0.5;
    }
`;
document.head.appendChild(style);