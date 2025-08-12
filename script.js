// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Animate skill bars when they come into view
const observerOptions = {
    threshold: 0.5,
    rootMargin: '0px 0px -100px 0px'
};

const skillObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const skillBars = entry.target.querySelectorAll('.skill-progress');
            skillBars.forEach(bar => {
                const width = bar.getAttribute('data-width');
                bar.style.width = width;
            });
        }
    });
}, observerOptions);

// Observe skills section
const skillsSection = document.querySelector('.skills');
if (skillsSection) {
    skillObserver.observe(skillsSection);
}

// Add scroll effect to navbar
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(10, 10, 10, 0.98)';
        navbar.style.backdropFilter = 'blur(15px)';
    } else {
        navbar.style.background = 'rgba(10, 10, 10, 0.95)';
        navbar.style.backdropFilter = 'blur(10px)';
    }
});

// Add animation to cards when they come into view
const cardObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
});

// Observe all cards
document.querySelectorAll('.project-card, .certification-card, .contact-item').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    cardObserver.observe(card);
});

// Add typing effect to hero title
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.innerHTML = '';
    
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Initialize typing effect when page loads
window.addEventListener('load', () => {
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        setTimeout(() => {
            typeWriter(heroTitle, originalText, 150);
        }, 500);
    }
});

// Add parallax effect to hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    if (hero) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
    }
});

// Add hover effects to timeline items
document.querySelectorAll('.timeline-item').forEach(item => {
    item.addEventListener('mouseenter', () => {
        item.style.transform = 'scale(1.02)';
        item.style.transition = 'transform 0.3s ease';
    });
    
    item.addEventListener('mouseleave', () => {
        item.style.transform = 'scale(1)';
    });
});

// Add click effect to certificate placeholders
document.querySelectorAll('.cert-placeholder').forEach(placeholder => {
    placeholder.addEventListener('click', () => {
        placeholder.style.transform = 'scale(0.95)';
        setTimeout(() => {
            placeholder.style.transform = 'scale(1)';
        }, 150);
    });
});

// Add loading animation
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// About section tab switching - Removed in favor of showAboutSection3D

// Slideshow functionality
let slideIndex = 1;

window.currentSlide = function(n) {
    showSlide(slideIndex = n);
}

function showSlide(n) {
    let slides = document.getElementsByClassName("slide");
    let dots = document.getElementsByClassName("dot");
    
    if (n > slides.length) { slideIndex = 1 }
    if (n < 1) { slideIndex = slides.length }
    
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove('active');
    }
    
    for (let i = 0; i < dots.length; i++) {
        dots[i].classList.remove('active');
    }
    
    slides[slideIndex-1].classList.add('active');
    dots[slideIndex-1].classList.add('active');
}

// Work slideshow functionality
let workSlideIndex = 1;

window.currentWorkSlide = function(n) {
    showWorkSlide(workSlideIndex = n);
}

function showWorkSlide(n) {
    let slides = document.getElementsByClassName("work-slide");
    let dots = document.getElementsByClassName("work-dot");
    
    if (n > slides.length) { workSlideIndex = 1 }
    if (n < 1) { workSlideIndex = slides.length }
    
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove('active');
    }
    
    for (let i = 0; i < dots.length; i++) {
        dots[i].classList.remove('active');
    }
    
    if (slides[workSlideIndex-1]) {
        slides[workSlideIndex-1].classList.add('active');
    }
    if (dots[workSlideIndex-1]) {
        dots[workSlideIndex-1].classList.add('active');
    }
}

// Mission slideshow functionality
let missionSlideIndex = 1;

window.currentMissionSlide = function(n) {
    showMissionSlide(missionSlideIndex = n);
}

function showMissionSlide(n) {
    let slides = document.getElementsByClassName("mission-slide");
    let dots = document.getElementsByClassName("mission-dot");
    
    if (n > slides.length) { missionSlideIndex = 1 }
    if (n < 1) { missionSlideIndex = slides.length }
    
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove('active');
    }
    
    for (let i = 0; i < dots.length; i++) {
        dots[i].classList.remove('active');
    }
    
    if (slides[missionSlideIndex-1]) {
        slides[missionSlideIndex-1].classList.add('active');
    }
    if (dots[missionSlideIndex-1]) {
        dots[missionSlideIndex-1].classList.add('active');
    }
}

// Initialize slideshows when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Auto-advance About Me slideshow
    setInterval(() => {
        slideIndex++;
        showSlide(slideIndex);
    }, 5000); // Change slide every 5 seconds
    
    // Auto-advance Work slideshow
    setInterval(() => {
        workSlideIndex++;
        showWorkSlide(workSlideIndex);
    }, 4000); // Change slide every 4 seconds
    
    // Auto-advance Mission slideshow
    setInterval(() => {
        missionSlideIndex++;
        showMissionSlide(missionSlideIndex);
    }, 4500); // Change slide every 4.5 seconds
});

// Audio control functionality
let isPlaying = false;
const audio = document.getElementById('background-audio');
const audioBtn = document.getElementById('audio-toggle');
const audioVisualizer = document.querySelector('.audio-visualizer');

function toggleAudio() {
    if (!audio) return;
    
    if (isPlaying) {
        audio.pause();
        audioBtn.classList.remove('playing');
        audioBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
        if (audioVisualizer) {
            audioVisualizer.style.display = 'none';
        }
        showAudioNotification('Music Paused ⏸️');
    } else {
        audio.volume = 0.5; // Set volume to 50%
        audio.play().then(() => {
            audioBtn.classList.add('playing');
            audioBtn.innerHTML = '<i class="fas fa-volume-mute"></i>';
            if (audioVisualizer) {
                audioVisualizer.style.display = 'flex';
            }
            showAudioNotification('Music Playing 🎵');
        }).catch(e => {
            console.log('Audio play failed:', e);
            showAudioNotification('Click again to play music');
        });
    }
    isPlaying = !isPlaying;
}

// Show audio notification
function showAudioNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'audio-notification';
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 2000);
}

// Initialize audio on user interaction
document.addEventListener('click', function initAudio() {
    if (audio && !isPlaying) {
        audio.volume = 0.3; // Set volume to 30%
        // Remove this listener after first interaction
        document.removeEventListener('click', initAudio);
    }
}, { once: true });

// New About Section 3D Functions
window.showAboutSection3D = function(section) {
    console.log('Switching to section:', section);
    
    // Prevent event bubbling
    if (event) {
        event.stopPropagation();
    }
    
    // Update tab states with a small delay for animation
    document.querySelectorAll('.about-tab-3d').forEach(tab => {
        tab.classList.remove('active');
        // Reset transform for non-active tabs
        if (tab.dataset.section !== section) {
            tab.style.transform = 'rotateY(0deg)';
        }
    });
    
    // Set active tab
    const activeTab = document.querySelector(`[data-section="${section}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
        activeTab.style.transform = 'rotateY(180deg)';
    }
    
    // Update section visibility with delay for smooth transition
    setTimeout(() => {
        document.querySelectorAll('.about-section').forEach(sec => {
            sec.classList.remove('active');
            sec.style.display = 'none';
        });
        
        const activeSection = document.getElementById(`${section}-section`);
        if (activeSection) {
            activeSection.style.display = 'block';
            setTimeout(() => {
                activeSection.classList.add('active');
            }, 50);
        }
        
        // Ensure the content wrapper has proper height
        const wrapper = document.querySelector('.about-content-wrapper');
        if (wrapper && activeSection) {
            setTimeout(() => {
                wrapper.style.minHeight = activeSection.offsetHeight + 'px';
            }, 100);
        }
    }, 300);
}

// Carousel Navigation
const carousels = {
    intro: { current: 1, total: 2 },
    work: { current: 1, total: 3 },
    mission: { current: 1, total: 2 }
};

window.navigateSlide = function(section, direction) {
    const carousel = carousels[section];
    carousel.current += direction;
    
    if (carousel.current > carousel.total) {
        carousel.current = 1;
    } else if (carousel.current < 1) {
        carousel.current = carousel.total;
    }
    
    updateCarousel(section);
}

window.goToSlide = function(section, slideNum) {
    carousels[section].current = slideNum;
    updateCarousel(section);
}

function updateCarousel(section) {
    const carousel = carousels[section];
    const container = document.querySelector(`#${section}-section .carousel-track`);
    
    // Update slide visibility
    container.querySelectorAll('.carousel-slide').forEach((slide, index) => {
        if (index + 1 === carousel.current) {
            slide.classList.add('active');
        } else {
            slide.classList.remove('active');
        }
    });
    
    // Update indicators
    const indicators = document.querySelectorAll(`#${section}-section .indicator`);
    indicators.forEach((indicator, index) => {
        if (index + 1 === carousel.current) {
            indicator.classList.add('active');
        } else {
            indicator.classList.remove('active');
        }
    });
}

// Auto-advance carousels
function startCarouselAutoAdvance() {
    setInterval(() => {
        // Auto-advance active section's carousel
        const activeSection = document.querySelector('.about-section.active');
        if (activeSection) {
            const sectionId = activeSection.id.replace('-section', '');
            navigateSlide(sectionId, 1);
        }
    }, 5000);
}

// Create dynamic particles
function createParticles() {
    const particleSystem = document.querySelector('.particle-system');
    if (!particleSystem) return;
    
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 1}px;
            height: ${Math.random() * 4 + 1}px;
            background: rgba(102, 126, 234, ${Math.random() * 0.5 + 0.3});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            animation: particle-float ${Math.random() * 20 + 10}s infinite linear;
            animation-delay: ${Math.random() * 20}s;
        `;
        particleSystem.appendChild(particle);
    }
}

// Create Minority Report style effects
function createMinorityReportEffects() {
    const holoInterface = document.querySelector('.holographic-interface');
    if (!holoInterface) return;
    
    // Create gesture trails
    const gestureContainer = holoInterface.querySelector('.gesture-container');
    setInterval(() => {
        const trail = document.createElement('div');
        trail.className = 'gesture-trail';
        trail.style.top = Math.random() * 100 + '%';
        trail.style.left = '-100px';
        gestureContainer.appendChild(trail);
        
        setTimeout(() => trail.remove(), 3000);
    }, 4000);
    
    // Create random touch effects
    setInterval(() => {
        const touchEffect = document.createElement('div');
        touchEffect.className = 'touch-effect';
        touchEffect.style.left = Math.random() * 100 + '%';
        touchEffect.style.top = Math.random() * 100 + '%';
        holoInterface.querySelector('.touch-effects').appendChild(touchEffect);
        
        setTimeout(() => touchEffect.remove(), 1000);
    }, 3000);
    
    // Animate data bars
    const dataBars = holoInterface.querySelector('.data-bars');
    if (dataBars) {
        for (let i = 0; i < 5; i++) {
            const bar = document.createElement('span');
            bar.style.height = Math.random() * 50 + 20 + 'px';
            bar.style.animationDelay = i * 0.2 + 's';
            dataBars.appendChild(bar);
        }
    }
    
    // Add hover interaction to screens
    const screens = holoInterface.querySelectorAll('.holo-screen');
    screens.forEach(screen => {
        screen.addEventListener('mouseenter', () => {
            screen.style.transform = `scale(1.05) ${screen.style.transform}`;
            screen.style.opacity = '0.9';
        });
        
        screen.addEventListener('mouseleave', () => {
            screen.style.transform = screen.style.transform.replace('scale(1.05) ', '');
            screen.style.opacity = '0.7';
        });
    });
}

// Initialize About Section
document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles
    createParticles();
    createMinorityReportEffects();
    
    // Start carousel auto-advance
    startCarouselAutoAdvance();
    
    // Initialize the first section properly
    const introSection = document.getElementById('intro-section');
    if (introSection) {
        introSection.style.display = 'block';
        introSection.classList.add('active');
    }
    
    // Add glitch effect on hover
    document.querySelectorAll('.glitch-text').forEach(text => {
        text.addEventListener('mouseenter', () => {
            text.style.animation = 'none';
            setTimeout(() => {
                text.style.animation = '';
            }, 100);
        });
    });
    
    // Timeline scroll animation
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const timelineObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);
    
    const timeline = document.querySelector('.about-timeline');
    if (timeline) {
        timelineObserver.observe(timeline);
    }
});

