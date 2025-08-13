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
    if (typeof event !== 'undefined' && event) {
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
        
        // Initialize carousel for the active section
        updateCarousel(section);
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
    
    if (!container) {
        console.log('Carousel container not found for section:', section);
        return;
    }
    
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

// Create skills particles
function createSkillsParticles() {
    const particlesContainer = document.querySelector('.skills-particles');
    if (!particlesContainer) return;
    
    for (let i = 0; i < 10; i++) {
        const particle = document.createElement('div');
        particle.className = 'skill-particle';
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 2}px;
            height: ${Math.random() * 4 + 2}px;
            background: rgba(102, 126, 234, ${Math.random() * 0.6 + 0.2});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            animation: skill-particle ${Math.random() * 10 + 10}s infinite linear;
            animation-delay: ${Math.random() * 10}s;
        `;
        particlesContainer.appendChild(particle);
    }
}



// Create futuristic city effects
function createFuturisticCityEffects() {
    // Add city lights animation
    const addCityLights = () => {
        const buildings = document.querySelectorAll('.skyscraper, .tower');
        buildings.forEach(building => {
            for (let i = 0; i < 5; i++) {
                const light = document.createElement('div');
                light.style.cssText = `
                    position: absolute;
                    width: 4px;
                    height: 4px;
                    background: #ffff00;
                    box-shadow: 0 0 5px #ffff00;
                    left: ${10 + Math.random() * 80}%;
                    top: ${10 + Math.random() * 80}%;
                    animation: light-flicker ${2 + Math.random() * 3}s ease-in-out infinite;
                    animation-delay: ${Math.random() * 2}s;
                `;
                building.appendChild(light);
            }
        });
    };
    
    // Create moving traffic lights
    const createTrafficLights = () => {
        const lanes = document.querySelectorAll('.vehicle-lane');
        lanes.forEach((lane, index) => {
            setInterval(() => {
                const light = document.createElement('div');
                light.style.cssText = `
                    position: absolute;
                    width: 20px;
                    height: 2px;
                    background: linear-gradient(to right, transparent, #ff6600, transparent);
                    left: ${index % 2 === 0 ? '-20px' : '100%'};
                    animation: traffic-light-move 10s linear forwards;
                `;
                lane.appendChild(light);
                setTimeout(() => light.remove(), 10000);
            }, 3000);
        });
    };
    
    // Add stars to sky
    const skyGradient = document.querySelector('.sky-gradient');
    if (skyGradient) {
        for (let i = 0; i < 200; i++) {
            const star = document.createElement('div');
            star.style.cssText = `
                position: absolute;
                width: ${Math.random() * 2}px;
                height: ${Math.random() * 2}px;
                background: white;
                border-radius: 50%;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 60}%;
                opacity: ${Math.random() * 0.8 + 0.2};
                animation: star-twinkle ${Math.random() * 3 + 2}s ease-in-out infinite;
                animation-delay: ${Math.random() * 3}s;
            `;
            skyGradient.appendChild(star);
        }
    }
    
    // Create energy particles
    const createEnergyParticles = () => {
        const particlesContainer = document.querySelector('.energy-particles');
        if (particlesContainer) {
            setInterval(() => {
                const particle = document.createElement('div');
                particle.className = 'energy-particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 5 + 's';
                particlesContainer.appendChild(particle);
                
                setTimeout(() => particle.remove(), 8000);
            }, 300);
        }
    };
    
    // Add window lights to buildings
    const addWindowLights = () => {
        const buildings = document.querySelectorAll('.mega-city, .main-metropolis');
        buildings.forEach(building => {
            const grid = document.createElement('div');
            grid.className = 'city-lights-grid';
            building.appendChild(grid);
        });
    };
    
    // Create shooting stars occasionally
    const createShootingStar = () => {
        const sky = document.querySelector('.planet-sky');
        if (sky) {
            setInterval(() => {
                const shootingStar = document.createElement('div');
                shootingStar.style.cssText = `
                    position: absolute;
                    width: 100px;
                    height: 2px;
                    background: linear-gradient(to right, transparent, white, transparent);
                    top: ${Math.random() * 40}%;
                    left: -100px;
                    animation: shooting-star 2s linear forwards;
                `;
                sky.appendChild(shootingStar);
                setTimeout(() => shootingStar.remove(), 2000);
            }, 8000);
        }
    };
    
    setTimeout(addCityLights, 100);
    setTimeout(createTrafficLights, 200);
    setTimeout(createEnergyParticles, 300);
    setTimeout(addWindowLights, 400);
    setTimeout(createShootingStar, 500);
}

// Add CSS for city animations
const cityStyles = document.createElement('style');
cityStyles.textContent = `
    @keyframes light-flicker {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    @keyframes traffic-light-move {
        to { transform: translateX(calc(100vw + 40px)); }
    }
    
    @keyframes star-twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    @keyframes shooting-star {
        0% {
            transform: translateX(0) translateY(0);
            opacity: 1;
        }
        100% {
            transform: translateX(300px) translateY(100px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(cityStyles);

// Initialize About Section
document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles
    createParticles();
    createSkillsParticles();
    createFuturisticCityEffects();
    
    // Start carousel auto-advance
    startCarouselAutoAdvance();
    
    // Initialize the first section properly
    const introSection = document.getElementById('intro-section');
    if (introSection) {
        introSection.style.display = 'block';
        introSection.classList.add('active');
    }
    
    // Initialize About Me section tabs
    setTimeout(() => {
        console.log('Initializing About Me section...');
        
        // Make sure the intro section is visible
        const introSection = document.getElementById('intro-section');
        if (introSection) {
            introSection.style.display = 'block';
            introSection.classList.add('active');
        }
        
        // Initialize first slide in each carousel
        const carouselSections = ['intro', 'work', 'mission'];
        carouselSections.forEach(section => {
            const firstSlide = document.querySelector(`#${section}-section .carousel-slide`);
            if (firstSlide) {
                firstSlide.classList.add('active');
            }
            const firstIndicator = document.querySelector(`#${section}-section .indicator`);
            if (firstIndicator) {
                firstIndicator.classList.add('active');
            }
        });
        
        // Now show the section
        showAboutSection3D('intro');
    }, 500);
    
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

// Simple audio control
window.addEventListener('DOMContentLoaded', function() {
    const audio = document.getElementById('background-audio');
    if (audio) {
        console.log('Audio element found and ready');
        // Audio will be controlled by the play button
    }
    
    // Initialize Tesseract effect with delay to ensure DOM is ready
    setTimeout(() => {
        initTesseractEffect();
    }, 100);
});

// Tesseract Effect Implementation
function initTesseractEffect() {
    console.log('Initializing Tesseract effect...');
    
    const grid = document.querySelector('.tesseract-grid');
    const timeStreams = document.querySelector('.time-streams');
    const canvas = document.getElementById('tesseract-canvas');
    
    if (!grid || !canvas) {
        console.error('Tesseract elements not found:', { grid, canvas });
        return;
    }
    
    console.log('Tesseract elements found, creating effect...');
    
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Create grid lines
    const gridSize = 12;
    const spacing = 100;
    
    // Horizontal lines
    for (let i = 0; i < gridSize; i++) {
        const line = document.createElement('div');
        line.className = 'grid-line horizontal';
        line.style.top = `${(i / gridSize) * 100}%`;
        line.style.animationDelay = `${i * 0.1}s`;
        grid.appendChild(line);
    }
    
    // Vertical lines
    for (let i = 0; i < gridSize; i++) {
        const line = document.createElement('div');
        line.className = 'grid-line vertical';
        line.style.left = `${(i / gridSize) * 100}%`;
        line.style.animationDelay = `${i * 0.15}s`;
        grid.appendChild(line);
    }
    
    // Depth points
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const point = document.createElement('div');
            point.className = 'grid-line depth';
            point.style.left = `${(i / gridSize) * 100}%`;
            point.style.top = `${(j / gridSize) * 100}%`;
            point.style.animationDelay = `${(i + j) * 0.1}s`;
            grid.appendChild(point);
        }
    }
    
    // Create time stream particles
    function createTimeParticle() {
        const particle = document.createElement('div');
        particle.className = 'time-particle';
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDuration = `${3 + Math.random() * 4}s`;
        particle.style.animationDelay = `${Math.random() * 2}s`;
        timeStreams.appendChild(particle);
        
        // Remove particle after animation
        setTimeout(() => particle.remove(), 7000);
    }
    
    // Continuously create particles
    setInterval(createTimeParticle, 200);
    
    // Canvas animation for additional effects
    let time = 0;
    function animateCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw connecting lines with time distortion
        ctx.strokeStyle = 'rgba(102, 224, 255, 0.1)';
        ctx.lineWidth = 1;
        
        for (let i = 0; i < 5; i++) {
            ctx.beginPath();
            const offset = time * 0.001 + i * 0.5;
            const x1 = Math.sin(offset) * canvas.width * 0.3 + canvas.width * 0.5;
            const y1 = Math.cos(offset * 0.7) * canvas.height * 0.3 + canvas.height * 0.5;
            const x2 = Math.sin(offset + Math.PI) * canvas.width * 0.3 + canvas.width * 0.5;
            const y2 = Math.cos((offset + Math.PI) * 0.7) * canvas.height * 0.3 + canvas.height * 0.5;
            
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        
        time++;
        requestAnimationFrame(animateCanvas);
    }
    
    animateCanvas();
    
    // Interactive ripple effect on click
    document.querySelector('.experience').addEventListener('click', function(e) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple';
        ripple.style.left = e.clientX + 'px';
        ripple.style.top = e.clientY - this.offsetTop + 'px';
        this.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 2000);
    });
    
    // Mouse parallax effect
    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', function(e) {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
        
        grid.style.transform = `
            perspective(1200px) 
            rotateX(${mouseY * 10}deg) 
            rotateY(${mouseX * 10}deg)
            translateZ(${Math.abs(mouseX * mouseY) * 50}px)
        `;
    });
    
    // Resize handler
    window.addEventListener('resize', function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

// Add event listeners for About Me tabs after everything loads
window.addEventListener('load', function() {
    // Re-attach click handlers to tabs
    document.querySelectorAll('.about-tab-3d').forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const section = this.getAttribute('data-section');
            if (section) {
                console.log('Tab clicked:', section);
                window.showAboutSection3D(section);
            }
        });
    });
    
    // Re-attach carousel navigation
    document.querySelectorAll('.carousel-nav .nav-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });
});
