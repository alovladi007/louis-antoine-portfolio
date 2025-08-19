// Portfolio Enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Enhance section titles with vibrant colors
    const titles = document.querySelectorAll('.section-title');
    titles.forEach(title => {
        title.style.background = 'linear-gradient(135deg, #00d4ff 0%, #ff00ff 50%, #00ff88 100%)';
        title.style.backgroundSize = '200% 200%';
        title.style.webkitBackgroundClip = 'text';
        title.style.webkitTextFillColor = 'transparent';
        title.style.filter = 'brightness(1.5)';
        title.style.animation = 'gradient-shift 3s ease infinite';
    });
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .contact-item a {
            color: #00d4ff !important;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .contact-item a:hover {
            color: #ff00ff !important;
            text-shadow: 0 0 15px currentColor;
        }
    `;
    document.head.appendChild(style);
    
    // Fix contact links
    const phoneItems = document.querySelectorAll('.contact-item');
    phoneItems.forEach(item => {
        if (item.textContent.includes('(203) 360-5619')) {
            const p = item.querySelector('p');
            if (p) p.innerHTML = '<a href="tel:+12033605619">(203) 360-5619</a>';
        }
        if (item.textContent.includes('alovladi@gmail.com')) {
            const p = item.querySelector('p');
            if (p) p.innerHTML = '<a href="mailto:alovladi@gmail.com">alovladi@gmail.com</a>';
        }
    });
    
    console.log('Enhancements loaded!');
});