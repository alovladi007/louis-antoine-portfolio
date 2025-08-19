// Portfolio Enhancements Script
document.addEventListener('DOMContentLoaded', function() {
    
    // A. ENHANCE SECTION TITLE COLORS
    const sectionTitles = document.querySelectorAll('.section-title');
    sectionTitles.forEach(title => {
        title.style.background = 'linear-gradient(135deg, #00d4ff 0%, #ff00ff 50%, #00ff88 100%)';
        title.style.backgroundSize = '200% 200%';
        title.style.webkitBackgroundClip = 'text';
        title.style.webkitTextFillColor = 'transparent';
        title.style.backgroundClip = 'text';
        title.style.filter = 'brightness(1.5)';
        title.style.fontWeight = 'bold';
    });
    
    // B. FIX CONTACT SECTION LINKS
    const phoneElements = document.querySelectorAll('.contact-item');
    phoneElements.forEach(item => {
        if (item.textContent.includes('(203) 360-5619')) {
            const phoneP = item.querySelector('p');
            if (phoneP) {
                phoneP.innerHTML = '<a href="tel:+12033605619">(203) 360-5619</a>';
            }
        }
    });
    
    console.log('Portfolio enhancements loaded!');
});
