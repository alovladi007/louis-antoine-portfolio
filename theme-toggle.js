// Theme Toggle System with Local Storage Persistence
(function() {
    'use strict';

    // Theme configuration
    const themes = {
        dark: {
            name: 'dark',
            icon: 'fa-moon',
            colors: {
                '--bg-primary': '#0f0f1e',
                '--bg-secondary': '#1a1a2e',
                '--bg-card': 'rgba(255, 255, 255, 0.05)',
                '--text-primary': '#ffffff',
                '--text-secondary': '#e5e7eb',
                '--border': 'rgba(255, 255, 255, 0.1)',
                '--accent': '#667eea',
                '--accent-secondary': '#764ba2',
                '--success': '#4ade80',
                '--warning': '#fbbf24',
                '--danger': '#ef4444'
            }
        },
        light: {
            name: 'light',
            icon: 'fa-sun',
            colors: {
                '--bg-primary': '#ffffff',
                '--bg-secondary': '#f3f4f6',
                '--bg-card': 'rgba(0, 0, 0, 0.05)',
                '--text-primary': '#1f2937',
                '--text-secondary': '#4b5563',
                '--border': 'rgba(0, 0, 0, 0.1)',
                '--accent': '#667eea',
                '--accent-secondary': '#764ba2',
                '--success': '#10b981',
                '--warning': '#f59e0b',
                '--danger': '#ef4444'
            }
        }
    };

    class ThemeManager {
        constructor() {
            this.currentTheme = this.loadTheme() || 'dark';
            this.init();
        }

        init() {
            // Apply saved theme immediately
            this.applyTheme(this.currentTheme);
            
            // Create and inject theme toggle button
            this.createToggleButton();
            
            // Listen for theme changes in other tabs
            window.addEventListener('storage', (e) => {
                if (e.key === 'portfolio-theme') {
                    this.currentTheme = e.newValue || 'dark';
                    this.applyTheme(this.currentTheme);
                    this.updateToggleButton();
                }
            });

            // Add transition class after initial load
            setTimeout(() => {
                document.body.classList.add('theme-transition');
            }, 100);
        }

        createToggleButton() {
            // Check if button already exists
            if (document.getElementById('theme-toggle-btn')) return;

            const button = document.createElement('button');
            button.id = 'theme-toggle-btn';
            button.className = 'theme-toggle-button';
            button.setAttribute('aria-label', 'Toggle theme');
            button.innerHTML = `
                <i class="fas ${this.currentTheme === 'dark' ? 'fa-sun' : 'fa-moon'}"></i>
            `;

            // Add styles
            const styles = document.createElement('style');
            styles.textContent = `
                .theme-toggle-button {
                    position: fixed;
                    bottom: 2rem;
                    right: 2rem;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    background: var(--accent, #667eea);
                    color: white;
                    border: none;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.2rem;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    transition: all 0.3s ease;
                    z-index: 9999;
                }

                .theme-toggle-button:hover {
                    transform: scale(1.1) rotate(20deg);
                    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                }

                .theme-toggle-button:active {
                    transform: scale(0.95);
                }

                .theme-transition,
                .theme-transition *,
                .theme-transition *::before,
                .theme-transition *::after {
                    transition: background 0.3s ease, 
                                background-color 0.3s ease,
                                border-color 0.3s ease,
                                color 0.3s ease !important;
                }

                body.light-theme {
                    background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%) !important;
                }

                body.light-theme .navbar {
                    background: rgba(255, 255, 255, 0.95) !important;
                    backdrop-filter: blur(10px);
                    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                }

                body.light-theme .project-card,
                body.light-theme .feature-card,
                body.light-theme .skill-card {
                    background: white !important;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }

                body.light-theme .nav-link {
                    color: #4b5563 !important;
                }

                body.light-theme .nav-link:hover {
                    color: #667eea !important;
                }

                body.light-theme h1,
                body.light-theme h2,
                body.light-theme h3 {
                    color: #1f2937 !important;
                }

                body.light-theme p,
                body.light-theme .description {
                    color: #4b5563 !important;
                }

                body.light-theme .tech-tag {
                    background: rgba(102, 126, 234, 0.1) !important;
                    color: #667eea !important;
                }

                body.light-theme code,
                body.light-theme pre {
                    background: #f3f4f6 !important;
                    color: #1f2937 !important;
                }

                /* Accessibility improvements */
                @media (prefers-reduced-motion: reduce) {
                    .theme-transition,
                    .theme-transition *,
                    .theme-transition *::before,
                    .theme-transition *::after {
                        transition: none !important;
                    }
                }

                /* High contrast mode support */
                @media (prefers-contrast: high) {
                    .theme-toggle-button {
                        border: 2px solid currentColor;
                    }
                    
                    body.light-theme {
                        --text-primary: #000000;
                        --text-secondary: #333333;
                        --bg-primary: #ffffff;
                        --bg-secondary: #f0f0f0;
                    }
                }

                /* Tooltip for theme toggle */
                .theme-toggle-button::after {
                    content: attr(data-tooltip);
                    position: absolute;
                    bottom: 100%;
                    right: 0;
                    margin-bottom: 0.5rem;
                    padding: 0.5rem 1rem;
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    border-radius: 8px;
                    font-size: 0.875rem;
                    white-space: nowrap;
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.3s ease;
                }

                .theme-toggle-button:hover::after {
                    opacity: 1;
                }
            `;

            document.head.appendChild(styles);
            document.body.appendChild(button);

            // Add click handler
            button.addEventListener('click', () => this.toggleTheme());

            // Update tooltip
            this.updateToggleButton();
        }

        toggleTheme() {
            this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
            this.applyTheme(this.currentTheme);
            this.saveTheme(this.currentTheme);
            this.updateToggleButton();
            
            // Announce theme change for screen readers
            this.announceThemeChange();
        }

        applyTheme(themeName) {
            const theme = themes[themeName];
            if (!theme) return;

            // Apply CSS variables
            const root = document.documentElement;
            Object.entries(theme.colors).forEach(([key, value]) => {
                root.style.setProperty(key, value);
            });

            // Update body class
            document.body.classList.remove('dark-theme', 'light-theme');
            document.body.classList.add(`${themeName}-theme`);

            // Update meta theme-color for mobile browsers
            const metaThemeColor = document.querySelector('meta[name="theme-color"]');
            if (metaThemeColor) {
                metaThemeColor.content = theme.colors['--bg-primary'];
            } else {
                const meta = document.createElement('meta');
                meta.name = 'theme-color';
                meta.content = theme.colors['--bg-primary'];
                document.head.appendChild(meta);
            }
        }

        updateToggleButton() {
            const button = document.getElementById('theme-toggle-btn');
            if (!button) return;

            const icon = button.querySelector('i');
            const nextTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
            
            icon.className = `fas ${themes[nextTheme].icon}`;
            button.setAttribute('data-tooltip', `Switch to ${nextTheme} mode`);
        }

        saveTheme(themeName) {
            try {
                localStorage.setItem('portfolio-theme', themeName);
                
                // Also save preference timestamp
                localStorage.setItem('portfolio-theme-timestamp', Date.now().toString());
            } catch (e) {
                console.warn('Could not save theme preference:', e);
            }
        }

        loadTheme() {
            try {
                const savedTheme = localStorage.getItem('portfolio-theme');
                
                // Check if user has a system preference and no saved theme
                if (!savedTheme) {
                    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    return prefersDark ? 'dark' : 'light';
                }
                
                return savedTheme;
            } catch (e) {
                console.warn('Could not load theme preference:', e);
                return 'dark';
            }
        }

        announceThemeChange() {
            const announcement = document.createElement('div');
            announcement.setAttribute('role', 'status');
            announcement.setAttribute('aria-live', 'polite');
            announcement.className = 'sr-only';
            announcement.textContent = `Theme changed to ${this.currentTheme} mode`;
            
            document.body.appendChild(announcement);
            setTimeout(() => announcement.remove(), 1000);
        }

        // Public API
        getTheme() {
            return this.currentTheme;
        }

        setTheme(themeName) {
            if (themes[themeName]) {
                this.currentTheme = themeName;
                this.applyTheme(themeName);
                this.saveTheme(themeName);
                this.updateToggleButton();
            }
        }
    }

    // Initialize theme manager when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.themeManager = new ThemeManager();
        });
    } else {
        window.themeManager = new ThemeManager();
    }

    // Export for use in other scripts
    window.ThemeManager = ThemeManager;
})();