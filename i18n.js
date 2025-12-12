// Internationalization (i18n) System
(function() {
    'use strict';

    const translations = {
        en: {
            nav: {
                home: 'Home',
                about: 'About',
                skills: 'Skills',
                projects: 'Projects',
                experience: 'Experience',
                certifications: 'Certifications',
                contact: 'Contact'
            },
            hero: {
                greeting: "Hi, I'm",
                name: 'Louis Antoine',
                subtitle: 'Electrical Engineer & Applied Scientist',
                desc1: 'Hands-on expertise in semiconductor manufacturing, optical metrology, and advanced process development. Strong foundation in materials science, device physics, and cleanroom operations.',
                desc2: 'Passionate about applying machine learning, data analytics, and statistical methods to solve complex engineering challenges. Experienced in process optimization, yield improvement, and cross-functional collaboration in fast-paced technical environments.',
                cta: 'View Projects',
                contact: 'Get in Touch'
            },
            about: {
                title: 'About Me',
                description: 'Engineer and Applied Scientist with expertise in advanced manufacturing, data-driven optimization, machine learning, and process development across semiconductor, photonics, and emerging technologies.'
            },
            skills: {
                title: 'Technical Skills',
                learnMore: 'Learn More About Skills'
            },
            projects: {
                title: 'Featured Projects',
                viewProject: 'View Project',
                viewDetails: 'View Details',
                learnMore: 'Learn More About Projects'
            },
            experience: {
                title: 'Professional Experience',
                learnMore: 'Learn More About Experience'
            },
            certifications: {
                title: 'Certifications',
                viewCertificate: 'View Certificate',
                learnMore: 'Learn More About Certifications'
            },
            contact: {
                title: 'Get In Touch',
                email: 'Email',
                phone: 'Phone',
                location: 'Location',
                schedule: 'Schedule a Meeting',
                scheduleDesc: 'Book a time directly on my calendar',
                bookMeeting: 'Book a Meeting',
                sendMessage: 'Send me a message',
                name: 'Name',
                company: 'Company/Organization',
                subject: 'Subject',
                message: 'Message',
                send: 'Send Message'
            },
            newsletter: {
                title: 'Stay Updated',
                description: 'Get notified about new projects, technical insights, and industry updates.',
                placeholder: 'Enter your email',
                subscribe: 'Subscribe',
                privacy: 'No spam, unsubscribe anytime.'
            },
            footer: {
                aboutTitle: 'About',
                aboutDesc: 'Engineer and Applied Scientist with expertise in manufacturing, data analytics, and process optimization across multiple technical domains.',
                quickLinks: 'Quick Links',
                features: 'Interactive Features',
                resources: 'Resources & Tools',
                allProjects: 'All Projects',
                githubProfile: 'GitHub Profile',
                contactMe: 'Contact Me',
                linkedinProfile: 'LinkedIn Profile',
                downloadResume: 'Download Resume',
                emailMe: 'Email Me',
                rights: 'All rights reserved.',
                privacy: 'Privacy Policy',
                terms: 'Terms of Service',
                sitemap: 'Sitemap'
            }
        },
        fr: {
            nav: {
                home: 'Accueil',
                about: 'À Propos',
                skills: 'Compétences',
                projects: 'Projets',
                experience: 'Expérience',
                certifications: 'Certifications',
                contact: 'Contact'
            },
            hero: {
                greeting: 'Bonjour, je suis',
                name: 'Louis Antoine',
                subtitle: 'Ingénieur Électrique & Scientifique Appliqué',
                desc1: 'Expertise pratique en fabrication de semiconducteurs, métrologie optique et développement de processus avancés. Solide formation en science des matériaux, physique des dispositifs et opérations en salle blanche.',
                desc2: 'Passionné par l\'application de l\'apprentissage automatique, l\'analyse de données et les méthodes statistiques pour résoudre des défis d\'ingénierie complexes. Expérimenté en optimisation de processus, amélioration du rendement et collaboration interfonctionnelle.',
                cta: 'Voir les Projets',
                contact: 'Me Contacter'
            },
            about: {
                title: 'À Propos de Moi',
                description: 'Ingénieur et scientifique appliqué avec expertise en fabrication avancée, optimisation basée sur les données, apprentissage automatique et développement de processus dans les semiconducteurs, la photonique et les technologies émergentes.'
            },
            skills: {
                title: 'Compétences Techniques',
                learnMore: 'En Savoir Plus sur les Compétences'
            },
            projects: {
                title: 'Projets en Vedette',
                viewProject: 'Voir le Projet',
                viewDetails: 'Voir les Détails',
                learnMore: 'En Savoir Plus sur les Projets'
            },
            experience: {
                title: 'Expérience Professionnelle',
                learnMore: 'En Savoir Plus sur l\'Expérience'
            },
            certifications: {
                title: 'Certifications',
                viewCertificate: 'Voir le Certificat',
                learnMore: 'En Savoir Plus sur les Certifications'
            },
            contact: {
                title: 'Me Contacter',
                email: 'Email',
                phone: 'Téléphone',
                location: 'Localisation',
                schedule: 'Planifier une Réunion',
                scheduleDesc: 'Réservez un créneau directement sur mon calendrier',
                bookMeeting: 'Réserver une Réunion',
                sendMessage: 'Envoyez-moi un message',
                name: 'Nom',
                company: 'Entreprise/Organisation',
                subject: 'Sujet',
                message: 'Message',
                send: 'Envoyer le Message'
            },
            newsletter: {
                title: 'Restez Informé',
                description: 'Recevez des notifications sur les nouveaux projets, les insights techniques et les mises à jour de l\'industrie.',
                placeholder: 'Entrez votre email',
                subscribe: 'S\'abonner',
                privacy: 'Pas de spam, désabonnez-vous à tout moment.'
            },
            footer: {
                aboutTitle: 'À Propos',
                aboutDesc: 'Ingénieur et scientifique appliqué avec expertise en fabrication, analyse de données et optimisation de processus dans plusieurs domaines techniques.',
                quickLinks: 'Liens Rapides',
                features: 'Fonctionnalités Interactives',
                resources: 'Ressources & Outils',
                allProjects: 'Tous les Projets',
                githubProfile: 'Profil GitHub',
                contactMe: 'Me Contacter',
                linkedinProfile: 'Profil LinkedIn',
                downloadResume: 'Télécharger le CV',
                emailMe: 'M\'envoyer un Email',
                rights: 'Tous droits réservés.',
                privacy: 'Politique de Confidentialité',
                terms: 'Conditions d\'Utilisation',
                sitemap: 'Plan du Site'
            }
        },
        es: {
            nav: {
                home: 'Inicio',
                about: 'Sobre Mí',
                skills: 'Habilidades',
                projects: 'Proyectos',
                experience: 'Experiencia',
                certifications: 'Certificaciones',
                contact: 'Contacto'
            },
            hero: {
                greeting: 'Hola, soy',
                name: 'Louis Antoine',
                subtitle: 'Ingeniero Eléctrico y Científico Aplicado',
                desc1: 'Experiencia práctica en fabricación de semiconductores, metrología óptica y desarrollo de procesos avanzados. Sólida formación en ciencia de materiales, física de dispositivos y operaciones en sala limpia.',
                desc2: 'Apasionado por aplicar aprendizaje automático, análisis de datos y métodos estadísticos para resolver desafíos de ingeniería complejos. Experimentado en optimización de procesos, mejora del rendimiento y colaboración interfuncional.',
                cta: 'Ver Proyectos',
                contact: 'Contáctame'
            },
            about: {
                title: 'Sobre Mí',
                description: 'Ingeniero y científico aplicado con experiencia en fabricación avanzada, optimización basada en datos, aprendizaje automático y desarrollo de procesos en semiconductores, fotónica y tecnologías emergentes.'
            },
            skills: {
                title: 'Habilidades Técnicas',
                learnMore: 'Más Información sobre Habilidades'
            },
            projects: {
                title: 'Proyectos Destacados',
                viewProject: 'Ver Proyecto',
                viewDetails: 'Ver Detalles',
                learnMore: 'Más Información sobre Proyectos'
            },
            experience: {
                title: 'Experiencia Profesional',
                learnMore: 'Más Información sobre Experiencia'
            },
            certifications: {
                title: 'Certificaciones',
                viewCertificate: 'Ver Certificado',
                learnMore: 'Más Información sobre Certificaciones'
            },
            contact: {
                title: 'Contáctame',
                email: 'Email',
                phone: 'Teléfono',
                location: 'Ubicación',
                schedule: 'Programar una Reunión',
                scheduleDesc: 'Reserve un horario directamente en mi calendario',
                bookMeeting: 'Reservar Reunión',
                sendMessage: 'Envíame un mensaje',
                name: 'Nombre',
                company: 'Empresa/Organización',
                subject: 'Asunto',
                message: 'Mensaje',
                send: 'Enviar Mensaje'
            },
            newsletter: {
                title: 'Mantente Actualizado',
                description: 'Recibe notificaciones sobre nuevos proyectos, insights técnicos y actualizaciones de la industria.',
                placeholder: 'Ingresa tu email',
                subscribe: 'Suscribirse',
                privacy: 'Sin spam, cancela cuando quieras.'
            },
            footer: {
                aboutTitle: 'Sobre Mí',
                aboutDesc: 'Ingeniero y científico aplicado con experiencia en fabricación, análisis de datos y optimización de procesos en múltiples dominios técnicos.',
                quickLinks: 'Enlaces Rápidos',
                features: 'Funciones Interactivas',
                resources: 'Recursos y Herramientas',
                allProjects: 'Todos los Proyectos',
                githubProfile: 'Perfil de GitHub',
                contactMe: 'Contáctame',
                linkedinProfile: 'Perfil de LinkedIn',
                downloadResume: 'Descargar CV',
                emailMe: 'Envíame un Email',
                rights: 'Todos los derechos reservados.',
                privacy: 'Política de Privacidad',
                terms: 'Términos de Servicio',
                sitemap: 'Mapa del Sitio'
            }
        }
    };

    const STORAGE_KEY = 'portfolio-language';

    // Get saved language or detect from browser
    function getLanguage() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved && translations[saved]) {
            return saved;
        }
        // Try to detect browser language
        const browserLang = navigator.language.split('-')[0];
        if (translations[browserLang]) {
            return browserLang;
        }
        return 'en';
    }

    // Save language preference
    function setLanguage(lang) {
        if (translations[lang]) {
            localStorage.setItem(STORAGE_KEY, lang);
            applyTranslations(lang);
            updateLanguageUI(lang);
        }
    }

    // Apply translations to all elements with data-i18n attribute
    function applyTranslations(lang) {
        const t = translations[lang];
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const keys = key.split('.');
            let value = t;
            for (const k of keys) {
                if (value && value[k]) {
                    value = value[k];
                } else {
                    value = null;
                    break;
                }
            }
            if (value) {
                if (el.tagName === 'INPUT' && el.hasAttribute('placeholder')) {
                    el.placeholder = value;
                } else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                    // For labels, update the associated label
                } else {
                    el.textContent = value;
                }
            }
        });

        // Update HTML lang attribute
        document.documentElement.lang = lang;
    }

    // Update language selector UI
    function updateLanguageUI(lang) {
        const currentLangEl = document.getElementById('currentLang');
        if (currentLangEl) {
            currentLangEl.textContent = lang.toUpperCase();
        }

        // Update active state in dropdown
        document.querySelectorAll('.lang-option').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === lang);
        });
    }

    // Initialize language system
    function init() {
        const currentLang = getLanguage();

        // Apply saved/detected language
        applyTranslations(currentLang);
        updateLanguageUI(currentLang);

        // Setup language toggle button
        const langToggle = document.getElementById('langToggle');
        const langDropdown = document.getElementById('langDropdown');

        if (langToggle && langDropdown) {
            langToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                langDropdown.classList.toggle('show');
            });

            // Handle language selection
            document.querySelectorAll('.lang-option').forEach(btn => {
                btn.addEventListener('click', () => {
                    const lang = btn.dataset.lang;
                    setLanguage(lang);
                    langDropdown.classList.remove('show');
                });
            });

            // Close dropdown when clicking outside
            document.addEventListener('click', () => {
                langDropdown.classList.remove('show');
            });
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Export for external use
    window.i18n = {
        getLanguage,
        setLanguage,
        translations
    };
})();
