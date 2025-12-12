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
                title: 'Electrical Engineer & Applied Scientist',
                subtitle: 'Transforming complex engineering challenges into innovative solutions through data-driven optimization and cutting-edge technology.',
                cta: 'View My Work',
                contact: 'Get In Touch'
            },
            about: {
                title: 'About Me',
                description: 'Engineer and Applied Scientist with expertise in advanced manufacturing, data-driven optimization, machine learning, and process development across semiconductor, photonics, and emerging technologies.'
            },
            skills: {
                title: 'Technical Skills'
            },
            projects: {
                title: 'Featured Projects',
                viewProject: 'View Project',
                viewDetails: 'View Details'
            },
            experience: {
                title: 'Professional Experience'
            },
            certifications: {
                title: 'Certifications',
                viewCertificate: 'View Certificate'
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
                send: 'Send Message',
                success: 'Thank you! Your message has been sent successfully.',
                error: 'Oops! Something went wrong. Please try again.'
            },
            newsletter: {
                title: 'Stay Updated',
                description: 'Get notified about new projects, technical insights, and industry updates.',
                placeholder: 'Enter your email',
                subscribe: 'Subscribe',
                privacy: 'No spam, unsubscribe anytime.',
                success: 'Thank you for subscribing!',
                error: 'Something went wrong. Please try again.'
            },
            footer: {
                about: 'About',
                quickLinks: 'Quick Links',
                features: 'Interactive Features',
                resources: 'Resources & Tools',
                rights: 'All rights reserved.'
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
                greeting: "Bonjour, je suis",
                title: 'Ingénieur Électrique & Scientifique Appliqué',
                subtitle: 'Transformer des défis d\'ingénierie complexes en solutions innovantes grâce à l\'optimisation basée sur les données et aux technologies de pointe.',
                cta: 'Voir Mon Travail',
                contact: 'Me Contacter'
            },
            about: {
                title: 'À Propos de Moi',
                description: 'Ingénieur et scientifique appliqué avec expertise en fabrication avancée, optimisation basée sur les données, apprentissage automatique et développement de processus dans les semiconducteurs, la photonique et les technologies émergentes.'
            },
            skills: {
                title: 'Compétences Techniques'
            },
            projects: {
                title: 'Projets en Vedette',
                viewProject: 'Voir le Projet',
                viewDetails: 'Voir les Détails'
            },
            experience: {
                title: 'Expérience Professionnelle'
            },
            certifications: {
                title: 'Certifications',
                viewCertificate: 'Voir le Certificat'
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
                send: 'Envoyer le Message',
                success: 'Merci! Votre message a été envoyé avec succès.',
                error: 'Oops! Une erreur s\'est produite. Veuillez réessayer.'
            },
            newsletter: {
                title: 'Restez Informé',
                description: 'Recevez des notifications sur les nouveaux projets, les insights techniques et les mises à jour de l\'industrie.',
                placeholder: 'Entrez votre email',
                subscribe: 'S\'abonner',
                privacy: 'Pas de spam, désabonnez-vous à tout moment.',
                success: 'Merci de vous être abonné!',
                error: 'Une erreur s\'est produite. Veuillez réessayer.'
            },
            footer: {
                about: 'À Propos',
                quickLinks: 'Liens Rapides',
                features: 'Fonctionnalités Interactives',
                resources: 'Ressources & Outils',
                rights: 'Tous droits réservés.'
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
                greeting: "Hola, soy",
                title: 'Ingeniero Eléctrico y Científico Aplicado',
                subtitle: 'Transformando desafíos de ingeniería complejos en soluciones innovadoras a través de optimización basada en datos y tecnología de vanguardia.',
                cta: 'Ver Mi Trabajo',
                contact: 'Contáctame'
            },
            about: {
                title: 'Sobre Mí',
                description: 'Ingeniero y científico aplicado con experiencia en fabricación avanzada, optimización basada en datos, aprendizaje automático y desarrollo de procesos en semiconductores, fotónica y tecnologías emergentes.'
            },
            skills: {
                title: 'Habilidades Técnicas'
            },
            projects: {
                title: 'Proyectos Destacados',
                viewProject: 'Ver Proyecto',
                viewDetails: 'Ver Detalles'
            },
            experience: {
                title: 'Experiencia Profesional'
            },
            certifications: {
                title: 'Certificaciones',
                viewCertificate: 'Ver Certificado'
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
                send: 'Enviar Mensaje',
                success: '¡Gracias! Tu mensaje ha sido enviado exitosamente.',
                error: '¡Ups! Algo salió mal. Por favor, inténtalo de nuevo.'
            },
            newsletter: {
                title: 'Mantente Actualizado',
                description: 'Recibe notificaciones sobre nuevos proyectos, insights técnicos y actualizaciones de la industria.',
                placeholder: 'Ingresa tu email',
                subscribe: 'Suscribirse',
                privacy: 'Sin spam, cancela cuando quieras.',
                success: '¡Gracias por suscribirte!',
                error: 'Algo salió mal. Por favor, inténtalo de nuevo.'
            },
            footer: {
                about: 'Sobre Mí',
                quickLinks: 'Enlaces Rápidos',
                features: 'Funciones Interactivas',
                resources: 'Recursos y Herramientas',
                rights: 'Todos los derechos reservados.'
            }
        },
        ht: {
            nav: {
                home: 'Akèy',
                about: 'Sou Mwen',
                skills: 'Konpetans',
                projects: 'Pwojè',
                experience: 'Eksperyans',
                certifications: 'Sètifikasyon',
                contact: 'Kontakte'
            },
            hero: {
                greeting: "Bonjou, mwen se",
                title: 'Enjenyè Elektrik & Syantis Aplike',
                subtitle: 'Transfòme defi enjenyeri konplèks nan solisyon inovatè atravè optimize done ak teknoloji modèn.',
                cta: 'Gade Travay Mwen',
                contact: 'Kontakte Mwen'
            },
            about: {
                title: 'Sou Mwen',
                description: 'Enjenyè ak syantis aplike ki gen ekspètiz nan fabrikasyon avanse, optimize done, aprantisaj machin, ak devlopman pwosesis nan semi-kondiktè, fotonik, ak teknoloji emèjan.'
            },
            skills: {
                title: 'Konpetans Teknik'
            },
            projects: {
                title: 'Pwojè Prensipal',
                viewProject: 'Gade Pwojè',
                viewDetails: 'Gade Detay'
            },
            experience: {
                title: 'Eksperyans Pwofesyonèl'
            },
            certifications: {
                title: 'Sètifikasyon',
                viewCertificate: 'Gade Sètifika'
            },
            contact: {
                title: 'Kontakte Mwen',
                email: 'Imèl',
                phone: 'Telefòn',
                location: 'Kote',
                schedule: 'Pwograme yon Reyinyon',
                scheduleDesc: 'Rezève yon lè dirèkteman sou kalandriye mwen',
                bookMeeting: 'Rezève Reyinyon',
                sendMessage: 'Voye yon mesaj ban mwen',
                name: 'Non',
                company: 'Konpayi/Òganizasyon',
                subject: 'Sijè',
                message: 'Mesaj',
                send: 'Voye Mesaj',
                success: 'Mèsi! Mesaj ou te voye avèk siksè.',
                error: 'Oup! Gen yon bagay ki mal pase. Tanpri eseye ankò.'
            },
            newsletter: {
                title: 'Rete Okouran',
                description: 'Resevwa notifikasyon sou nouvo pwojè, enfòmasyon teknik, ak mizajou endistri.',
                placeholder: 'Antre imèl ou',
                subscribe: 'Abònman',
                privacy: 'Pa gen spam, dezabònman nenpòt lè.',
                success: 'Mèsi pou abònman ou!',
                error: 'Gen yon bagay ki mal pase. Tanpri eseye ankò.'
            },
            footer: {
                about: 'Sou Mwen',
                quickLinks: 'Lyen Rapid',
                features: 'Fonksyon Entèaktif',
                resources: 'Resous & Zouti',
                rights: 'Tout dwa rezève.'
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
