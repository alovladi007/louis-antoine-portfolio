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
                contact: 'Get in Touch',
                scroll: 'Scroll to explore'
            },
            about: {
                title: 'About Me',
                description: 'Engineer and Applied Scientist with expertise in advanced manufacturing, data-driven optimization, machine learning, and process development across semiconductor, photonics, and emerging technologies.',
                tabIntro: 'About Me',
                tabIntroBack: 'Discover',
                tabWork: 'What I Do',
                tabWorkBack: 'Explore',
                tabMission: 'My Mission',
                tabMissionBack: 'Vision',
                introHeading: 'Electrical Engineer & Applied Scientist',
                introBody: 'Electrical engineer with a strong background in semiconductor manufacturing, optical systems, and data-driven process optimization. I combine hands-on technical expertise with analytical skills to tackle complex challenges across manufacturing, R&D, and emerging technology domains.',
                slideProfileTitle: 'Louis Antoine',
                slideProfileRole: 'Process Engineer',
                slideAcademicTitle: 'Academic Excellence',
                slideAcademicSub: 'Continuous Learning & Growth',
                tagSemiconductor: 'Semiconductor Fabrication',
                tagQuantum: 'Quantum Optics',
                tagML: 'Machine Learning',
                tagDesign: 'Engineering Design',
                workHeading: 'Bridging Theory and Practice',
                workBody: 'With expertise in both theoretical research and hands-on implementation, I bridge the gap between complex scientific concepts and real-world applications.',
                slideCleanroomTitle: 'Clean Room Operations',
                slideCleanroomSub: 'Advanced Semiconductor Processing',
                slideOpticsTitle: 'Optical Research',
                slideOpticsSub: 'Photonics & Quantum Systems',
                slideThinFilmTitle: 'Thin Film Technology',
                slideThinFilmSub: 'Material Science Innovation',
                workSemiTitle: 'Semiconductor Process',
                workSemiSub: 'Advanced fabrication techniques',
                workDesignTitle: 'Design Systems',
                workDesignSub: 'Next-generation solutions',
                workControlTitle: 'Control Systems',
                workControlSub: 'Advanced automation',
                workInnovationTitle: 'Innovation',
                workInnovationSub: 'Scientific breakthroughs',
                missionHeading: 'Driving Innovation Forward',
                missionBody: "I'm passionate about pushing the boundaries of technology and driving breakthrough innovations that shape the future.",
                slideGlobalTitle: 'Global Research',
                slideGlobalSub: 'Pushing Scientific Boundaries',
                slideMaterialsTitle: 'Advanced Materials',
                slideMaterialsSub: 'Next-Gen Technology',
                hexElectronicsTitle: 'Electronics',
                hexElectronicsSub: 'Next-gen semiconductors',
                hexPhotonicsTitle: 'Photonics',
                hexPhotonicsSub: 'Optical innovations',
                hexAutomationTitle: 'Automated Systems',
                hexAutomationSub: 'Intelligent solutions'
            },
            skills: {
                title: 'Technical Skills',
                subtitle: 'Expertise across semiconductor fabrication, programming, and cutting-edge technologies',
                learnMore: 'Learn More About Skills',
                catSemiTitle: 'Semiconductor Fabrication',
                catSemi1: 'Photolithography',
                catSemi2: 'Thin Film Deposition (PVD/CVD/ALD)',
                catSemi3: 'Dry Etch & CMP',
                catSemi4: 'ASML Metrology Systems',
                catProgTitle: 'Programming & Analysis',
                catProg1: 'Python Programming',
                catProg2: 'MATLAB/Simulink',
                catProg3: 'Machine Learning & Data Analysis',
                catProg4: 'Statistical Process Control (SPC)',
                catAdvTitle: 'Advanced Technologies',
                catAdv1: 'Quantum Optics & EIT',
                catAdv2: 'GaN/SiC Power Devices',
                catAdv3: 'EUV Lithography',
                catAdv4: 'Monte Carlo Simulation'
            },
            projects: {
                title: 'Featured Projects',
                viewProject: 'View Project',
                viewDetails: 'View Project Details',
                learnMore: 'Learn More About Projects',
                quantumMemTitle: 'Quantum Memory System for EIT',
                quantumMemDesc: 'Advanced quantum memory system utilizing Electromagnetically Induced Transparency in cold atom ensembles for quantum information processing.',
                maxwellBlochTitle: 'Full Maxwell-Bloch Simulation for EIT',
                maxwellBlochDesc: 'Complete simulation framework for quantum optics experiments based on Maxwell-Bloch equations, featuring Doppler broadening, pulse storage/retrieval, and geometry effects.',
                semiOptTitle: 'Semiconductor Process Optimization',
                semiOptDesc: 'Advanced ML-driven process control system combining CatBoost-based virtual metrology, double-EWMA control algorithms, and SPC/FDC integration. Achieves 15-25% yield improvement through real-time parameter tuning, predictive maintenance, and automated recipe optimization across lithography, etch, and deposition modules.',
                duvTitle: 'DUV Energy Deposition: Monte Carlo vs Double Gaussian',
                duvDesc: 'Complete simulator comparing aerial image energy deposition predicted by Double-Gaussian PSF (FFT convolution) against Monte Carlo particle model. Features partial coherence modeling, flare analysis, swing curves, and comprehensive statistical validation with PDF report generation.',
                ganTitle: 'Vertical GaN Power Electronics',
                ganDesc: "Revolutionary 3D architecture for >1kV power devices. Vertical GaN FETs merge GaN's superior switching speed with SiC-class voltage handling, enabling next-generation EV drivetrains and utility-scale power converters.",
                siPhotTitle: 'Silicon Photonics - Microring Resonator',
                siPhotDesc: 'Weekend photonics project: Design and simulate a silicon microring resonator WDM filter @ 1550nm. Complete PIC workflow from layout to FDTD to circuit verification. Q≈1950, ER≥20dB, FSR≈100GHz.',
                hubElectronics: 'Electronics',
                hubPhotonics: 'Photonics',
                hubMlds: 'ML & DS',
                hubInnovation: 'Innovation'
            },
            experience: {
                title: 'Experience & Education',
                learnMore: 'Learn More About Experience',
                catEducation: 'Education',
                catExperience: 'Experience',
                catMilitary: 'Military Service',
                catAdditional: 'Additional Experience',
                learnCoursework: 'Learn More About Coursework',
                learnRole: 'Learn More About This Role',
                msTitle: 'Master of Science, Electrical Engineering',
                msSchool: 'University of Connecticut - Storrs, CT',
                msDate: '05/2025',
                msDesc: 'Specialized in Electronics, Photonics, and Bio-Photonics. Applied ML methods to data analysis and experimental optimization. Research focus on GaN-based power devices, SiC MOSFETs, and Monte Carlo simulation for EUV lithography.',
                gradResTitle: 'STUDENT EQUIPMENT DESIGN & TEST SPECIALIST',
                gradResSchool: 'University of Connecticut - Storrs, CT',
                gradResDate: '10/2021 - 04/2024',
                gradResDesc: 'Supported semiconductor and optoelectronics labs with precision measurements, front-end and back-end processing technologies. Provided guidance to undergraduate students on independent research projects.',
                asmlTitle: 'Optical Metrology Equipment Operator',
                asmlSchool: 'ASML US - Wilton, CT',
                asmlDate: '05/2021 - 06/2023',
                asmlDesc: 'Worked with YieldStar optical metrology systems for lithography process monitoring. Performed sub-system assembly, optical alignment, equipment calibration, and root cause analysis in cleanroom environment.',
                bsTitle: 'Bachelor of Science, Physics',
                bsSchool: 'University of Connecticut - Storrs, CT',
                bsDesc: 'Graduated with focus on optics, quantum physics, and experimental research. Completed undergraduate research in EIT for slow light and quantum memory demonstrations.',
                ugResTitle: 'Undergraduate Student Research Assistant',
                ugResSchool: 'University of Connecticut - Storrs, CT',
                ugResDesc: 'Conducted experimental and theoretical research on Electromagnetically Induced Transparency (EIT) in atomic systems, Monte Carlo simulations of molecular beams, and advanced quantum materials. Senior thesis received departmental honors.',
                armyTitle: 'Patient Administration Specialist (68G)',
                armySchool: 'United States Army - Fort Stewart, GA',
                armyDesc: 'Managed medical records, patient admissions, and healthcare administration for over 3,000 active duty personnel. Coordinated MEDEVAC operations and maintained HIPAA compliance.',
                aaTitle: 'Associate of Science, Engineering Science',
                aaSchool: 'CT State Community College Housatonic - Bridgeport',
                aaDesc: 'Foundation in engineering principles and mathematics, preparing for advanced studies in physics and electrical engineering.',
                addTitle: 'Additional Work Experience',
                addUberTitle: 'Professional Driver (Uber/Lyft)',
                addUberLoc: 'Storrs, CT',
                addUberDesc: 'Provided professional transportation services while completing undergraduate degree. Maintained excellent customer service ratings and demonstrated strong time management skills.',
                addCashierTitle: 'Cashier',
                addCashierLoc: 'Walmart - Norwalk, Connecticut',
                addCashierDesc: 'Provided excellent customer service in high-volume retail environment. Handled cash transactions and inventory management with accuracy and professionalism.'
            },
            certifications: {
                title: 'Professional Certifications',
                viewCertificate: 'View Certificate',
                learnMore: 'Learn More About Certifications',
                badgeScheduled: 'Scheduled',
                placeholderScheduled: 'Scheduled',
                usptoTitle: 'USPTO Patent Bar',
                usptoDesc: 'Qualified to practice before the United States Patent and Trademark Office',
                feTitle: 'FE Electrical Engineering',
                feDesc: 'Fundamentals of Engineering certification in Electrical Engineering',
                comptiaTitle: 'CompTIA Security+',
                comptiaDesc: 'Industry-standard cybersecurity certification',
                sixSigmaTitle: 'Six Sigma Green Belt',
                sixSigmaDesc: 'Process improvement and quality management methodology',
                gadaTitle: 'Google Advanced Data Analytics',
                gadaDesc: 'Professional certificate in advanced data analytics and machine learning',
                gitaTitle: 'Google IT Automation with Python',
                gitaDesc: 'Professional certificate in Python automation and scripting'
            },
            contact: {
                title: 'Get In Touch',
                email: 'Email',
                phone: 'Phone',
                location: 'Location',
                locationValue: 'New Haven, CT 06515',
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
                sitemap: 'Sitemap',
                projectComparison: 'Project Comparison Tool',
                skillsMatrix: 'Skills Matrix Dashboard',
                caseStudies: 'Case Studies',
                interactiveTutorials: 'Interactive Tutorials',
                designDecisions: 'Design Decisions',
                performanceOpt: 'Performance Optimization',
                copyright: '© 2024 Louis Antoine. All rights reserved.'
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
                desc2: "Passionné par l'application de l'apprentissage automatique, l'analyse de données et les méthodes statistiques pour résoudre des défis d'ingénierie complexes. Expérimenté en optimisation de processus, amélioration du rendement et collaboration interfonctionnelle.",
                cta: 'Voir les Projets',
                contact: 'Me Contacter',
                scroll: 'Faites défiler pour explorer'
            },
            about: {
                title: 'À Propos de Moi',
                description: 'Ingénieur et scientifique appliqué avec expertise en fabrication avancée, optimisation basée sur les données, apprentissage automatique et développement de processus dans les semiconducteurs, la photonique et les technologies émergentes.',
                tabIntro: 'À Propos',
                tabIntroBack: 'Découvrir',
                tabWork: 'Mon Travail',
                tabWorkBack: 'Explorer',
                tabMission: 'Ma Mission',
                tabMissionBack: 'Vision',
                introHeading: 'Ingénieur Électrique & Scientifique Appliqué',
                introBody: "Ingénieur électrique avec une solide formation en fabrication de semiconducteurs, en systèmes optiques et en optimisation de processus axée sur les données. Je combine une expertise technique pratique avec des compétences analytiques pour relever des défis complexes dans les domaines de la fabrication, de la R&D et des technologies émergentes.",
                slideProfileTitle: 'Louis Antoine',
                slideProfileRole: 'Ingénieur Procédés',
                slideAcademicTitle: 'Excellence Académique',
                slideAcademicSub: 'Apprentissage et Évolution Continus',
                tagSemiconductor: 'Fabrication de Semiconducteurs',
                tagQuantum: 'Optique Quantique',
                tagML: 'Apprentissage Automatique',
                tagDesign: 'Conception en Ingénierie',
                workHeading: 'Lier Théorie et Pratique',
                workBody: "Avec une expertise en recherche théorique et en mise en œuvre pratique, je fais le pont entre les concepts scientifiques complexes et leurs applications concrètes.",
                slideCleanroomTitle: 'Opérations en Salle Blanche',
                slideCleanroomSub: 'Procédés Avancés de Semiconducteurs',
                slideOpticsTitle: 'Recherche Optique',
                slideOpticsSub: 'Photonique & Systèmes Quantiques',
                slideThinFilmTitle: 'Technologie des Couches Minces',
                slideThinFilmSub: 'Innovation en Science des Matériaux',
                workSemiTitle: 'Procédés Semiconducteurs',
                workSemiSub: 'Techniques de fabrication avancées',
                workDesignTitle: 'Systèmes de Conception',
                workDesignSub: 'Solutions de nouvelle génération',
                workControlTitle: 'Systèmes de Contrôle',
                workControlSub: 'Automatisation avancée',
                workInnovationTitle: 'Innovation',
                workInnovationSub: 'Avancées scientifiques',
                missionHeading: "Faire Avancer l'Innovation",
                missionBody: "Je suis passionné par le repoussement des limites de la technologie et la mise en place d'innovations de rupture qui façonnent l'avenir.",
                slideGlobalTitle: 'Recherche Mondiale',
                slideGlobalSub: 'Repousser les Frontières Scientifiques',
                slideMaterialsTitle: 'Matériaux Avancés',
                slideMaterialsSub: 'Technologie de Nouvelle Génération',
                hexElectronicsTitle: 'Électronique',
                hexElectronicsSub: 'Semiconducteurs nouvelle génération',
                hexPhotonicsTitle: 'Photonique',
                hexPhotonicsSub: 'Innovations optiques',
                hexAutomationTitle: 'Systèmes Automatisés',
                hexAutomationSub: 'Solutions intelligentes'
            },
            skills: {
                title: 'Compétences Techniques',
                subtitle: 'Expertise dans la fabrication de semiconducteurs, la programmation et les technologies de pointe',
                learnMore: 'En Savoir Plus sur les Compétences',
                catSemiTitle: 'Fabrication de Semiconducteurs',
                catSemi1: 'Photolithographie',
                catSemi2: 'Dépôt de Couches Minces (PVD/CVD/ALD)',
                catSemi3: 'Gravure Sèche & CMP',
                catSemi4: 'Systèmes de Métrologie ASML',
                catProgTitle: 'Programmation & Analyse',
                catProg1: 'Programmation Python',
                catProg2: 'MATLAB/Simulink',
                catProg3: 'Apprentissage Automatique & Analyse de Données',
                catProg4: 'Contrôle Statistique des Procédés (SPC)',
                catAdvTitle: 'Technologies Avancées',
                catAdv1: 'Optique Quantique & EIT',
                catAdv2: 'Dispositifs de Puissance GaN/SiC',
                catAdv3: 'Lithographie EUV',
                catAdv4: 'Simulation Monte Carlo'
            },
            projects: {
                title: 'Projets en Vedette',
                viewProject: 'Voir le Projet',
                viewDetails: 'Voir les Détails du Projet',
                learnMore: 'En Savoir Plus sur les Projets',
                quantumMemTitle: 'Système de Mémoire Quantique pour EIT',
                quantumMemDesc: "Système de mémoire quantique avancé exploitant la Transparence Induite Électromagnétiquement dans des ensembles d'atomes froids pour le traitement de l'information quantique.",
                maxwellBlochTitle: 'Simulation Complète Maxwell-Bloch pour EIT',
                maxwellBlochDesc: "Cadre de simulation complet pour les expériences d'optique quantique basé sur les équations de Maxwell-Bloch, comprenant l'élargissement Doppler, le stockage/récupération d'impulsions et les effets géométriques.",
                semiOptTitle: 'Optimisation des Procédés Semiconducteurs',
                semiOptDesc: "Système avancé de contrôle de procédés piloté par ML, combinant la métrologie virtuelle basée sur CatBoost, des algorithmes de contrôle double EWMA et l'intégration SPC/FDC. Permet une amélioration du rendement de 15 à 25 % grâce à un réglage des paramètres en temps réel, une maintenance prédictive et une optimisation automatisée des recettes pour la lithographie, la gravure et le dépôt.",
                duvTitle: 'Dépôt d’Énergie DUV : Monte Carlo vs Double Gaussienne',
                duvDesc: "Simulateur complet comparant le dépôt d'énergie de l'image aérienne prédit par PSF Double-Gaussienne (convolution FFT) à un modèle particulaire Monte Carlo. Modélisation de la cohérence partielle, analyse du flare, courbes de swing et validation statistique complète avec génération de rapports PDF.",
                ganTitle: 'Électronique de Puissance GaN Verticale',
                ganDesc: "Architecture 3D révolutionnaire pour des dispositifs de puissance >1 kV. Les FET GaN verticaux combinent la vitesse de commutation supérieure du GaN avec la tenue en tension de classe SiC, permettant les chaînes de traction VE de nouvelle génération et les convertisseurs de puissance à grande échelle.",
                siPhotTitle: 'Photonique Silicium - Résonateur en Anneau',
                siPhotDesc: "Projet photonique du week-end : conception et simulation d'un filtre WDM à résonateur en anneau silicium @ 1550 nm. Flux PIC complet, du layout à la FDTD jusqu'à la vérification du circuit. Q≈1950, ER≥20 dB, FSR≈100 GHz.",
                hubElectronics: 'Électronique',
                hubPhotonics: 'Photonique',
                hubMlds: 'ML & DS',
                hubInnovation: 'Innovation'
            },
            experience: {
                title: 'Expérience & Formation',
                learnMore: "En Savoir Plus sur l'Expérience",
                catEducation: 'Formation',
                catExperience: 'Expérience',
                catMilitary: 'Service Militaire',
                catAdditional: 'Expérience Complémentaire',
                learnCoursework: 'En Savoir Plus sur le Programme',
                learnRole: 'En Savoir Plus sur ce Poste',
                msTitle: 'Master en Génie Électrique',
                msSchool: 'University of Connecticut - Storrs, CT',
                msDate: '05/2025',
                msDesc: "Spécialisation en électronique, photonique et bio-photonique. Application des méthodes de ML à l'analyse de données et à l'optimisation expérimentale. Recherche axée sur les dispositifs de puissance GaN, les MOSFET SiC et la simulation Monte Carlo pour la lithographie EUV.",
                gradResTitle: "ÉTUDIANT SPÉCIALISTE EN CONCEPTION ET ESSAI D'ÉQUIPEMENTS",
                gradResSchool: 'University of Connecticut - Storrs, CT',
                gradResDate: '10/2021 - 04/2024',
                gradResDesc: "Soutien des laboratoires de semiconducteurs et d'optoélectronique avec des mesures de précision et des technologies de procédé front-end et back-end. Encadrement d'étudiants de premier cycle sur des projets de recherche indépendants.",
                asmlTitle: 'Opérateur de Métrologie Optique',
                asmlSchool: 'ASML US - Wilton, CT',
                asmlDate: '05/2021 - 06/2023',
                asmlDesc: "Travail sur les systèmes de métrologie optique YieldStar pour la surveillance des procédés de lithographie. Assemblage de sous-systèmes, alignement optique, étalonnage des équipements et analyse des causes profondes en environnement salle blanche.",
                bsTitle: 'Licence en Physique',
                bsSchool: 'University of Connecticut - Storrs, CT',
                bsDesc: "Diplômé avec un focus sur l'optique, la physique quantique et la recherche expérimentale. Réalisation de recherches en EIT pour la lumière lente et démonstrations de mémoire quantique.",
                ugResTitle: 'Assistant Étudiant de Recherche',
                ugResSchool: 'University of Connecticut - Storrs, CT',
                ugResDesc: "Recherches expérimentales et théoriques sur la Transparence Induite Électromagnétiquement (EIT) dans les systèmes atomiques, simulations Monte Carlo de faisceaux moléculaires et matériaux quantiques avancés. Mémoire de fin d'études récompensé par les honneurs du département.",
                armyTitle: 'Spécialiste en Administration des Patients (68G)',
                armySchool: 'Armée des États-Unis - Fort Stewart, GA',
                armyDesc: "Gestion des dossiers médicaux, des admissions et de l'administration des soins pour plus de 3 000 militaires en service actif. Coordination des opérations MEDEVAC et conformité HIPAA.",
                aaTitle: "Diplôme Associé en Sciences de l'Ingénieur",
                aaSchool: 'CT State Community College Housatonic - Bridgeport',
                aaDesc: "Bases en principes d'ingénierie et mathématiques, en préparation d'études avancées en physique et génie électrique.",
                addTitle: 'Expérience Professionnelle Complémentaire',
                addUberTitle: 'Chauffeur Professionnel (Uber/Lyft)',
                addUberLoc: 'Storrs, CT',
                addUberDesc: "Services de transport professionnels durant mes études de licence. Excellentes notes de service client et solides compétences en gestion du temps.",
                addCashierTitle: 'Caissier',
                addCashierLoc: 'Walmart - Norwalk, Connecticut',
                addCashierDesc: 'Excellent service client en environnement de vente à fort volume. Gestion précise et professionnelle des transactions et de l’inventaire.'
            },
            certifications: {
                title: 'Certifications Professionnelles',
                viewCertificate: 'Voir le Certificat',
                learnMore: 'En Savoir Plus sur les Certifications',
                badgeScheduled: 'Programmé',
                placeholderScheduled: 'Programmé',
                usptoTitle: 'USPTO Patent Bar',
                usptoDesc: "Qualifié pour exercer devant l'Office des Brevets et des Marques des États-Unis",
                feTitle: 'FE Génie Électrique',
                feDesc: "Certification Fundamentals of Engineering en génie électrique",
                comptiaTitle: 'CompTIA Security+',
                comptiaDesc: 'Certification standard du secteur en cybersécurité',
                sixSigmaTitle: 'Six Sigma Ceinture Verte',
                sixSigmaDesc: "Méthodologie d'amélioration des processus et de gestion de la qualité",
                gadaTitle: "Google Advanced Data Analytics",
                gadaDesc: "Certificat professionnel en analyse de données avancée et apprentissage automatique",
                gitaTitle: 'Google IT Automation avec Python',
                gitaDesc: "Certificat professionnel en automatisation et scripting Python"
            },
            contact: {
                title: 'Me Contacter',
                email: 'Email',
                phone: 'Téléphone',
                location: 'Localisation',
                locationValue: 'New Haven, CT 06515',
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
                description: "Recevez des notifications sur les nouveaux projets, les insights techniques et les mises à jour de l'industrie.",
                placeholder: 'Entrez votre email',
                subscribe: "S'abonner",
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
                emailMe: "M'envoyer un Email",
                rights: 'Tous droits réservés.',
                privacy: 'Politique de Confidentialité',
                terms: "Conditions d'Utilisation",
                sitemap: 'Plan du Site',
                projectComparison: 'Outil de Comparaison de Projets',
                skillsMatrix: 'Tableau de Bord des Compétences',
                caseStudies: 'Études de Cas',
                interactiveTutorials: 'Tutoriels Interactifs',
                designDecisions: 'Décisions de Conception',
                performanceOpt: 'Optimisation des Performances',
                copyright: '© 2024 Louis Antoine. Tous droits réservés.'
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
                contact: 'Contáctame',
                scroll: 'Desplázate para explorar'
            },
            about: {
                title: 'Sobre Mí',
                description: 'Ingeniero y científico aplicado con experiencia en fabricación avanzada, optimización basada en datos, aprendizaje automático y desarrollo de procesos en semiconductores, fotónica y tecnologías emergentes.',
                tabIntro: 'Sobre Mí',
                tabIntroBack: 'Descubrir',
                tabWork: 'Lo Que Hago',
                tabWorkBack: 'Explorar',
                tabMission: 'Mi Misión',
                tabMissionBack: 'Visión',
                introHeading: 'Ingeniero Eléctrico y Científico Aplicado',
                introBody: 'Ingeniero eléctrico con sólida formación en fabricación de semiconductores, sistemas ópticos y optimización de procesos basada en datos. Combino experiencia técnica práctica con habilidades analíticas para abordar desafíos complejos en fabricación, I+D y tecnologías emergentes.',
                slideProfileTitle: 'Louis Antoine',
                slideProfileRole: 'Ingeniero de Procesos',
                slideAcademicTitle: 'Excelencia Académica',
                slideAcademicSub: 'Aprendizaje y Crecimiento Continuos',
                tagSemiconductor: 'Fabricación de Semiconductores',
                tagQuantum: 'Óptica Cuántica',
                tagML: 'Aprendizaje Automático',
                tagDesign: 'Diseño en Ingeniería',
                workHeading: 'Uniendo Teoría y Práctica',
                workBody: 'Con experiencia tanto en investigación teórica como en implementación práctica, conecto los conceptos científicos complejos con sus aplicaciones reales.',
                slideCleanroomTitle: 'Operaciones en Sala Limpia',
                slideCleanroomSub: 'Procesos Avanzados de Semiconductores',
                slideOpticsTitle: 'Investigación Óptica',
                slideOpticsSub: 'Fotónica y Sistemas Cuánticos',
                slideThinFilmTitle: 'Tecnología de Capas Delgadas',
                slideThinFilmSub: 'Innovación en Ciencia de Materiales',
                workSemiTitle: 'Procesos de Semiconductores',
                workSemiSub: 'Técnicas avanzadas de fabricación',
                workDesignTitle: 'Sistemas de Diseño',
                workDesignSub: 'Soluciones de próxima generación',
                workControlTitle: 'Sistemas de Control',
                workControlSub: 'Automatización avanzada',
                workInnovationTitle: 'Innovación',
                workInnovationSub: 'Avances científicos',
                missionHeading: 'Impulsando la Innovación',
                missionBody: 'Me apasiona empujar los límites de la tecnología e impulsar innovaciones disruptivas que dan forma al futuro.',
                slideGlobalTitle: 'Investigación Global',
                slideGlobalSub: 'Empujando los Límites Científicos',
                slideMaterialsTitle: 'Materiales Avanzados',
                slideMaterialsSub: 'Tecnología de Próxima Generación',
                hexElectronicsTitle: 'Electrónica',
                hexElectronicsSub: 'Semiconductores de próxima generación',
                hexPhotonicsTitle: 'Fotónica',
                hexPhotonicsSub: 'Innovaciones ópticas',
                hexAutomationTitle: 'Sistemas Automatizados',
                hexAutomationSub: 'Soluciones inteligentes'
            },
            skills: {
                title: 'Habilidades Técnicas',
                subtitle: 'Experiencia en fabricación de semiconductores, programación y tecnologías de vanguardia',
                learnMore: 'Más Información sobre Habilidades',
                catSemiTitle: 'Fabricación de Semiconductores',
                catSemi1: 'Fotolitografía',
                catSemi2: 'Deposición de Capas Delgadas (PVD/CVD/ALD)',
                catSemi3: 'Grabado en Seco y CMP',
                catSemi4: 'Sistemas de Metrología ASML',
                catProgTitle: 'Programación y Análisis',
                catProg1: 'Programación en Python',
                catProg2: 'MATLAB/Simulink',
                catProg3: 'Aprendizaje Automático y Análisis de Datos',
                catProg4: 'Control Estadístico de Procesos (SPC)',
                catAdvTitle: 'Tecnologías Avanzadas',
                catAdv1: 'Óptica Cuántica y EIT',
                catAdv2: 'Dispositivos de Potencia GaN/SiC',
                catAdv3: 'Litografía EUV',
                catAdv4: 'Simulación Monte Carlo'
            },
            projects: {
                title: 'Proyectos Destacados',
                viewProject: 'Ver Proyecto',
                viewDetails: 'Ver Detalles del Proyecto',
                learnMore: 'Más Información sobre Proyectos',
                quantumMemTitle: 'Sistema de Memoria Cuántica para EIT',
                quantumMemDesc: 'Sistema avanzado de memoria cuántica que utiliza Transparencia Inducida Electromagnéticamente en conjuntos de átomos fríos para el procesamiento de información cuántica.',
                maxwellBlochTitle: 'Simulación Completa Maxwell-Bloch para EIT',
                maxwellBlochDesc: 'Marco de simulación completo para experimentos de óptica cuántica basado en las ecuaciones de Maxwell-Bloch, con ensanchamiento Doppler, almacenamiento/recuperación de pulsos y efectos geométricos.',
                semiOptTitle: 'Optimización de Procesos de Semiconductores',
                semiOptDesc: 'Sistema avanzado de control de procesos impulsado por ML que combina metrología virtual basada en CatBoost, algoritmos de control EWMA doble e integración SPC/FDC. Logra una mejora del 15-25 % en el rendimiento mediante ajuste de parámetros en tiempo real, mantenimiento predictivo y optimización automática de recetas en litografía, grabado y deposición.',
                duvTitle: 'Deposición de Energía DUV: Monte Carlo vs Doble Gaussiana',
                duvDesc: 'Simulador completo que compara la deposición de energía de la imagen aérea predicha por PSF Doble-Gaussiana (convolución FFT) con un modelo particulado Monte Carlo. Incluye modelado de coherencia parcial, análisis de flare, curvas de swing y validación estadística completa con generación de informes PDF.',
                ganTitle: 'Electrónica de Potencia GaN Vertical',
                ganDesc: 'Arquitectura 3D revolucionaria para dispositivos de potencia >1 kV. Los FET GaN verticales combinan la velocidad de conmutación superior del GaN con el manejo de tensión clase SiC, habilitando transmisiones EV de próxima generación y convertidores de potencia a escala de servicios públicos.',
                siPhotTitle: 'Fotónica de Silicio - Resonador de Microanillo',
                siPhotDesc: 'Proyecto fotónico de fin de semana: Diseño y simulación de un filtro WDM con resonador de microanillo de silicio @ 1550 nm. Flujo PIC completo desde el diseño hasta FDTD y verificación de circuito. Q≈1950, ER≥20 dB, FSR≈100 GHz.',
                hubElectronics: 'Electrónica',
                hubPhotonics: 'Fotónica',
                hubMlds: 'ML y DS',
                hubInnovation: 'Innovación'
            },
            experience: {
                title: 'Experiencia y Educación',
                learnMore: 'Más Información sobre Experiencia',
                catEducation: 'Educación',
                catExperience: 'Experiencia',
                catMilitary: 'Servicio Militar',
                catAdditional: 'Experiencia Adicional',
                learnCoursework: 'Más Información sobre el Plan de Estudios',
                learnRole: 'Más Información sobre este Rol',
                msTitle: 'Máster en Ingeniería Eléctrica',
                msSchool: 'University of Connecticut - Storrs, CT',
                msDate: '05/2025',
                msDesc: 'Especialización en Electrónica, Fotónica y Bio-Fotónica. Aplicación de métodos de ML al análisis de datos y optimización experimental. Investigación enfocada en dispositivos de potencia GaN, MOSFET de SiC y simulación Monte Carlo para litografía EUV.',
                gradResTitle: 'ESPECIALISTA ESTUDIANTIL EN DISEÑO Y PRUEBAS DE EQUIPOS',
                gradResSchool: 'University of Connecticut - Storrs, CT',
                gradResDate: '10/2021 - 04/2024',
                gradResDesc: 'Apoyo a laboratorios de semiconductores y optoelectrónica con mediciones de precisión y tecnologías de procesamiento de front-end y back-end. Orientación a estudiantes de pregrado en proyectos de investigación independientes.',
                asmlTitle: 'Operador de Equipos de Metrología Óptica',
                asmlSchool: 'ASML US - Wilton, CT',
                asmlDate: '05/2021 - 06/2023',
                asmlDesc: 'Trabajo con sistemas de metrología óptica YieldStar para el monitoreo de procesos de litografía. Realización de ensamblaje de subsistemas, alineación óptica, calibración de equipos y análisis de causa raíz en entorno de sala limpia.',
                bsTitle: 'Licenciatura en Física',
                bsSchool: 'University of Connecticut - Storrs, CT',
                bsDesc: 'Graduado con enfoque en óptica, física cuántica e investigación experimental. Realización de investigación de pregrado en EIT para luz lenta y demostraciones de memoria cuántica.',
                ugResTitle: 'Asistente Estudiantil de Investigación de Pregrado',
                ugResSchool: 'University of Connecticut - Storrs, CT',
                ugResDesc: 'Investigación experimental y teórica sobre Transparencia Inducida Electromagnéticamente (EIT) en sistemas atómicos, simulaciones Monte Carlo de haces moleculares y materiales cuánticos avanzados. La tesis de grado recibió honores del departamento.',
                armyTitle: 'Especialista en Administración de Pacientes (68G)',
                armySchool: 'Ejército de los Estados Unidos - Fort Stewart, GA',
                armyDesc: 'Gestión de registros médicos, admisiones de pacientes y administración de salud para más de 3.000 efectivos en servicio activo. Coordinación de operaciones MEDEVAC y cumplimiento de HIPAA.',
                aaTitle: 'Asociado en Ciencias, Ciencias de la Ingeniería',
                aaSchool: 'CT State Community College Housatonic - Bridgeport',
                aaDesc: 'Fundamentos en principios de ingeniería y matemáticas, preparación para estudios avanzados en física e ingeniería eléctrica.',
                addTitle: 'Experiencia Laboral Adicional',
                addUberTitle: 'Conductor Profesional (Uber/Lyft)',
                addUberLoc: 'Storrs, CT',
                addUberDesc: 'Servicios profesionales de transporte mientras completaba la licenciatura. Excelentes calificaciones de servicio al cliente y sólidas habilidades de gestión del tiempo.',
                addCashierTitle: 'Cajero',
                addCashierLoc: 'Walmart - Norwalk, Connecticut',
                addCashierDesc: 'Excelente servicio al cliente en entorno minorista de alto volumen. Manejo preciso y profesional de transacciones e inventario.'
            },
            certifications: {
                title: 'Certificaciones Profesionales',
                viewCertificate: 'Ver Certificado',
                learnMore: 'Más Información sobre Certificaciones',
                badgeScheduled: 'Programado',
                placeholderScheduled: 'Programado',
                usptoTitle: 'USPTO Patent Bar',
                usptoDesc: 'Calificado para ejercer ante la Oficina de Patentes y Marcas de los Estados Unidos',
                feTitle: 'FE Ingeniería Eléctrica',
                feDesc: 'Certificación Fundamentals of Engineering en Ingeniería Eléctrica',
                comptiaTitle: 'CompTIA Security+',
                comptiaDesc: 'Certificación estándar de la industria en ciberseguridad',
                sixSigmaTitle: 'Six Sigma Cinturón Verde',
                sixSigmaDesc: 'Metodología de mejora de procesos y gestión de la calidad',
                gadaTitle: 'Google Advanced Data Analytics',
                gadaDesc: 'Certificado profesional en análisis de datos avanzado y aprendizaje automático',
                gitaTitle: 'Google IT Automation con Python',
                gitaDesc: 'Certificado profesional en automatización y scripting con Python'
            },
            contact: {
                title: 'Contáctame',
                email: 'Email',
                phone: 'Teléfono',
                location: 'Ubicación',
                locationValue: 'New Haven, CT 06515',
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
                sitemap: 'Mapa del Sitio',
                projectComparison: 'Herramienta de Comparación de Proyectos',
                skillsMatrix: 'Panel de Matriz de Habilidades',
                caseStudies: 'Estudios de Caso',
                interactiveTutorials: 'Tutoriales Interactivos',
                designDecisions: 'Decisiones de Diseño',
                performanceOpt: 'Optimización del Rendimiento',
                copyright: '© 2024 Louis Antoine. Todos los derechos reservados.'
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
