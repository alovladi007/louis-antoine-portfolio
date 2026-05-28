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
                contact: 'Contact',
                search: 'Search'
            },
            hero: {
                greeting: "Hi, I'm",
                name: 'Louis Vladimir Antoine',
                subtitle: 'Electrical Engineer',
                desc1: 'Hardware engineer across mmWave RF, GaN devices, and semiconductor process. Interested in architecture, system integration, and trade-space work.',
                desc2: 'US Army veteran. Registered for the USPTO patent bar.',
                cta: 'View Projects',
                contact: 'Get in Touch',
                scroll: 'Scroll to explore'
            },
            about: {
                title: 'About Me',
                description: 'Electrical engineer focused on semiconductor manufacturing, device research, and quality and inspection work. Production-floor experience at ASML Wilton; current quality control NDT inspection at General Dynamics Electric Boat; graduate device research at UConn.',
                tabIntro: 'About Me',
                tabIntroBack: 'Discover',
                tabWork: 'What I Do',
                tabWorkBack: 'Explore',
                tabMission: 'My Mission',
                tabMissionBack: 'Vision',
                introHeading: 'Electrical Engineer',
                introBody: 'Electrical engineer who learned how production scanners are built before learning how the devices on the wafers behind them actually work. Most useful at the intersection of equipment, process, and device physics, where the same person can read the data, run the diagnostic, and explain what it means.',
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
                spectraLabTitle: 'SPECTRA-Lab: Semiconductor Characterization & MES Platform',
                spectraLabDesc: 'Independent project, in active development. Semiconductor characterization and manufacturing execution platform combining a Python/FastAPI backend with a Next.js 14 frontend, PostgreSQL persistence, and a Celery + Redis async layer. Covers metrology, seven process control areas, SPC with Western Electric rules, ML and virtual metrology, and a LIMS/ELN.',
                maxwellBlochTitle: 'Full Maxwell-Bloch Simulation for EIT',
                maxwellBlochDesc: 'Complete simulation framework for quantum optics experiments based on Maxwell-Bloch equations, featuring Doppler broadening, pulse storage/retrieval, and geometry effects.',
                semiOptTitle: 'Semiconductor Process Optimization',
                semiOptDesc: 'Advanced ML-driven process control system combining CatBoost-based virtual metrology, double-EWMA control algorithms, and SPC/FDC integration. Achieves 15-25% yield improvement through real-time parameter tuning, predictive maintenance, and automated recipe optimization across lithography, etch, and deposition modules.',
                duvTitle: 'DUV Energy Deposition: Monte Carlo vs Double Gaussian',
                duvDesc: 'Complete simulator comparing aerial image energy deposition predicted by Double-Gaussian PSF (FFT convolution) against Monte Carlo particle model. Features partial coherence modeling, flare analysis, swing curves, and comprehensive statistical validation with PDF report generation.',
                ganTitle: 'Vertical GaN Power Electronics',
                ganDesc: "Revolutionary 3D architecture for >1kV power devices. Vertical GaN FETs merge GaN's superior switching speed with SiC-class voltage handling, enabling next-generation EV drivetrains and utility-scale power converters.",
                ganDeviceTitle: 'GaN Power Device Characterization Study',
                ganDeviceDesc: 'Independent project. Study of thermal modeling for InGaN/GaN HEMTs covering 3D thermal analysis, electro-thermal coupling, and piezoelectric effects. Quantitative results on the detail page are reproduced from cited literature (NASA HEMT, Talukder et al, IJAER) using MATLAB, Sentaurus TCAD, and Python.',
                hubElectronics: 'Electronics',
                hubPhotonics: 'Photonics',
                hubMlds: 'ML & DS',
                hubInnovation: 'Innovation'
            },
            experience: {
                title: 'Experience & Education',
                expHeading: 'Experience',
                eduHeading: 'Education',
                learnMore: 'Learn More About Experience',
                catEducation: 'Education',
                catExperience: 'Experience',
                catMilitary: 'Military Service',
                learnCoursework: 'Learn More About Coursework',
                learnRole: 'Learn More About This Role',
                ebTitle: 'Quality Inspector / MT Inspector',
                ebSchool: 'General Dynamics Electric Boat - Quonset Point, RI',
                ebDate: 'Current',
                ebDesc: 'Quality and MT inspector on Virginia and Columbia class submarine programs under DoD controlled technology and NAVSEA standards.',
                asmlTitle: 'Senior Production Technician',
                asmlSchool: 'ASML US - Wilton, CT',
                asmlDate: '05/2021 - 2023',
                asmlDesc: 'Build, alignment, and diagnostic execution on production photolithography systems shipped to leading-edge semiconductor fabs. Owned high-precision sub-assembly qualification under tight contamination, ESD, and tolerance controls; partnered with engineering on 8D root cause for electromechanical and optical anomalies.',
                gradResTitle: 'Graduate Researcher / Equipment Design & Test Specialist',
                gradResSchool: 'University of Connecticut - Storrs, CT',
                gradResDate: '10/2021 - 04/2024',
                gradResDesc: 'Supported semiconductor and optoelectronics labs with precision measurements, front-end and back-end processing technologies. Provided guidance to undergraduate students on independent research projects.',
                armyTitle: 'Patient Administration Specialist (68G)',
                armySchool: 'United States Army (Active Duty) - Fort Stewart, GA',
                armyDesc: 'Managed medical records, patient admissions, and healthcare administration for over 3,000 active duty personnel. Coordinated MEDEVAC operations and maintained HIPAA compliance.',
                ugResTitle: 'Undergraduate Student Research Assistant',
                ugResSchool: 'University of Connecticut - Storrs, CT',
                ugResDate: '2020',
                ugResDesc: 'Conducted experimental and theoretical research on Electromagnetically Induced Transparency (EIT) in atomic systems, Monte Carlo simulations of molecular beams, and advanced quantum materials.',
                phdTitle: 'Ph.D. Program, Electrical Engineering',
                phdBadge: 'Not Completed',
                phdSchool: 'University of Connecticut - Storrs, CT',
                phdDate: '2023 - 2025',
                phdDesc: 'Pursued doctoral coursework and research in wide-bandgap semiconductor devices (AlGaN/GaN HEMTs); did not complete the program. Transitioned to and completed the M.S. in Electrical Engineering for industry entry.',
                msTitle: 'Master of Science, Electrical Engineering',
                msSchool: 'University of Connecticut - Storrs, CT',
                msDate: '05/2025',
                msDesc: 'Specialized in Electronics, Photonics, and Bio-Photonics. Applied ML methods to data analysis and experimental optimization. Research focus on GaN-based power devices, SiC MOSFETs, and Monte Carlo simulation for EUV lithography.',
                bsTitle: 'Bachelor of Science, Physics',
                bsSchool: 'University of Connecticut - Storrs, CT',
                bsDate: '2020',
                bsDesc: 'Graduated with focus on optics, quantum physics, and experimental research. Completed undergraduate research in EIT for slow light and quantum memory demonstrations.',
                aaTitle: 'Associate of Science, Engineering Science',
                aaSchool: 'CT State Community College Housatonic - Bridgeport',
                aaDesc: 'Foundation in engineering principles and mathematics, preparing for advanced studies in physics and electrical engineering.'
            },
            certifications: {
                title: 'Professional Certifications',
                viewCertificate: 'View Certificate',
                learnMore: 'Learn More About Certifications',
                badgeScheduled: 'Scheduled',
                badgeRegistered: 'Registered',
                placeholderScheduled: 'Scheduled',
                placeholderRegistered: 'Registered',
                patentBarTitle: 'USPTO Patent Bar Examination',
                patentBarDesc: 'Registered with the USPTO Office of Enrollment and Discipline to sit for the Registration Examination for Patent Attorneys and Agents.',
                feTitle: 'FE Electrical and Computer Engineering',
                feDesc: 'Fundamentals of Engineering exam: Electrical and Computer Engineering.',
                comptiaTitle: 'CompTIA Security+',
                comptiaDesc: 'Industry-standard cybersecurity certification.',
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
                aboutDesc: 'Electrical engineer focused on semiconductor manufacturing, device research, and quality and inspection work. Production-floor experience at ASML Wilton; current quality control NDT inspection at General Dynamics Electric Boat; graduate device research at UConn.',
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
            },
            chatbot: {
                bubble: 'Ask the AI Assistant'
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
                contact: 'Contact',
                search: 'Rechercher'
            },
            hero: {
                greeting: 'Bonjour, je suis',
                name: 'Louis Vladimir Antoine',
                subtitle: 'Ingénieur Électrique',
                desc1: "Ingénieur matériel travaillant sur la RF mmWave, les dispositifs GaN et les procédés semi-conducteurs. Intéressé par l'architecture, l'intégration système et l'analyse d'espaces de compromis.",
                desc2: "Vétéran de l'armée américaine. Inscrit à l'examen du barreau des brevets de l'USPTO.",
                cta: 'Voir les Projets',
                contact: 'Me Contacter',
                scroll: 'Faites défiler pour explorer'
            },
            about: {
                title: 'À Propos de Moi',
                description: "Ingénieur électrique axé sur la fabrication de semiconducteurs, la recherche sur les dispositifs et le travail de qualité et d'inspection. Expérience sur le plancher de production chez ASML Wilton ; contrôle qualité et essais non destructifs en cours chez General Dynamics Electric Boat ; recherche doctorale sur les dispositifs à UConn.",
                tabIntro: 'À Propos',
                tabIntroBack: 'Découvrir',
                tabWork: 'Mon Travail',
                tabWorkBack: 'Explorer',
                tabMission: 'Ma Mission',
                tabMissionBack: 'Vision',
                introHeading: 'Ingénieur Électrique',
                introBody: "Ingénieur électrique qui a appris comment se construisent les scanners de production avant d'apprendre comment fonctionnent réellement les dispositifs gravés sur les wafers qui en sortent. Le plus utile à l'intersection de l'équipement, du procédé et de la physique des dispositifs, là où une seule personne peut lire les données, exécuter le diagnostic, et en expliquer le sens.",
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
                spectraLabTitle: 'SPECTRA-Lab : Plateforme de Caractérisation Semiconducteur et MES',
                spectraLabDesc: "Projet indépendant, en développement actif. Plateforme de caractérisation semiconducteur et d'exécution de fabrication combinant un backend Python/FastAPI avec un frontend Next.js 14, persistance PostgreSQL et couche async Celery + Redis. Couvre la métrologie, sept domaines de contrôle de procédé, SPC avec règles Western Electric, ML et métrologie virtuelle, et un LIMS/ELN.",
                maxwellBlochTitle: 'Simulation Complète Maxwell-Bloch pour EIT',
                maxwellBlochDesc: "Cadre de simulation complet pour les expériences d'optique quantique basé sur les équations de Maxwell-Bloch, comprenant l'élargissement Doppler, le stockage/récupération d'impulsions et les effets géométriques.",
                semiOptTitle: 'Optimisation des Procédés Semiconducteurs',
                semiOptDesc: "Système avancé de contrôle de procédés piloté par ML, combinant la métrologie virtuelle basée sur CatBoost, des algorithmes de contrôle double EWMA et l'intégration SPC/FDC. Permet une amélioration du rendement de 15 à 25 % grâce à un réglage des paramètres en temps réel, une maintenance prédictive et une optimisation automatisée des recettes pour la lithographie, la gravure et le dépôt.",
                duvTitle: 'Dépôt d’Énergie DUV : Monte Carlo vs Double Gaussienne',
                duvDesc: "Simulateur complet comparant le dépôt d'énergie de l'image aérienne prédit par PSF Double-Gaussienne (convolution FFT) à un modèle particulaire Monte Carlo. Modélisation de la cohérence partielle, analyse du flare, courbes de swing et validation statistique complète avec génération de rapports PDF.",
                ganTitle: 'Électronique de Puissance GaN Verticale',
                ganDesc: "Architecture 3D révolutionnaire pour des dispositifs de puissance >1 kV. Les FET GaN verticaux combinent la vitesse de commutation supérieure du GaN avec la tenue en tension de classe SiC, permettant les chaînes de traction VE de nouvelle génération et les convertisseurs de puissance à grande échelle.",
                ganDeviceTitle: "Étude de Caractérisation de Dispositifs de Puissance GaN",
                ganDeviceDesc: "Projet indépendant. Étude de la modélisation thermique des HEMT InGaN/GaN couvrant l'analyse thermique 3D, le couplage électro-thermique et les effets piézoélectriques. Les résultats quantitatifs de la page détaillée sont reproduits à partir de la littérature citée (NASA HEMT, Talukder et al, IJAER) en utilisant MATLAB, Sentaurus TCAD et Python.",
                hubElectronics: 'Électronique',
                hubPhotonics: 'Photonique',
                hubMlds: 'ML & DS',
                hubInnovation: 'Innovation'
            },
            experience: {
                title: 'Expérience & Formation',
                expHeading: 'Expérience',
                eduHeading: 'Formation',
                learnMore: "En Savoir Plus sur l'Expérience",
                catEducation: 'Formation',
                catExperience: 'Expérience',
                catMilitary: 'Service Militaire',
                learnCoursework: 'En Savoir Plus sur le Programme',
                learnRole: 'En Savoir Plus sur ce Poste',
                ebTitle: "Inspecteur Qualité / Inspecteur MT",
                ebSchool: 'General Dynamics Electric Boat - Quonset Point, RI',
                ebDate: 'Actuel',
                ebDesc: 'Inspection qualité et MT sur les programmes de sous-marins des classes Virginia et Columbia, conformément aux protocoles de technologie contrôlée du DoD et aux normes NAVSEA.',
                asmlTitle: 'Technicien Senior de Production',
                asmlSchool: 'ASML US - Wilton, CT',
                asmlDate: '05/2021 - 2023',
                asmlDesc: "Construction, alignement et diagnostic de systèmes de photolithographie de production destinés aux fabs de semiconducteurs de pointe. Qualification haute précision de sous-ensembles sous contrôles stricts de contamination, ESD et tolérances ; collaboration avec l'ingénierie sur les analyses 8D des anomalies électromécaniques et optiques.",
                gradResTitle: "Chercheur Diplômé / Spécialiste en Conception et Essai d'Équipements",
                gradResSchool: 'University of Connecticut - Storrs, CT',
                gradResDate: '10/2021 - 04/2024',
                gradResDesc: "Soutien des laboratoires de semiconducteurs et d'optoélectronique avec des mesures de précision et des technologies de procédé front-end et back-end. Encadrement d'étudiants de premier cycle sur des projets de recherche indépendants.",
                armyTitle: 'Spécialiste en Administration des Patients (68G)',
                armySchool: 'Armée des États-Unis (Service Actif) - Fort Stewart, GA',
                armyDesc: "Gestion des dossiers médicaux, des admissions et de l'administration des soins pour plus de 3 000 militaires en service actif. Coordination des opérations MEDEVAC et conformité HIPAA.",
                ugResTitle: 'Assistant Étudiant de Recherche',
                ugResSchool: 'University of Connecticut - Storrs, CT',
                ugResDate: '2020',
                ugResDesc: "Recherches expérimentales et théoriques sur la Transparence Induite Électromagnétiquement (EIT) dans les systèmes atomiques, simulations Monte Carlo de faisceaux moléculaires et matériaux quantiques avancés.",
                phdTitle: 'Programme de Doctorat, Génie Électrique',
                phdBadge: 'Non Terminé',
                phdSchool: 'University of Connecticut - Storrs, CT',
                phdDate: '2023 - 2025',
                phdDesc: "Cours et recherches doctorales sur les dispositifs semiconducteurs à large bande interdite (HEMTs AlGaN/GaN) ; programme non terminé. Transition vers le Master en Génie Électrique, complété pour intégration en industrie.",
                msTitle: 'Master en Génie Électrique',
                msSchool: 'University of Connecticut - Storrs, CT',
                msDate: '05/2025',
                msDesc: "Spécialisation en électronique, photonique et bio-photonique. Application des méthodes de ML à l'analyse de données et à l'optimisation expérimentale. Recherche axée sur les dispositifs de puissance GaN, les MOSFET SiC et la simulation Monte Carlo pour la lithographie EUV.",
                bsTitle: 'Licence en Physique',
                bsSchool: 'University of Connecticut - Storrs, CT',
                bsDate: '2020',
                bsDesc: "Diplômé avec un focus sur l'optique, la physique quantique et la recherche expérimentale. Réalisation de recherches en EIT pour la lumière lente et démonstrations de mémoire quantique.",
                aaTitle: "Diplôme Associé en Sciences de l'Ingénieur",
                aaSchool: 'CT State Community College Housatonic - Bridgeport',
                aaDesc: "Bases en principes d'ingénierie et mathématiques, en préparation d'études avancées en physique et génie électrique."
            },
            certifications: {
                title: 'Certifications Professionnelles',
                viewCertificate: 'Voir le Certificat',
                learnMore: 'En Savoir Plus sur les Certifications',
                badgeScheduled: 'Programmé',
                badgeRegistered: 'Inscrit',
                placeholderScheduled: 'Programmé',
                placeholderRegistered: 'Inscrit',
                patentBarTitle: "Examen du Barreau des Brevets de l'USPTO",
                patentBarDesc: "Inscrit auprès de l'USPTO Office of Enrollment and Discipline pour passer le Registration Examination for Patent Attorneys and Agents.",
                feTitle: 'FE Génie Électrique et Informatique',
                feDesc: "Examen Fundamentals of Engineering : Génie Électrique et Informatique.",
                comptiaTitle: 'CompTIA Security+',
                comptiaDesc: 'Certification standard du secteur en cybersécurité.',
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
                aboutDesc: "Ingénieur électrique axé sur la fabrication de semiconducteurs, la recherche sur les dispositifs et le travail de qualité et d'inspection. Expérience sur le plancher de production chez ASML Wilton ; contrôle qualité et essais non destructifs en cours chez General Dynamics Electric Boat ; recherche doctorale sur les dispositifs à UConn.",
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
            },
            chatbot: {
                bubble: "Demander à l'assistant IA"
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
                contact: 'Contacto',
                search: 'Buscar'
            },
            hero: {
                greeting: 'Hola, soy',
                name: 'Louis Vladimir Antoine',
                subtitle: 'Ingeniero Eléctrico',
                desc1: 'Ingeniero de hardware que trabaja en RF mmWave, dispositivos GaN y procesos de semiconductores. Interesado en arquitectura, integración de sistemas y análisis de espacios de compromiso.',
                desc2: 'Veterano del Ejército de EE. UU. Inscrito en el examen del colegio de patentes de la USPTO.',
                cta: 'Ver Proyectos',
                contact: 'Contáctame',
                scroll: 'Desplázate para explorar'
            },
            about: {
                title: 'Sobre Mí',
                description: 'Ingeniero eléctrico enfocado en fabricación de semiconductores, investigación de dispositivos y trabajo de calidad e inspección. Experiencia en planta de producción en ASML Wilton; control de calidad y END en curso en General Dynamics Electric Boat; investigación doctoral de dispositivos en UConn.',
                tabIntro: 'Sobre Mí',
                tabIntroBack: 'Descubrir',
                tabWork: 'Lo Que Hago',
                tabWorkBack: 'Explorar',
                tabMission: 'Mi Misión',
                tabMissionBack: 'Visión',
                introHeading: 'Ingeniero Eléctrico',
                introBody: 'Ingeniero eléctrico que aprendió cómo se construyen los escáneres de producción antes de aprender cómo funcionan realmente los dispositivos en las obleas que salen de ellos. Más útil en la intersección de equipo, proceso y física de dispositivos, donde una misma persona puede leer los datos, ejecutar el diagnóstico y explicar lo que significan.',
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
                spectraLabTitle: 'SPECTRA-Lab: Plataforma de Caracterización de Semiconductores y MES',
                spectraLabDesc: 'Proyecto independiente, en desarrollo activo. Plataforma de caracterización de semiconductores y ejecución de fabricación que combina un backend Python/FastAPI con un frontend Next.js 14, persistencia PostgreSQL y una capa async Celery + Redis. Cubre metrología, siete áreas de control de procesos, SPC con reglas de Western Electric, ML y metrología virtual, y un LIMS/ELN.',
                maxwellBlochTitle: 'Simulación Completa Maxwell-Bloch para EIT',
                maxwellBlochDesc: 'Marco de simulación completo para experimentos de óptica cuántica basado en las ecuaciones de Maxwell-Bloch, con ensanchamiento Doppler, almacenamiento/recuperación de pulsos y efectos geométricos.',
                semiOptTitle: 'Optimización de Procesos de Semiconductores',
                semiOptDesc: 'Sistema avanzado de control de procesos impulsado por ML que combina metrología virtual basada en CatBoost, algoritmos de control EWMA doble e integración SPC/FDC. Logra una mejora del 15-25 % en el rendimiento mediante ajuste de parámetros en tiempo real, mantenimiento predictivo y optimización automática de recetas en litografía, grabado y deposición.',
                duvTitle: 'Deposición de Energía DUV: Monte Carlo vs Doble Gaussiana',
                duvDesc: 'Simulador completo que compara la deposición de energía de la imagen aérea predicha por PSF Doble-Gaussiana (convolución FFT) con un modelo particulado Monte Carlo. Incluye modelado de coherencia parcial, análisis de flare, curvas de swing y validación estadística completa con generación de informes PDF.',
                ganTitle: 'Electrónica de Potencia GaN Vertical',
                ganDesc: 'Arquitectura 3D revolucionaria para dispositivos de potencia >1 kV. Los FET GaN verticales combinan la velocidad de conmutación superior del GaN con el manejo de tensión clase SiC, habilitando transmisiones EV de próxima generación y convertidores de potencia a escala de servicios públicos.',
                ganDeviceTitle: 'Estudio de Caracterización de Dispositivos de Potencia GaN',
                ganDeviceDesc: 'Proyecto independiente. Estudio de modelado térmico de HEMT InGaN/GaN que cubre análisis térmico 3D, acoplamiento electrotérmico y efectos piezoeléctricos. Los resultados cuantitativos en la página de detalle se reproducen a partir de literatura citada (NASA HEMT, Talukder et al, IJAER) utilizando MATLAB, Sentaurus TCAD y Python.',
                hubElectronics: 'Electrónica',
                hubPhotonics: 'Fotónica',
                hubMlds: 'ML y DS',
                hubInnovation: 'Innovación'
            },
            experience: {
                title: 'Experiencia y Educación',
                expHeading: 'Experiencia',
                eduHeading: 'Educación',
                learnMore: 'Más Información sobre Experiencia',
                catEducation: 'Educación',
                catExperience: 'Experiencia',
                catMilitary: 'Servicio Militar',
                learnCoursework: 'Más Información sobre el Plan de Estudios',
                learnRole: 'Más Información sobre este Rol',
                ebTitle: 'Inspector de Calidad / Inspector MT',
                ebSchool: 'General Dynamics Electric Boat - Quonset Point, RI',
                ebDate: 'Actual',
                ebDesc: 'Inspección de calidad y MT en los programas de submarinos clase Virginia y Columbia, bajo protocolos de tecnología controlada del DoD y normas NAVSEA.',
                asmlTitle: 'Técnico Senior de Producción',
                asmlSchool: 'ASML US - Wilton, CT',
                asmlDate: '05/2021 - 2023',
                asmlDesc: 'Construcción, alineación y diagnóstico de sistemas de fotolitografía de producción enviados a fabs de semiconductores de vanguardia. Calificación de subensambles de alta precisión bajo estrictos controles de contaminación, ESD y tolerancias; colaboración con ingeniería en análisis 8D de causa raíz para anomalías electromecánicas y ópticas.',
                gradResTitle: 'Investigador de Posgrado / Especialista en Diseño y Pruebas de Equipos',
                gradResSchool: 'University of Connecticut - Storrs, CT',
                gradResDate: '10/2021 - 04/2024',
                gradResDesc: 'Apoyo a laboratorios de semiconductores y optoelectrónica con mediciones de precisión y tecnologías de procesamiento de front-end y back-end. Orientación a estudiantes de pregrado en proyectos de investigación independientes.',
                armyTitle: 'Especialista en Administración de Pacientes (68G)',
                armySchool: 'Ejército de los Estados Unidos (Servicio Activo) - Fort Stewart, GA',
                armyDesc: 'Gestión de registros médicos, admisiones de pacientes y administración de salud para más de 3.000 efectivos en servicio activo. Coordinación de operaciones MEDEVAC y cumplimiento de HIPAA.',
                ugResTitle: 'Asistente Estudiantil de Investigación de Pregrado',
                ugResSchool: 'University of Connecticut - Storrs, CT',
                ugResDate: '2020',
                ugResDesc: 'Investigación experimental y teórica sobre Transparencia Inducida Electromagnéticamente (EIT) en sistemas atómicos, simulaciones Monte Carlo de haces moleculares y materiales cuánticos avanzados.',
                phdTitle: 'Programa de Doctorado, Ingeniería Eléctrica',
                phdBadge: 'No Completado',
                phdSchool: 'University of Connecticut - Storrs, CT',
                phdDate: '2023 - 2025',
                phdDesc: 'Cursos doctorales e investigación en dispositivos semiconductores de banda ancha (HEMTs AlGaN/GaN); programa no completado. Transición al Máster en Ingeniería Eléctrica, completado para ingreso a la industria.',
                msTitle: 'Máster en Ingeniería Eléctrica',
                msSchool: 'University of Connecticut - Storrs, CT',
                msDate: '05/2025',
                msDesc: 'Especialización en Electrónica, Fotónica y Bio-Fotónica. Aplicación de métodos de ML al análisis de datos y optimización experimental. Investigación enfocada en dispositivos de potencia GaN, MOSFET de SiC y simulación Monte Carlo para litografía EUV.',
                bsTitle: 'Licenciatura en Física',
                bsSchool: 'University of Connecticut - Storrs, CT',
                bsDate: '2020',
                bsDesc: 'Graduado con enfoque en óptica, física cuántica e investigación experimental. Realización de investigación de pregrado en EIT para luz lenta y demostraciones de memoria cuántica.',
                aaTitle: 'Asociado en Ciencias, Ciencias de la Ingeniería',
                aaSchool: 'CT State Community College Housatonic - Bridgeport',
                aaDesc: 'Fundamentos en principios de ingeniería y matemáticas, preparación para estudios avanzados en física e ingeniería eléctrica.'
            },
            certifications: {
                title: 'Certificaciones Profesionales',
                viewCertificate: 'Ver Certificado',
                learnMore: 'Más Información sobre Certificaciones',
                badgeScheduled: 'Programado',
                badgeRegistered: 'Inscrito',
                placeholderScheduled: 'Programado',
                placeholderRegistered: 'Inscrito',
                patentBarTitle: 'Examen del Colegio de Patentes de la USPTO',
                patentBarDesc: 'Inscrito en la USPTO Office of Enrollment and Discipline para presentar el Registration Examination for Patent Attorneys and Agents.',
                feTitle: 'FE Ingeniería Eléctrica e Informática',
                feDesc: 'Examen Fundamentals of Engineering: Ingeniería Eléctrica e Informática.',
                comptiaTitle: 'CompTIA Security+',
                comptiaDesc: 'Certificación estándar de la industria en ciberseguridad.',
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
                aboutDesc: 'Ingeniero eléctrico enfocado en fabricación de semiconductores, investigación de dispositivos y trabajo de calidad e inspección. Experiencia en planta de producción en ASML Wilton; control de calidad y END en curso en General Dynamics Electric Boat; investigación doctoral de dispositivos en UConn.',
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
            },
            chatbot: {
                bubble: 'Pregunta al asistente de IA'
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
