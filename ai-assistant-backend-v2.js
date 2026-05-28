// AI Assistant Backend V2 - Enhanced with Real Conversations
// Persistent, unlimited conversations with dynamic responses

class AIAssistantBackendV2 {
    constructor() {
        this.conversations = this.loadAllConversations();
        this.activeConversation = null;
        this.knowledgeBase = this.initializeKnowledgeBase();
        this.contextWindow = 20; // Keep last 20 messages for context
        this.initializeResponseEngine();
    }

    // Initialize enhanced response engine
    initializeResponseEngine() {
        this.responsePatterns = {
            // Dynamic response templates that can be combined
            greetings: [
                "Hello! How can I help you explore the portfolio today?",
                "Hi there! What would you like to know about?",
                "Welcome back! Ready to continue our conversation?",
                "Great to see you! What interests you today?"
            ],
            
            acknowledgments: [
                "I understand you're asking about",
                "Let me help you with",
                "That's an interesting question about",
                "I can definitely explain"
            ],
            
            transitions: [
                "Speaking of which,",
                "On a related note,",
                "That reminds me,",
                "Additionally,",
                "Furthermore,"
            ],
            
            clarifications: [
                "Could you tell me more about what you mean by",
                "Are you specifically interested in",
                "Would you like me to focus on",
                "Do you mean"
            ]
        };
    }

    // Initialize comprehensive knowledge base
    initializeKnowledgeBase() {
        // Knowledge base sourced from Louis Antoine's actual resume
        // (assets/pdfs/ElectricalEngineeringresume2026.pdf) and the homepage
        // timeline/skills sections. Do not add fabricated content here —
        // if a fact isn't on the resume or the site, it doesn't belong.

        return {
            // Personal contact info — matches the homepage and resume.
            personal: {
                name: "Louis Vladimir Antoine",
                title: "Electrical Engineer",
                location: "New Haven, CT 06515",
                email: "alovladi@gmail.com",
                phone: "(203) 360-5619",
                portfolio: "https://alovladi007.github.io/louis-antoine-portfolio/",
                github: "https://github.com/alovladi007",
                linkedin: "https://www.linkedin.com/in/louis-antoine-333199a0",
                summary: "Hardware engineer across mmWave RF, GaN devices, and semiconductor process. Interested in architecture, system integration, and trade space work. Production-floor experience at ASML Wilton (Senior Production Technician on the YieldStar optical metrology platform). Current quality control NDT inspection at General Dynamics Electric Boat on Virginia and Columbia class submarine programs. Graduate device research at UConn on AlGaN/GaN HEMT electrothermal TCAD modeling. US Army veteran. Scheduled for the FE Electrical and Computer Engineering exam."
            },

            // Featured projects — these are real pages on the portfolio site.
            // Specs that aren't backed by the resume or a real project page
            // have been removed; only the title, category, summary, and the
            // canonical URL remain so the assistant can route users there.
            projects: {
                "mmwave-rf": {
                    title: "mmWave RF Frontend Design",
                    category: "Communications & RF",
                    description: "28 GHz 5G NR (n257 band) phased-array RF frontend study with beamforming, GaN PA, and LNA design.",
                    url: "projects/comms/mmwave-rf-complete.html",
                    tools_used: ["Keysight ADS", "ANSYS HFSS", "Cadence", "MATLAB", "Python"]
                },

                "riscv-soc": {
                    title: "RISC-V SoC with Custom Accelerators",
                    category: "Power Electronics & Hardware",
                    description: "64-bit RISC-V processor with ML and crypto accelerators on a 2D mesh NoC; Verilog/SystemVerilog implementation with UVM verification flow.",
                    url: "projects/power-electronics/riscv-soc-complete.html",
                    tools_used: ["Verilog", "SystemVerilog", "UVM", "Vivado", "ModelSim"]
                },

                "cryptography": {
                    title: "Cryptography Research",
                    category: "Research",
                    description: "Survey and implementation of post-quantum cryptography algorithms (CRYSTALS-Kyber, Dilithium, FALCON, SPHINCS+) with hardware acceleration.",
                    url: "research/cryptography-research.html",
                    tools_used: ["Python", "C", "Verilog", "FPGA"]
                },

                "pcm": {
                    title: "Phase-Change Memory (GST) Research",
                    category: "Semiconductor Research",
                    description: "Ge₂Sb₂Te₅ (GST) phase-change memory device modeling, materials engineering, and circuit/array design.",
                    url: "projects/semiconductor/pcm-complete-project.html",
                    tools_used: ["Python", "MATLAB", "Device simulation"]
                },

                "duv-energy-deposition": {
                    title: "DUV Energy Deposition: Monte Carlo vs Double Gaussian",
                    category: "Semiconductor Research",
                    description: "Compares aerial-image energy deposition predicted by a Double-Gaussian PSF against a Monte Carlo particle model. Includes partial-coherence modeling, flare analysis, swing curves, and statistical validation with PDF report generation.",
                    url: "projects/semiconductor/duv-energy-deposition-project.html",
                    tools_used: ["Python", "MATLAB", "Monte Carlo simulation"]
                },

                "quantum-memory-eit": {
                    title: "EIT Computational Study: Slow Light and Quantum Memory in 87Rb",
                    category: "Quantum (undergraduate)",
                    description: "Undergraduate computational study of Electromagnetically Induced Transparency (EIT) in 87Rb-style atomic systems, reproducing slow-light and quantum memory phenomena from the published literature.",
                    url: "projects/quantum/quantum-memory-project.html",
                    tools_used: ["Python", "MATLAB"]
                },

                "maxwell-bloch-eit": {
                    title: "Full Maxwell-Bloch Simulation for EIT",
                    category: "Quantum",
                    description: "Simulation framework for quantum-optics experiments based on Maxwell-Bloch equations: Doppler broadening, pulse storage and retrieval, geometry effects.",
                    url: "research/maxwell-bloch-eit-project.html",
                    tools_used: ["Python", "Density-matrix formalism"]
                },

                "vertical-gan": {
                    title: "Vertical GaN Power Electronics",
                    category: "Power Electronics",
                    description: "Vertical-architecture GaN FETs for >1 kV power devices, combining GaN switching speed with SiC-class voltage handling for EV drivetrains and utility-scale converters.",
                    url: "demos/vertical-gan-project.html",
                    tools_used: ["MATLAB", "Sentaurus / TCAD", "Python"]
                },

                "silicon-photonics-ring": {
                    title: "Silicon Photonics — Microring Resonator WDM Filter",
                    category: "Photonics",
                    description: "Design and FDTD simulation of a silicon microring resonator WDM filter at 1550 nm. Q ≈ 1950, ER ≥ 20 dB, FSR ≈ 100 GHz.",
                    url: "projects/semiconductor/silicon-photonics-project.html",
                    tools_used: ["Lumerical FDTD", "Python", "PIC layout tools"]
                },

                "semi-process-optimization": {
                    title: "Semiconductor Process Optimization",
                    category: "Semiconductor Process",
                    description: "ML-driven process control combining CatBoost virtual metrology, double-EWMA control, and SPC/FDC integration across lithography, etch, and deposition modules.",
                    url: "projects/semiconductor/semiconductor-optimization-complete-suite.html",
                    tools_used: ["Python", "CatBoost", "EWMA", "SPC/FDC"]
                }
            },

            // Skills — adapted from the resume's "Skills" section and the
            // homepage Skills section. Programming list is intentionally
            // small and honest; tooling and domains reflect actual ASML +
            // UConn work.
            skills: {
                programming: {
                    proficient: ["Python", "MATLAB", "Bash"],
                    familiar:   ["JavaScript", "C", "TCL", "Verilog", "SystemVerilog"]
                },

                semiconductor: {
                    lithography: [
                        "Photolithography process development",
                        "EUV / DUV tool operation and calibration",
                        "Optical Proximity Correction (OPC)",
                        "Resolution Enhancement Techniques (RET)",
                        "Reticle and pellicle management (binary masks, PSM)",
                        "Immersion lithography",
                        "Overlay metrology"
                    ],
                    deposition_etch: [
                        "Thin-film deposition: PVD, CVD, LPCVD, MOCVD, PECVD, ALD, MBE",
                        "Crystal growth, dopant diffusion, oxidation kinetics",
                        "Wet etching, dry/plasma etching, atomic layer etching",
                        "Etchback process using photoresist and spin-on-glass (SOG)",
                        "CMP (Applied Materials Reflection)"
                    ],
                    metrology: [
                        "Ellipsometry, XRR/XRD, profilometry",
                        "CD-SEM, AFM, scatterometry",
                        "Diffraction-based overlay metrology (YieldStar)",
                        "Line-edge roughness (LER), swing curves, focus-exposure data",
                        "I-V, C-V, mobility, sheet-resistance correlations"
                    ],
                    process_control: [
                        "SPC (control charts, Cp/Cpk)",
                        "FDC (Fault Detection and Classification)",
                        "DOE (Design of Experiments)",
                        "FMEA, 5 Whys, Pareto analysis",
                        "Tool qualification (install, baseline, spec check, system readiness)",
                        "Chamber matching and recipe optimization"
                    ]
                },

                systems_engineering: [
                    "Requirements decomposition and traceability",
                    "CONOPS / use cases",
                    "Interface Control Documents (ICDs)",
                    "Verification Cross-Reference Matrix (VCRM)",
                    "Verification & Validation (V&V) execution",
                    "Trade studies, risk and issues management",
                    "Configuration control and documentation"
                ],

                analysis: [
                    "Image processing and analysis (MATLAB, Python, OpenCV, scikit-image)",
                    "Monte Carlo simulation (electron scattering / energy deposition)",
                    "Density-matrix formalism for three-level Λ systems",
                    "Statistical data analysis"
                ],

                tools: {
                    rf:        ["Keysight ADS", "ANSYS HFSS", "CST Studio"],
                    fpga_eda:  ["Vivado", "Quartus", "ModelSim", "Cadence Virtuoso"],
                    cleanroom: ["YieldStar 375 F / 380 G / 1385", "AMAT Reflection CMP"],
                    software:  ["Git", "Docker", "TensorFlow", "PyTorch", "SAP", "Excel"],
                    lab:       ["Oscilloscopes", "Multimeters", "Power probes", "Spectrometers", "Power meters"]
                }
            },

            // Experience — verbatim from the resume.
            experience: {
                current: null, // No current full-time role on the resume.
                previous: [
                    {
                        position: "Student Equipment Design & Test Specialist",
                        company:  "University of Connecticut",
                        location: "Storrs, CT",
                        duration: "10/2021 – 04/2024",
                        highlights: [
                            "Modeling and evaluation of GaN-based power devices for high-frequency switching applications",
                            "Detailed SiC MOSFET models for harsh, high-voltage environments",
                            "Monte Carlo simulation of electron scattering and energy deposition in EUV masks; outperformed analytical proximity-correction models on 3D absorber structures",
                            "Supported semiconductor and optoelectronics labs with precision measurements (front-end and back-end interconnect / dual-Damascene)",
                            "Lithography (optical pattern formation, photoresists, wafer steppers, scanners, immersion, EUV)",
                            "Etching and CMP (wet, dry/plasma, atomic-layer; familiarity with AMAT Reflection CMP)",
                            "Deposition (CVD, ELO, LPCVD, MOCVD, PECVD, ALD, MBE)",
                            "Mentored undergraduates on independent research projects"
                        ]
                    },
                    {
                        position: "Optical Metrology Equipment Operator",
                        company:  "ASML US",
                        location: "Wilton, CT",
                        duration: "05/2021 – 06/2023",
                        highlights: [
                            "Assembled, installed, and tested an optical metrology system using torque wrenches, electric and pneumatic screwdrivers, ball drivers, oscilloscopes, voltage and current probes, multimeters, micrometers, and calipers",
                            "Worked with optical fibers, spectrometers, and power meters for analysis",
                            "YieldStar 375 F, 380 G, and 1385 systems for lithography process monitoring, control, system stability, and matching",
                            "Sub-system assembly, optical alignment, functional testing, equipment calibration, station setup, and process-development support",
                            "Sensor / UIA alignment, focus branch, Z-stage installation, pre- and final qualifications, Pupil / Alignment / Reference branches, cables and covers",
                            "Final qualifications: sensor performance, calibrations, branch-optic alignment, objective-lens performance, diffraction-based overlay measurements",
                            "Excel-based KPI tracking",
                            "Cleanroom environment with strict safety procedures (PPE, ASML 5S+1) — 12-hour night shifts in coveralls, hoods, booties, safety glasses, gloves",
                            "Root-cause analysis on production issues with corrective actions tracked in SAP, Opti Angle, and YieldStar software",
                            "Cross-functional collaboration with manufacturing, quality, and procurement"
                        ]
                    },
                    {
                        position: "Undergraduate Student Research Assistant",
                        company:  "University of Connecticut",
                        location: "Storrs, CT",
                        duration: "05/2020 – 08/2020",
                        highlights: [
                            "Used EIT for slow-light and quantum-memory demonstrations in vapor cells and cold-atom ensembles",
                            "Modeled Λ-type three-level systems using density-matrix formalism to simulate optical susceptibility under varying laser detuning and intensity",
                            "Developed laser-locking and modulation systems for EIT experiments using AOMs and EOMs"
                        ]
                    },
                    {
                        position: "Patient Administration Specialist",
                        company:  "United States Army",
                        location: "Fort Stewart, GA",
                        duration: "01/2016 – 01/2018",
                        highlights: [
                            "Used the computerized Resource and Patient Management System (RPMS) and Electronic Health Record (EHR) system to update patient records, transmit prescriptions, and transfer files",
                            "Maintained strict patient-data procedures to comply with HIPAA and prevent information breaches",
                            "Hands-on experience with MHS GENESIS, the DoD's enterprise EHR system",
                            "Supported patient registration, records management, data accuracy, and system navigation in a highly regulated healthcare-IT environment"
                        ]
                    },
                    {
                        position: "Cashier",
                        company:  "Walmart",
                        location: "Norwalk, CT",
                        duration: "10/2011 – 12/2015",
                        highlights: [
                            "Processed customer transactions accurately and efficiently",
                            "Assisted customers with product inquiries"
                        ]
                    }
                ],
                additional: [
                    {
                        position: "Professional Driver (Uber/Lyft)",
                        company:  "Self-employed",
                        location: "Storrs, CT",
                        duration: "While completing undergraduate degree",
                        highlights: [
                            "Provided professional transportation services while completing undergraduate degree"
                        ]
                    }
                ]
            },

            // Education — matches the resume; degrees only, no fabricated GPAs or honors.
            education: {
                graduate: {
                    degree: "Master of Science, Electrical Engineering",
                    school: "University of Connecticut, Storrs, CT",
                    completed: "05/2025",
                    notes: "Specialized in Electronics, Photonics, and Bio-Photonics. Applied ML methods to data analysis and experimental optimization. Research focus on GaN-based power devices, SiC MOSFETs, and Monte Carlo simulation for EUV lithography."
                },
                undergraduate: {
                    degree: "Bachelor of Science, Physics",
                    school: "University of Connecticut, Storrs, CT",
                    notes: "Focus on optics, quantum physics, and experimental research. Undergraduate research in EIT for slow-light and quantum-memory demonstrations."
                },
                associate: {
                    degree: "Associate of Science, Engineering Science",
                    school: "CT State Community College Housatonic, Bridgeport, CT",
                    notes: "Foundation in engineering principles and mathematics, preparing for advanced studies in physics and electrical engineering."
                }
            },

            // Certifications — split into earned (have a PDF) and scheduled
            // (sitting for the exam, mirrors the homepage's "Scheduled" badges).
            certifications: {
                earned: [
                    { name: "Six Sigma Green Belt",                                  pdf: "assets/pdfs/6Sigma Green Belt.pdf" },
                    { name: "Google IT Automation with Python",                      pdf: "assets/pdfs/Coursera IT Automation with Python.pdf" },
                    { name: "Google Advanced Data Analytics Professional Certificate", pdf: "assets/pdfs/Google Advanced Data Analytics.pdf" },
                    { name: "MATLAB Programming for Engineers and Scientists Specialization" }
                ],
                scheduled: [
                    { name: "USPTO Patent Bar" },
                    { name: "FE Electrical Engineering" },
                    { name: "CompTIA Security+" }
                ]
            },

            // Military — separate so the assistant can answer "did you serve?" cleanly.
            military: {
                branch: "United States Army",
                role:   "Patient Administration Specialist (68G)",
                base:   "Fort Stewart, GA",
                duration: "01/2016 – 01/2018"
            }
        };
    }

    // Load all conversations from storage
    loadAllConversations() {
        const saved = localStorage.getItem('aiAssistantConversations');
        if (saved) {
            const conversations = JSON.parse(saved);
            // Convert date strings back to Date objects
            Object.keys(conversations).forEach(id => {
                conversations[id].messages.forEach(msg => {
                    msg.timestamp = new Date(msg.timestamp);
                });
                conversations[id].created = new Date(conversations[id].created);
                conversations[id].lastModified = new Date(conversations[id].lastModified);
            });
            return conversations;
        }
        return {};
    }

    // Save all conversations
    saveAllConversations() {
        localStorage.setItem('aiAssistantConversations', JSON.stringify(this.conversations));
    }

    // Create new conversation
    createConversation(userId) {
        const conversationId = `conv_${userId}_${Date.now()}`;
        this.conversations[conversationId] = {
            id: conversationId,
            userId: userId,
            title: "New Conversation",
            messages: [],
            context: {
                topics: [],
                entities: new Set(),
                userPreferences: {},
                conversationFlow: []
            },
            created: new Date(),
            lastModified: new Date(),
            messageCount: 0
        };
        
        this.activeConversation = conversationId;
        this.saveAllConversations();
        return conversationId;
    }

    // Load existing conversation
    loadConversation(conversationId) {
        if (this.conversations[conversationId]) {
            this.activeConversation = conversationId;
            return this.conversations[conversationId];
        }
        return null;
    }

    // Get or create conversation for user
    getOrCreateConversation(userId) {
        // Find most recent conversation for user
        const userConversations = Object.values(this.conversations)
            .filter(c => c.userId === userId)
            .sort((a, b) => b.lastModified - a.lastModified);
        
        if (userConversations.length > 0) {
            // Continue most recent conversation
            this.activeConversation = userConversations[0].id;
            return userConversations[0].id;
        } else {
            // Create new conversation
            return this.createConversation(userId);
        }
    }

    // Process message with full context
    async processMessage(userId, message, conversationId = null) {
        // Get or create conversation
        if (!conversationId) {
            conversationId = this.getOrCreateConversation(userId);
        }
        
        const conversation = this.conversations[conversationId];
        if (!conversation) {
            throw new Error('Conversation not found');
        }

        // Add user message
        const userMessage = {
            role: 'user',
            content: message,
            timestamp: new Date()
        };
        conversation.messages.push(userMessage);
        conversation.messageCount++;

        // Get conversation context
        const context = this.buildContext(conversation);
        
        // Generate response based on full context
        const response = await this.generateDynamicResponse(message, context, conversation);
        
        // Add assistant response
        const assistantMessage = {
            role: 'assistant',
            content: response.content,
            timestamp: new Date(),
            metadata: response.metadata
        };
        conversation.messages.push(assistantMessage);
        conversation.messageCount++;

        // Update conversation metadata
        conversation.lastModified = new Date();
        if (conversation.messages.length === 2) {
            // First exchange - set title based on topic
            conversation.title = this.generateConversationTitle(message);
        }

        // Update context
        this.updateConversationContext(conversation, message, response);

        // Save to storage
        this.saveAllConversations();

        return {
            conversationId: conversationId,
            response: response,
            conversation: conversation
        };
    }

    // Build context from conversation history
    buildContext(conversation) {
        const recentMessages = conversation.messages.slice(-this.contextWindow);
        
        return {
            messageHistory: recentMessages,
            topics: conversation.context.topics,
            entities: Array.from(conversation.context.entities || []),
            messageCount: conversation.messageCount,
            conversationDuration: new Date() - conversation.created,
            userPreferences: conversation.context.userPreferences,
            lastTopics: this.extractRecentTopics(recentMessages)
        };
    }

    // Generate truly dynamic response based on context
    async generateDynamicResponse(message, context, conversation) {
        const analysis = this.analyzeMessage(message);
        const relevantKnowledge = this.retrieveRelevantKnowledge(message, analysis, context);
        
        // Build response based on multiple factors
        let responseContent = '';
        let metadata = {
            intent: analysis.intent,
            confidence: analysis.confidence,
            entities: analysis.entities,
            suggestions: [],
            code: null,
            links: []
        };

        // Check if this is a follow-up question
        const isFollowUp = this.isFollowUpQuestion(message, context);
        
        // Generate contextual response
        if (isFollowUp && context.messageHistory.length > 2) {
            // Continue previous topic with context
            responseContent = this.generateFollowUpResponse(message, context, relevantKnowledge);
        } else {
            // New topic or direct question
            responseContent = this.generateTopicResponse(message, analysis, relevantKnowledge);
        }

        // Add code examples if relevant
        if (this.shouldIncludeCode(message, analysis)) {
            metadata.code = this.selectRelevantCode(message, analysis);
        }

        // Add links if relevant
        if (this.shouldIncludeLinks(message, analysis)) {
            metadata.links = this.selectRelevantLinks(message, analysis);
        }

        // Generate smart suggestions based on conversation flow
        metadata.suggestions = this.generateContextualSuggestions(message, context, analysis);

        return {
            content: responseContent,
            metadata: metadata
        };
    }

    // Analyze message for intent and entities
    analyzeMessage(message) {
        const lower = message.toLowerCase();
        const words = lower.split(/\s+/);
        
        // Enhanced intent detection
        const intents = {
            project_inquiry: /tell me about|explain|describe|what is the|show me the/i,
            technical_question: /how does|how to|implement|design|architecture|technical/i,
            comparison: /compare|versus|vs|difference|better|choose between/i,
            code_request: /code|example|snippet|implement|function|algorithm/i,
            skills_inquiry: /skills|experience|proficient|know|languages|tools/i,
            contact: /contact|email|hire|freelance|collaborate|reach/i,
            help: /help|assist|guide|what can you|how can I/i,
            clarification: /what do you mean|clarify|elaborate|more about/i,
            continuation: /continue|go on|tell me more|and then|what else/i
        };

        let detectedIntent = 'general';
        let confidence = 0;

        for (const [intent, pattern] of Object.entries(intents)) {
            if (pattern.test(message)) {
                detectedIntent = intent;
                confidence = 0.8;
                break;
            }
        }

        // Extract entities
        const entities = this.extractEntities(message);

        // Extract key phrases
        const keyPhrases = this.extractKeyPhrases(message);

        return {
            intent: detectedIntent,
            confidence: confidence,
            entities: entities,
            keyPhrases: keyPhrases,
            sentiment: this.analyzeSentiment(message),
            isQuestion: message.includes('?'),
            wordCount: words.length
        };
    }

    // Extract entities from message
    extractEntities(message) {
        const lower = message.toLowerCase();
        const entities = {
            projects: [],
            technologies: [],
            concepts: [],
            actions: []
        };

        // Check for project mentions
        Object.entries(this.knowledgeBase.projects).forEach(([key, project]) => {
            const titleLower = (project.title || '').toLowerCase();
            const descLower  = (project.description || '').toLowerCase();
            if (lower.includes(key.replace('-', ' ')) ||
                (titleLower && lower.includes(titleLower)) ||
                (descLower && descLower.split(' ').some(word => word.length > 4 && lower.includes(word)))) {
                entities.projects.push(key);
            }
        });

        // Check for technology mentions
        const allTechs = [
            ...(this.knowledgeBase.skills.programming.proficient || []),
            ...(this.knowledgeBase.skills.programming.familiar  || []),
            ...Object.values(this.knowledgeBase.skills.tools || {}).flat()
        ];

        allTechs.forEach(tech => {
            if (lower.includes(tech.toLowerCase())) {
                entities.technologies.push(tech);
            }
        });

        // Extract concepts
        const concepts = ['beamforming', 'pipeline', 'cache', 'cryptography', 'memory', 'rf', 'fpga', 'asic'];
        concepts.forEach(concept => {
            if (lower.includes(concept)) {
                entities.concepts.push(concept);
            }
        });

        return entities;
    }

    // Extract key phrases
    extractKeyPhrases(message) {
        // Simple key phrase extraction
        const phrases = [];
        const patterns = [
            /(?:how|what|when|where|why|who)\s+(?:\w+\s+){0,3}\w+/gi,
            /(?:can|could|would|should)\s+(?:\w+\s+){0,3}\w+/gi,
            /\b(?:\w+\s+){1,3}(?:project|design|implementation|algorithm|system)\b/gi
        ];

        patterns.forEach(pattern => {
            const matches = message.match(pattern);
            if (matches) {
                phrases.push(...matches);
            }
        });

        return [...new Set(phrases)]; // Remove duplicates
    }

    // Analyze sentiment
    analyzeSentiment(message) {
        const positive = ['good', 'great', 'excellent', 'love', 'amazing', 'interested', 'exciting'];
        const negative = ['bad', 'poor', 'difficult', 'confused', 'problem', 'issue', 'error'];
        
        const lower = message.toLowerCase();
        const positiveCount = positive.filter(word => lower.includes(word)).length;
        const negativeCount = negative.filter(word => lower.includes(word)).length;
        
        if (positiveCount > negativeCount) return 'positive';
        if (negativeCount > positiveCount) return 'negative';
        return 'neutral';
    }

    // Check if message is a follow-up
    isFollowUpQuestion(message, context) {
        if (context.messageHistory.length < 2) return false;
        
        const lower = message.toLowerCase();
        const followUpIndicators = [
            'it', 'that', 'this', 'those', 'these',
            'more', 'else', 'also', 'another',
            'continue', 'go on', 'and'
        ];
        
        // Check if message starts with follow-up indicator
        const startsWithFollowUp = followUpIndicators.some(indicator => 
            lower.startsWith(indicator) || lower.startsWith('what about') || lower.startsWith('how about')
        );
        
        // Check if message references previous entities
        const previousEntities = this.extractEntitiesFromHistory(context.messageHistory.slice(-4));
        const currentEntities = this.extractEntities(message);
        
        const hasSharedEntities = 
            currentEntities.projects.some(p => previousEntities.projects.includes(p)) ||
            currentEntities.technologies.some(t => previousEntities.technologies.includes(t));
        
        return startsWithFollowUp || hasSharedEntities || message.length < 20;
    }

    // Extract entities from message history
    extractEntitiesFromHistory(messages) {
        const allEntities = {
            projects: [],
            technologies: [],
            concepts: []
        };
        
        messages.forEach(msg => {
            if (msg.role === 'user') {
                const entities = this.extractEntities(msg.content);
                allEntities.projects.push(...entities.projects);
                allEntities.technologies.push(...entities.technologies);
                allEntities.concepts.push(...entities.concepts);
            }
        });
        
        return allEntities;
    }

    // Retrieve relevant knowledge based on context
    retrieveRelevantKnowledge(message, analysis, context) {
        const relevant = {
            projects: {},
            skills: {},
            experience: {},
            education: {}
        };

        // Get relevant projects
        if (analysis.entities.projects.length > 0) {
            analysis.entities.projects.forEach(projectKey => {
                relevant.projects[projectKey] = this.knowledgeBase.projects[projectKey];
            });
        }

        // Get relevant skills
        if (analysis.entities.technologies.length > 0) {
            relevant.skills = this.findRelatedSkills(analysis.entities.technologies);
        }

        // Add context-based knowledge
        if (context.lastTopics.includes('experience')) {
            relevant.experience = this.knowledgeBase.experience;
        }

        if (context.lastTopics.includes('education')) {
            relevant.education = this.knowledgeBase.education;
        }

        return relevant;
    }

    // Find related skills
    findRelatedSkills(technologies) {
        const related = {
            languages: [],
            tools: [],
            domains: []
        };

        technologies.forEach(tech => {
            const techLower = tech.toLowerCase();
            
            // Find in programming languages
            Object.entries(this.knowledgeBase.skills.programming).forEach(([level, langs]) => {
                if (langs.some(l => l.toLowerCase() === techLower)) {
                    related.languages.push({ tech, level });
                }
            });

            // Find in tools
            Object.entries(this.knowledgeBase.skills.tools).forEach(([category, tools]) => {
                if (tools.some(t => t.toLowerCase() === techLower)) {
                    related.tools.push({ tech, category });
                }
            });
        });

        return related;
    }

    // Generate follow-up response
    generateFollowUpResponse(message, context, knowledge) {
        const lastAssistantMessage = this.getLastAssistantMessage(context.messageHistory);
        const lastTopic = this.extractTopicFromMessage(lastAssistantMessage);
        
        let response = '';
        
        // Acknowledge continuation
        const acknowledgments = [
            "Continuing from where we left off, ",
            "Building on that, ",
            "To elaborate further, ",
            "Additionally, ",
            "Going deeper into this, "
        ];
        
        response += acknowledgments[Math.floor(Math.random() * acknowledgments.length)];
        
        // Add specific information based on the follow-up
        if (message.toLowerCase().includes('more')) {
            response += this.provideMoreDetails(lastTopic, knowledge);
        } else if (message.toLowerCase().includes('how')) {
            response += this.explainHow(lastTopic, knowledge);
        } else if (message.toLowerCase().includes('why')) {
            response += this.explainWhy(lastTopic, knowledge);
        } else {
            response += this.expandOnTopic(lastTopic, knowledge, message);
        }
        
        return response;
    }

    // Generate response for new topic
    generateTopicResponse(message, analysis, knowledge) {
        let response = '';
        
        // Handle specific intents
        switch (analysis.intent) {
            case 'project_inquiry':
                response = this.generateProjectResponse(analysis.entities.projects[0], knowledge);
                break;
                
            case 'technical_question':
                response = this.generateTechnicalResponse(message, analysis, knowledge);
                break;
                
            case 'comparison':
                response = this.generateComparisonResponse(analysis.entities, knowledge);
                break;
                
            case 'code_request':
                response = this.generateCodeResponse(message, analysis, knowledge);
                break;
                
            case 'skills_inquiry':
                response = this.generateSkillsResponse(analysis.entities.technologies, knowledge);
                break;
                
            case 'contact':
                response = this.generateContactResponse();
                break;
                
            default:
                response = this.generateGeneralResponse(message, analysis, knowledge);
        }
        
        return response;
    }

    // Generate project-specific response.
    // Stays tight to what's actually on the project page; doesn't invent specs.
    generateProjectResponse(projectKey, knowledge) {
        const projects = this.knowledgeBase.projects;
        if (!projectKey || !projects[projectKey]) {
            const titles = Object.values(projects).map(p => p.title);
            const sample = titles.slice(0, 4).join('; ');
            return `I can walk you through any of my featured projects, e.g. ${sample}. Which one are you curious about?`;
        }

        const project = projects[projectKey];
        let response = `**${project.title}** — ${project.category}.\n\n`;
        response += `${project.description}\n\n`;

        if (Array.isArray(project.tools_used) && project.tools_used.length) {
            response += `**Tools:** ${project.tools_used.join(', ')}\n\n`;
        }

        if (project.url) {
            response += `📄 Full write-up: \`${project.url}\` (open it from the portfolio site for the interactive version).\n\n`;
        }

        response += `Want me to suggest related projects, or pull up the resume?`;
        return response;
    }

    // Generate technical response
    generateTechnicalResponse(message, analysis, knowledge) {
        const lower = message.toLowerCase();
        
        // Check for specific technical topics
        if (lower.includes('beamform')) {
            return this.explainBeamforming();
        } else if (lower.includes('pipeline')) {
            return this.explainPipeline();
        } else if (lower.includes('cryptograph') || lower.includes('quantum')) {
            return this.explainCryptography();
        } else if (lower.includes('memory') || lower.includes('pcm')) {
            return this.explainPCM();
        }
        
        // General technical response
        return `That's a great technical question. Based on your interest in ${analysis.keyPhrases.join(', ')}, I can provide detailed explanations about the implementation, architecture, and design decisions. What specific aspect would you like me to focus on?`;
    }

    // Concept explanations — kept generic and accurate. The deep
    // project-specific specs and "I implemented X" claims that used to
    // live here have been removed because they were not backed by the
    // resume or the actual project pages. The chatbot now points users
    // at the real project page instead of inventing performance numbers.

    explainBeamforming() {
        const proj = this.knowledgeBase.projects['mmwave-rf'];
        return `**Beamforming** uses an array of antennas with controlled phase and amplitude at each element to create a directional radiation pattern, electronically steering a beam without any mechanical movement.

**Fundamentals**
The array factor is AF(θ,φ) = Σ w_n · exp(j·k·r_n·û), where w_n are the complex element weights, k is the wave number, r_n the element position, and û the direction vector. By tuning the w_n you can steer the main lobe and shape sidelobes.

**Common practical concerns**
• Maintaining phase coherence across many elements
• Mutual coupling between antenna elements
• Wideband matching at mmWave frequencies
• Calibration and per-element variation

For my own work in this area, see the mmWave RF Frontend project page: \`${proj ? proj.url : 'projects/comms/mmwave-rf-complete.html'}\`.`;
    }

    explainPipeline() {
        const proj = this.knowledgeBase.projects['riscv-soc'];
        return `**Classic 5-stage RISC pipeline:** IF → ID → EX → MEM → WB.

1. **IF (Instruction Fetch)** — fetch from I-cache; branch prediction can override the next-PC.
2. **ID (Instruction Decode)** — decode the instruction, read register file, check dependencies.
3. **EX (Execute)** — ALU operations, address calculation for loads/stores, branch resolution.
4. **MEM (Memory Access)** — D-cache access, store buffer, load/store disambiguation.
5. **WB (Write Back)** — commit results to the register file.

**Hazards**
• **Data hazards** — forwarding paths from EX/MEM/WB to avoid stalls.
• **Control hazards** — branch prediction plus a fast misprediction recovery.
• **Structural hazards** — multi-port resources or duplication.

For my RISC-V SoC project page (Verilog/SystemVerilog implementation with UVM verification), see: \`${proj ? proj.url : 'projects/power-electronics/riscv-soc-complete.html'}\`.`;
    }

    explainCryptography() {
        const proj = this.knowledgeBase.projects['cryptography'];
        return `**Post-quantum cryptography (PQC)** are algorithms believed to remain secure against attackers with a large-scale quantum computer. Shor's algorithm threatens RSA and ECC, so NIST is standardizing new schemes.

**Key NIST PQC families**
• **CRYSTALS-Kyber** — key encapsulation, based on Module-LWE.
• **CRYSTALS-Dilithium** — digital signatures, also lattice-based.
• **FALCON** — compact lattice-based signatures via NTRU.
• **SPHINCS+** — stateless hash-based signatures, very conservative assumptions.

**Implementation concerns**
• Constant-time code to resist timing side-channels
• Side-channel countermeasures (masking, shuffling)
• Fault-injection resistance
• Hardware acceleration for the costly NTT operations

For my cryptography research page, see: \`${proj ? proj.url : 'research/cryptography-research.html'}\`.`;
    }

    explainPCM() {
        const proj = this.knowledgeBase.projects['pcm'];
        return `**Phase-Change Memory (PCM)** stores bits using chalcogenide materials (typically Ge₂Sb₂Te₅, "GST") that switch between a low-resistance crystalline state and a high-resistance amorphous state.

**Operation**
• **SET (write '1')** — moderate current pulse, crystallizes the cell.
• **RESET (write '0')** — short, high current pulse melts and quenches the cell into the amorphous state.
• **READ** — small sensing current measures resistance without disturbing the state.

**Open research areas**
• Materials engineering (doped GST, alternative chalcogenides, superlattice structures)
• Device physics — threshold switching, crystallization dynamics, thermal modeling, drift, retention
• Circuit design — write drivers, sense amplifiers, wear leveling, ECC
• Multi-level cells (more than 1 bit per cell)

For my PCM/GST research page, see: \`${proj ? proj.url : 'projects/semiconductor/pcm-complete-project.html'}\`.`;
    }

    // Generate comparison response — only echoes what's actually in the
    // knowledge base. No invented "both resulted in publications" claims.
    generateComparisonResponse(entities, knowledge) {
        if (entities.projects.length >= 2) {
            const proj1 = this.knowledgeBase.projects[entities.projects[0]];
            const proj2 = this.knowledgeBase.projects[entities.projects[1]];
            if (!proj1 || !proj2) {
                return "I couldn't match both projects you mentioned. Try names like 'mmWave RF', 'RISC-V SoC', 'cryptography', or 'PCM'.";
            }
            const tools1 = (proj1.tools_used || []).join(', ') || '—';
            const tools2 = (proj2.tools_used || []).join(', ') || '—';
            return `Comparing **${proj1.title}** and **${proj2.title}**:

**Scope**
• ${proj1.title}: ${proj1.description}
• ${proj2.title}: ${proj2.description}

**Domain**
• ${proj1.title}: ${proj1.category}
• ${proj2.title}: ${proj2.category}

**Tools**
• ${proj1.title}: ${tools1}
• ${proj2.title}: ${tools2}

For deeper specs, the full project pages are at \`${proj1.url || ''}\` and \`${proj2.url || ''}\`.`;
        }
        
        return "I can compare different projects, technologies, or approaches. What specific comparison would you like me to make?";
    }

    // Generate code response
    generateCodeResponse(message, analysis, knowledge) {
        const lower = message.toLowerCase();
        
        // Determine language and topic
        let code = '';
        let language = 'python';
        
        if (lower.includes('verilog') || lower.includes('hardware') || lower.includes('rtl')) {
            language = 'verilog';
            code = this.getVerilogExample(lower);
        } else if (lower.includes('python') || lower.includes('rf') || lower.includes('algorithm')) {
            language = 'python';
            code = this.getPythonExample(lower);
        } else {
            // Default to most relevant based on context
            code = this.getPythonExample('general');
        }
        
        return `Here's a code example that demonstrates the concept:

\`\`\`${language}
${code}
\`\`\`

This implementation shows ${this.explainCode(code, language)}. 

Would you like me to explain specific parts of the code, show a different example, or help you adapt it for your use case?`;
    }

    // Get Verilog example
    getVerilogExample(topic) {
        if (topic.includes('alu')) {
            return `// RISC-V ALU Module
module riscv_alu #(
    parameter WIDTH = 64
)(
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire [3:0]       alu_op,
    output reg  [WIDTH-1:0] result,
    output wire             zero
);

    always @(*) begin
        case(alu_op)
            4'b0000: result = a + b;           // ADD
            4'b0001: result = a - b;           // SUB
            4'b0010: result = a << b[5:0];     // SLL
            4'b0011: result = ($signed(a) < $signed(b)); // SLT
            4'b0100: result = (a < b);         // SLTU
            4'b0101: result = a ^ b;           // XOR
            4'b0110: result = a >> b[5:0];     // SRL
            4'b0111: result = $signed(a) >>> b[5:0]; // SRA
            4'b1000: result = a | b;           // OR
            4'b1001: result = a & b;           // AND
            default: result = {WIDTH{1'b0}};
        endcase
    end
    
    assign zero = (result == {WIDTH{1'b0}});
    
endmodule`;
        } else if (topic.includes('cache')) {
            return `// Direct-Mapped Cache Controller
module cache_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter CACHE_SIZE = 8192,  // 8KB
    parameter BLOCK_SIZE = 32      // 32 bytes per block
)(
    input  wire                    clk,
    input  wire                    rst_n,
    // CPU interface
    input  wire [ADDR_WIDTH-1:0]   cpu_addr,
    input  wire [DATA_WIDTH-1:0]   cpu_write_data,
    input  wire                    cpu_read,
    input  wire                    cpu_write,
    output reg  [DATA_WIDTH-1:0]   cpu_read_data,
    output reg                     cpu_ready,
    // Memory interface
    output reg  [ADDR_WIDTH-1:0]   mem_addr,
    output reg  [DATA_WIDTH-1:0]   mem_write_data,
    output reg                     mem_read,
    output reg                     mem_write,
    input  wire [DATA_WIDTH-1:0]   mem_read_data,
    input  wire                    mem_ready
);

    localparam NUM_BLOCKS = CACHE_SIZE / BLOCK_SIZE;
    localparam INDEX_BITS = $clog2(NUM_BLOCKS);
    localparam OFFSET_BITS = $clog2(BLOCK_SIZE);
    localparam TAG_BITS = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;
    
    // Cache storage
    reg [DATA_WIDTH-1:0] cache_data [NUM_BLOCKS-1:0];
    reg [TAG_BITS-1:0]   cache_tags [NUM_BLOCKS-1:0];
    reg                   cache_valid [NUM_BLOCKS-1:0];
    reg                   cache_dirty [NUM_BLOCKS-1:0];
    
    // Address breakdown
    wire [TAG_BITS-1:0]    tag = cpu_addr[ADDR_WIDTH-1:ADDR_WIDTH-TAG_BITS];
    wire [INDEX_BITS-1:0]  index = cpu_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS];
    wire [OFFSET_BITS-1:0] offset = cpu_addr[OFFSET_BITS-1:0];
    
    // Cache hit logic
    wire cache_hit = cache_valid[index] && (cache_tags[index] == tag);
    
    // FSM states
    typedef enum logic [2:0] {
        IDLE,
        COMPARE_TAG,
        WRITEBACK,
        ALLOCATE
    } state_t;
    
    state_t state, next_state;
    
    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (cpu_read || cpu_write)
                    next_state = COMPARE_TAG;
            end
            
            COMPARE_TAG: begin
                if (cache_hit)
                    next_state = IDLE;
                else if (cache_dirty[index])
                    next_state = WRITEBACK;
                else
                    next_state = ALLOCATE;
            end
            
            WRITEBACK: begin
                if (mem_ready)
                    next_state = ALLOCATE;
            end
            
            ALLOCATE: begin
                if (mem_ready)
                    next_state = IDLE;
            end
        endcase
    end
    
endmodule`;
        }
        
        // Default pipeline example
        return `// 5-Stage Pipeline Register
module pipeline_reg #(
    parameter WIDTH = 32
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             stall,
    input  wire             flush,
    input  wire [WIDTH-1:0] data_in,
    output reg  [WIDTH-1:0] data_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || flush)
            data_out <= {WIDTH{1'b0}};
        else if (!stall)
            data_out <= data_in;
    end

endmodule`;
    }

    // Get Python example
    getPythonExample(topic) {
        if (topic.includes('rf') || topic.includes('link')) {
            return `import numpy as np
import matplotlib.pyplot as plt

class RFLinkBudget:
    """RF Link Budget Calculator for mmWave Systems"""
    
    def __init__(self, freq_ghz=28, tx_power_dbm=30, tx_gain_db=25, rx_gain_db=25):
        self.freq_ghz = freq_ghz
        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_db = tx_gain_db
        self.rx_gain_db = rx_gain_db
        self.noise_figure_db = 3
        self.bandwidth_mhz = 100
        
    def calculate_fspl(self, distance_km):
        """Calculate Free Space Path Loss"""
        # FSPL = 20*log10(d) + 20*log10(f) + 92.45
        return 20 * np.log10(distance_km) + 20 * np.log10(self.freq_ghz) + 92.45
    
    def calculate_link_budget(self, distance_km):
        """Complete link budget calculation"""
        fspl = self.calculate_fspl(distance_km)
        
        # Received power
        rx_power_dbm = self.tx_power_dbm + self.tx_gain_db + self.rx_gain_db - fspl
        
        # Noise floor
        noise_floor_dbm = -174 + 10*np.log10(self.bandwidth_mhz * 1e6) + self.noise_figure_db
        
        # SNR
        snr_db = rx_power_dbm - noise_floor_dbm
        
        # Shannon capacity
        capacity_mbps = self.bandwidth_mhz * np.log2(1 + 10**(snr_db/10))
        
        return {
            'distance_km': distance_km,
            'fspl_db': fspl,
            'rx_power_dbm': rx_power_dbm,
            'noise_floor_dbm': noise_floor_dbm,
            'snr_db': snr_db,
            'capacity_mbps': capacity_mbps
        }
    
    def plot_link_budget(self, max_distance_km=10):
        """Plot link budget vs distance"""
        distances = np.linspace(0.1, max_distance_km, 100)
        results = [self.calculate_link_budget(d) for d in distances]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Path Loss
        axes[0,0].plot(distances, [r['fspl_db'] for r in results])
        axes[0,0].set_xlabel('Distance (km)')
        axes[0,0].set_ylabel('Path Loss (dB)')
        axes[0,0].set_title('Free Space Path Loss')
        axes[0,0].grid(True)
        
        # Received Power
        axes[0,1].plot(distances, [r['rx_power_dbm'] for r in results])
        axes[0,1].axhline(y=[r['noise_floor_dbm'] for r in results][0], 
                         color='r', linestyle='--', label='Noise Floor')
        axes[0,1].set_xlabel('Distance (km)')
        axes[0,1].set_ylabel('Power (dBm)')
        axes[0,1].set_title('Received Power vs Noise Floor')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # SNR
        axes[1,0].plot(distances, [r['snr_db'] for r in results])
        axes[1,0].axhline(y=0, color='r', linestyle='--', label='0 dB SNR')
        axes[1,0].set_xlabel('Distance (km)')
        axes[1,0].set_ylabel('SNR (dB)')
        axes[1,0].set_title('Signal-to-Noise Ratio')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Capacity
        axes[1,1].plot(distances, [r['capacity_mbps'] for r in results])
        axes[1,1].set_xlabel('Distance (km)')
        axes[1,1].set_ylabel('Capacity (Mbps)')
        axes[1,1].set_title('Shannon Capacity')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        return fig

# Example usage
rf_system = RFLinkBudget(freq_ghz=28, tx_power_dbm=30)
result = rf_system.calculate_link_budget(distance_km=1)
print(f"Link Budget at 1km: {result}")`;
        } else if (topic.includes('beam') || topic.includes('array')) {
            return `import numpy as np
from scipy import signal

class BeamformingArray:
    """Phased Array Beamforming Simulator"""
    
    def __init__(self, num_elements=8, freq_ghz=28, spacing_lambda=0.5):
        self.num_elements = num_elements
        self.freq_ghz = freq_ghz
        self.wavelength = 3e8 / (freq_ghz * 1e9)  # meters
        self.spacing = spacing_lambda * self.wavelength
        self.element_positions = np.arange(num_elements) * self.spacing
        
    def calculate_array_factor(self, theta_deg, weights=None):
        """Calculate array factor for given angle"""
        theta_rad = np.deg2rad(theta_deg)
        k = 2 * np.pi / self.wavelength  # wave number
        
        if weights is None:
            weights = np.ones(self.num_elements)
        
        # Phase shift for each element
        phase_shifts = k * self.element_positions * np.sin(theta_rad)
        
        # Array factor
        af = np.sum(weights * np.exp(1j * phase_shifts))
        return np.abs(af)
    
    def steer_beam(self, target_angle_deg):
        """Calculate weights to steer beam to target angle"""
        theta_rad = np.deg2rad(target_angle_deg)
        k = 2 * np.pi / self.wavelength
        
        # Progressive phase shift for beam steering
        phase_shifts = -k * self.element_positions * np.sin(theta_rad)
        weights = np.exp(1j * phase_shifts)
        
        return weights
    
    def plot_pattern(self, weights=None, angles=None):
        """Plot radiation pattern"""
        if angles is None:
            angles = np.linspace(-90, 90, 361)
        
        pattern = [self.calculate_array_factor(angle, weights) for angle in angles]
        pattern_db = 20 * np.log10(np.array(pattern) / np.max(pattern))
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(angles, pattern_db)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Normalized Pattern (dB)')
        plt.title(f'{self.num_elements}-Element Array Pattern at {self.freq_ghz} GHz')
        plt.grid(True)
        plt.ylim([-40, 0])
        plt.axhline(y=-3, color='r', linestyle='--', label='-3 dB')
        plt.legend()
        
        return pattern_db

# Example: Steer beam to 30 degrees
array = BeamformingArray(num_elements=8, freq_ghz=28)
weights = array.steer_beam(target_angle_deg=30)
pattern = array.plot_pattern(weights)`;
        } else if (topic.includes('crypto') || topic.includes('quantum')) {
            return `import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import hashlib

@dataclass
class KyberParams:
    """CRYSTALS-Kyber parameters for different security levels"""
    n: int = 256      # polynomial degree
    q: int = 3329     # modulus
    k: int = 3        # module rank (3 for Kyber768)
    eta1: int = 2     # noise parameter for key generation
    eta2: int = 2     # noise parameter for encryption
    du: int = 10      # ciphertext compression
    dv: int = 4       # message compression

class KyberSimulator:
    """Simplified CRYSTALS-Kyber implementation for educational purposes"""
    
    def __init__(self, params: KyberParams = KyberParams()):
        self.params = params
        self.n = params.n
        self.q = params.q
        self.k = params.k
        
    def generate_polynomial(self, seed: bytes, nonce: int) -> np.ndarray:
        """Generate polynomial from seed using shake128"""
        # Simplified - in real implementation use SHAKE128
        np.random.seed(int.from_bytes(seed[:4], 'little') + nonce)
        return np.random.randint(0, self.q, self.n)
    
    def sample_noise(self, eta: int) -> np.ndarray:
        """Sample from centered binomial distribution"""
        # CBD_eta(PRF(seed))
        a = np.random.binomial(eta, 0.5, self.n)
        b = np.random.binomial(eta, 0.5, self.n)
        return (a - b) % self.q
    
    def ntt(self, poly: np.ndarray) -> np.ndarray:
        """Number Theoretic Transform (simplified)"""
        # Real implementation uses fast NTT with precomputed twiddle factors
        # This is a placeholder showing the concept
        return np.fft.fft(poly)[:self.n].real.astype(int) % self.q
    
    def inv_ntt(self, poly: np.ndarray) -> np.ndarray:
        """Inverse NTT"""
        return np.fft.ifft(poly)[:self.n].real.astype(int) % self.q
    
    def poly_mult(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Polynomial multiplication in NTT domain"""
        a_ntt = self.ntt(a)
        b_ntt = self.ntt(b)
        c_ntt = (a_ntt * b_ntt) % self.q
        return self.inv_ntt(c_ntt)
    
    def keygen(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate public and private key pair"""
        # Generate matrix A
        seed = hashlib.sha3_256(b"seed").digest()
        A = np.array([[self.generate_polynomial(seed, i*self.k+j) 
                       for j in range(self.k)] 
                      for i in range(self.k)])
        
        # Sample secret and error
        s = np.array([self.sample_noise(self.params.eta1) for _ in range(self.k)])
        e = np.array([self.sample_noise(self.params.eta1) for _ in range(self.k)])
        
        # Compute public key: pk = A*s + e
        pk = np.zeros((self.k, self.n), dtype=int)
        for i in range(self.k):
            for j in range(self.k):
                pk[i] = (pk[i] + self.poly_mult(A[i][j], s[j])) % self.q
            pk[i] = (pk[i] + e[i]) % self.q
        
        return pk, s
    
    def encrypt(self, pk: np.ndarray, message: bytes) -> np.ndarray:
        """Encrypt message using public key"""
        # Sample randomness
        r = np.array([self.sample_noise(self.params.eta1) for _ in range(self.k)])
        e1 = np.array([self.sample_noise(self.params.eta2) for _ in range(self.k)])
        e2 = self.sample_noise(self.params.eta2)
        
        # Encode message as polynomial
        m = np.array([int(bit) * (self.q // 2) for bit in 
                     format(int.from_bytes(message, 'little'), '0256b')])
        
        # Encryption (simplified)
        # u = A^T * r + e1
        # v = pk^T * r + e2 + m
        
        return np.concatenate([r, [m]])  # Simplified ciphertext
    
    def performance_test(self, iterations: int = 100):
        """Benchmark key generation and encryption"""
        import time
        
        # Key generation
        start = time.time()
        for _ in range(iterations):
            pk, sk = self.keygen()
        keygen_time = (time.time() - start) / iterations * 1000
        
        # Encryption
        message = b"Hello Quantum World!"
        start = time.time()
        for _ in range(iterations):
            ct = self.encrypt(pk, message)
        encrypt_time = (time.time() - start) / iterations * 1000
        
        print(f"Kyber-768 Performance (simplified):")
        print(f"Key Generation: {keygen_time:.2f} ms")
        print(f"Encryption: {encrypt_time:.2f} ms")
        print(f"Public Key Size: {pk.size * 2} bytes")
        print(f"Ciphertext Size: {ct.size * 2} bytes")
        
        return keygen_time, encrypt_time

# Example usage
kyber = KyberSimulator()
pk, sk = kyber.keygen()
print(f"Generated Kyber keypair")
print(f"Public key shape: {pk.shape}")
print(f"Secret key shape: {sk.shape}")

# Performance benchmark
kyber.performance_test(iterations=10)`;
        }
        
        // Default general example
        return `def optimize_algorithm(data, threshold=0.8):
    """Example optimization algorithm"""
    # Preprocess data
    processed = preprocess(data)
    
    # Apply optimization
    result = []
    for item in processed:
        if evaluate(item) > threshold:
            result.append(transform(item))
    
    return result

def preprocess(data):
    """Data preprocessing step"""
    return [normalize(x) for x in data if validate(x)]

def evaluate(item):
    """Evaluation metric"""
    return sum(item) / len(item) if item else 0

def transform(item):
    """Apply transformation"""
    return [x * 2 for x in item]`;
    }

    // Explain code functionality
    explainCode(code, language) {
        if (language === 'verilog') {
            if (code.includes('alu')) {
                return "a complete RISC-V ALU implementation with all standard operations";
            } else if (code.includes('cache')) {
                return "a direct-mapped cache controller with hit/miss detection and memory interface";
            } else {
                return "a pipeline register with stall and flush support";
            }
        } else if (language === 'python') {
            if (code.includes('RFLink')) {
                return "RF link budget calculations including path loss, SNR, and Shannon capacity";
            } else if (code.includes('Beamform')) {
                return "phased array beamforming with beam steering capabilities";
            } else if (code.includes('Kyber')) {
                return "a simplified CRYSTALS-Kyber post-quantum encryption implementation";
            }
        }
        return "the core algorithm implementation";
    }

    // Generate skills response — sourced from the resume.
    generateSkillsResponse(technologies, knowledge) {
        const s = this.knowledgeBase.skills;

        let response = "Here's a summary of my technical skill set, drawn directly from my resume:\n\n";

        response += "**Programming Languages**\n";
        if (s.programming.proficient && s.programming.proficient.length) {
            response += `• Proficient: ${s.programming.proficient.join(', ')}\n`;
        }
        if (s.programming.familiar && s.programming.familiar.length) {
            response += `• Familiar: ${s.programming.familiar.join(', ')}\n`;
        }
        response += "\n";

        if (s.semiconductor) {
            response += "**Semiconductor Process**\n";
            Object.entries(s.semiconductor).forEach(([category, items]) => {
                const label = category.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                response += `• ${label}: ${items.slice(0, 4).join('; ')}${items.length > 4 ? '; …' : ''}\n`;
            });
            response += "\n";
        }

        if (s.systems_engineering && s.systems_engineering.length) {
            response += "**Systems Engineering & V&V**\n";
            response += `• ${s.systems_engineering.slice(0, 5).join('; ')}${s.systems_engineering.length > 5 ? '; …' : ''}\n\n`;
        }

        if (s.tools) {
            response += "**Tools & Software**\n";
            const toolLabels = { rf: 'RF', fpga_eda: 'FPGA / EDA', cleanroom: 'Cleanroom', software: 'Software', lab: 'Lab' };
            Object.entries(s.tools).forEach(([category, items]) => {
                const label = toolLabels[category] || category.toUpperCase();
                response += `• ${label}: ${items.join(', ')}\n`;
            });
        }

        if (technologies && technologies.length > 0) {
            response += `\n\nYou mentioned ${technologies.join(' and ')} — happy to point you to specific projects on the site that use those.`;
        }

        return response;
    }

    // Generate contact response
    generateContactResponse() {
        const p = this.knowledgeBase.personal;
        return `Here are the best ways to reach ${p.name}:

**Direct contact**
📧 Email: ${p.email}
📞 Phone: ${p.phone}
📍 Location: ${p.location}

**Online**
🌐 Portfolio: ${p.portfolio}
💼 LinkedIn: ${p.linkedin}
🐙 GitHub:   ${p.github}

You can also send a message via the contact form on the portfolio homepage, or book a meeting on Calendly through the "Get in Touch" section.`;
    }

    // Generate general response
    generateGeneralResponse(message, analysis, knowledge) {
        // Try to provide helpful response even for general queries
        let response = `I understand you're interested in "${message}". `;
        
        if (analysis.entities.projects.length > 0) {
            response += `\n\nBased on your mention of ${analysis.entities.projects.join(' and ')}, `;
            response += `I can provide detailed information about these projects, including technical specifications, challenges solved, and outcomes. `;
        }
        
        if (analysis.entities.technologies.length > 0) {
            response += `\n\nRegarding ${analysis.entities.technologies.join(' and ')}, `;
            response += `I have practical experience with these technologies and can share implementation details or code examples. `;
        }
        
        if (analysis.entities.concepts.length > 0) {
            response += `\n\nThe concepts you mentioned (${analysis.entities.concepts.join(', ')}) `;
            response += `are areas I've worked on extensively. I can explain the theory, show practical implementations, or discuss real-world applications. `;
        }
        
        response += `\n\nHow can I help you explore this topic further? Would you like:
• Technical details and explanations
• Code examples and implementations
• Project demonstrations
• Practical applications
• Related resources and documentation`;
        
        return response;
    }

    // Check if should include code
    shouldIncludeCode(message, analysis) {
        const codeKeywords = ['code', 'example', 'implement', 'snippet', 'function', 'algorithm', 'how to write'];
        return codeKeywords.some(keyword => message.toLowerCase().includes(keyword)) ||
               analysis.intent === 'code_request';
    }

    // Select relevant code
    selectRelevantCode(message, analysis) {
        const lower = message.toLowerCase();
        
        // Determine most relevant code
        if (lower.includes('verilog') || lower.includes('hardware')) {
            return {
                language: 'verilog',
                code: this.getVerilogExample(lower)
            };
        } else if (lower.includes('python') || lower.includes('rf') || lower.includes('crypto')) {
            return {
                language: 'python',
                code: this.getPythonExample(lower)
            };
        }
        
        return null;
    }

    // Check if should include links
    shouldIncludeLinks(message, analysis) {
        return analysis.entities.projects.length > 0 ||
               message.toLowerCase().includes('demo') ||
               message.toLowerCase().includes('github') ||
               message.toLowerCase().includes('documentation');
    }

    // Select relevant links
    selectRelevantLinks(message, analysis) {
        const links = [];
        
        if (analysis.entities.projects.length > 0) {
            analysis.entities.projects.forEach(projectKey => {
                const project = this.knowledgeBase.projects[projectKey];
                if (project) {
                    links.push({
                        text: `${project.title} Demo`,
                        url: `${projectKey}-complete.html`
                    });
                }
            });
        }
        
        if (message.toLowerCase().includes('github')) {
            links.push({
                text: 'GitHub Profile',
                url: this.knowledgeBase.personal.github
            });
        }
        
        return links;
    }

    // Generate contextual suggestions
    generateContextualSuggestions(message, context, analysis) {
        const suggestions = [];
        
        // Based on current topic
        if (analysis.entities.projects.length > 0) {
            suggestions.push('Show me the technical specifications');
            suggestions.push('What were the main challenges?');
            suggestions.push('Can I see code examples?');
            suggestions.push('How does it compare to other projects?');
        } else if (analysis.intent === 'technical_question') {
            suggestions.push('Explain in more detail');
            suggestions.push('Show me an implementation');
            suggestions.push('What are the alternatives?');
            suggestions.push('Real-world applications?');
        } else if (context.messageCount < 3) {
            // Early in conversation
            suggestions.push('Tell me about your projects');
            suggestions.push('What technologies do you use?');
            suggestions.push('Show me your best work');
            suggestions.push('How can I contact you?');
        } else {
            // Deep in conversation - suggest related topics
            const lastTopics = context.lastTopics || [];
            if (lastTopics.includes('rf')) {
                suggestions.push('Explain beamforming in detail');
                suggestions.push('Show RF circuit designs');
            } else if (lastTopics.includes('hardware')) {
                suggestions.push('Tell me about the RISC-V pipeline');
                suggestions.push('Cache architecture details');
            } else if (lastTopics.includes('crypto')) {
                suggestions.push('Post-quantum algorithms comparison');
                suggestions.push('Implementation challenges');
            }
        }
        
        return suggestions.slice(0, 4); // Max 4 suggestions
    }

    // Update conversation context
    updateConversationContext(conversation, message, response) {
        const analysis = this.analyzeMessage(message);
        
        // Update topics
        if (!conversation.context.topics) {
            conversation.context.topics = [];
        }
        
        if (analysis.entities.projects.length > 0) {
            conversation.context.topics.push(...analysis.entities.projects);
        }
        
        if (analysis.entities.concepts.length > 0) {
            conversation.context.topics.push(...analysis.entities.concepts);
        }
        
        // Update entities set
        if (!conversation.context.entities) {
            conversation.context.entities = new Set();
        }
        
        analysis.entities.projects.forEach(p => conversation.context.entities.add(p));
        analysis.entities.technologies.forEach(t => conversation.context.entities.add(t));
        
        // Keep topics list reasonable size
        if (conversation.context.topics.length > 20) {
            conversation.context.topics = conversation.context.topics.slice(-20);
        }
    }

    // Generate conversation title
    generateConversationTitle(firstMessage) {
        const analysis = this.analyzeMessage(firstMessage);
        
        if (analysis.entities.projects.length > 0) {
            const project = this.knowledgeBase.projects[analysis.entities.projects[0]];
            return project ? project.title : 'Project Discussion';
        }
        
        if (analysis.entities.technologies.length > 0) {
            return `${analysis.entities.technologies[0]} Discussion`;
        }
        
        if (analysis.intent !== 'general') {
            return analysis.intent.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        }
        
        // Use first few words
        const words = firstMessage.split(' ').slice(0, 5).join(' ');
        return words.length > 30 ? words.substring(0, 30) + '...' : words;
    }

    // Extract recent topics from messages
    extractRecentTopics(messages) {
        const topics = new Set();
        
        messages.forEach(msg => {
            if (msg.role === 'user') {
                const analysis = this.analyzeMessage(msg.content);
                analysis.entities.projects.forEach(p => topics.add(p));
                analysis.entities.concepts.forEach(c => topics.add(c));
            }
        });
        
        return Array.from(topics);
    }

    // Get last assistant message
    getLastAssistantMessage(messages) {
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === 'assistant') {
                return messages[i].content;
            }
        }
        return '';
    }

    // Extract topic from message
    extractTopicFromMessage(message) {
        const analysis = this.analyzeMessage(message);
        if (analysis.entities.projects.length > 0) {
            return analysis.entities.projects[0];
        }
        if (analysis.entities.concepts.length > 0) {
            return analysis.entities.concepts[0];
        }
        return 'general';
    }

    // Provide more details about a topic / project.
    // Stays factual: only echoes data actually in the knowledge base
    // and points users to the real project page for the rest.
    provideMoreDetails(topic, knowledge) {
        const project = this.knowledgeBase.projects[topic];
        if (project) {
            let details = `Here's a bit more on **${project.title}**:\n\n`;
            details += `${project.description}\n\n`;
            if (Array.isArray(project.tools_used) && project.tools_used.length) {
                details += `**Tools used:** ${project.tools_used.join(', ')}\n\n`;
            }
            if (project.url) {
                details += `For the full write-up, demos, and figures, see \`${project.url}\` on the portfolio site.`;
            }
            return details;
        }
        return "I can dig deeper if you tell me which project, skill, or experience you're curious about.";
    }

    // Explain how something works
    explainHow(topic, knowledge) {
        // Provide detailed how-to explanation based on topic
        if (topic.includes('beam')) {
            return this.explainBeamforming();
        } else if (topic.includes('pipe')) {
            return this.explainPipeline();
        } else if (topic.includes('crypto')) {
            return this.explainCryptography();
        } else if (topic.includes('pcm')) {
            return this.explainPCM();
        }
        
        return "let me explain the step-by-step process and implementation details.";
    }

    // Explain why
    explainWhy(topic, knowledge) {
        return `the reasoning behind this approach involves several factors:

1. **Technical Requirements:** The specific constraints and requirements that drove this design
2. **Performance Optimization:** How this approach maximizes efficiency and performance
3. **Trade-offs:** The balance between complexity, cost, and functionality
4. **Industry Standards:** Alignment with established practices and standards

Would you like me to elaborate on any of these aspects?`;
    }

    // Expand on topic
    expandOnTopic(topic, knowledge, message) {
        // Intelligently expand based on the follow-up question
        const lower = message.toLowerCase();
        
        if (lower.includes('challenge') || lower.includes('difficult')) {
            return `the main challenges involved:

• Technical complexity in implementation
• Resource constraints and optimization needs
• Integration with existing systems
• Performance requirements
• Validation and testing

Each of these presented unique problems that required innovative solutions.`;
        }
        
        if (lower.includes('result') || lower.includes('outcome')) {
            return `the outcomes and achievements include:

• Successful implementation meeting all requirements
• Performance improvements over baseline
• Publications and recognition
• Practical applications in production
• Lessons learned for future projects`;
        }
        
        return `let me provide additional context and details about this topic. The implementation involves multiple layers of complexity, from theoretical foundations to practical considerations.`;
    }

    // Get all conversations for a user
    getUserConversations(userId) {
        return Object.values(this.conversations)
            .filter(c => c.userId === userId)
            .sort((a, b) => b.lastModified - a.lastModified);
    }

    // Delete conversation
    deleteConversation(conversationId) {
        if (this.conversations[conversationId]) {
            delete this.conversations[conversationId];
            this.saveAllConversations();
            return true;
        }
        return false;
    }

    // Clear all conversations for a user
    clearUserConversations(userId) {
        Object.keys(this.conversations).forEach(id => {
            if (this.conversations[id].userId === userId) {
                delete this.conversations[id];
            }
        });
        this.saveAllConversations();
    }

    // Export conversation
    exportConversation(conversationId, format = 'json') {
        const conversation = this.conversations[conversationId];
        if (!conversation) return null;
        
        if (format === 'json') {
            return JSON.stringify(conversation, null, 2);
        } else if (format === 'text') {
            let text = `Conversation: ${conversation.title}\n`;
            text += `Created: ${conversation.created}\n`;
            text += `Messages: ${conversation.messageCount}\n\n`;
            text += '---\n\n';
            
            conversation.messages.forEach(msg => {
                text += `[${msg.timestamp}] ${msg.role.toUpperCase()}:\n`;
                text += `${msg.content}\n\n`;
            });
            
            return text;
        }
        
        return conversation;
    }

    // Search conversations
    searchConversations(userId, query) {
        const userConversations = this.getUserConversations(userId);
        const queryLower = query.toLowerCase();
        
        return userConversations.filter(conv => {
            // Search in title
            if (conv.title.toLowerCase().includes(queryLower)) return true;
            
            // Search in messages
            return conv.messages.some(msg => 
                msg.content.toLowerCase().includes(queryLower)
            );
        });
    }

    // Get conversation statistics
    getConversationStats(conversationId) {
        const conversation = this.conversations[conversationId];
        if (!conversation) return null;
        
        const topics = {};
        const intents = {};
        
        conversation.messages.forEach(msg => {
            if (msg.role === 'user') {
                const analysis = this.analyzeMessage(msg.content);
                
                // Count topics
                analysis.entities.projects.forEach(p => {
                    topics[p] = (topics[p] || 0) + 1;
                });
                
                // Count intents
                intents[analysis.intent] = (intents[analysis.intent] || 0) + 1;
            }
        });
        
        return {
            messageCount: conversation.messageCount,
            duration: new Date() - new Date(conversation.created),
            topics: topics,
            intents: intents,
            avgMessageLength: conversation.messages.reduce((sum, msg) => 
                sum + msg.content.length, 0) / conversation.messages.length
        };
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.AIAssistantBackendV2 = AIAssistantBackendV2;
}