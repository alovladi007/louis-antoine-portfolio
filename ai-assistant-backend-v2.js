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
        return {
            projects: {
                "mmwave-rf": {
                    title: "mmWave RF Frontend Design",
                    category: "RF Semiconductor Design",
                    description: "28 GHz 5G NR n257 band phased array system",
                    details: {
                        frequency: "27-29 GHz",
                        architecture: "8x8 phased array",
                        power: "38 dBm output",
                        efficiency: "42% PAE",
                        noise: "1.2 dB NF",
                        steering: "±60° beam steering",
                        technology: "GaN HEMT",
                        applications: ["5G NR", "mmWave communications", "Beamforming"]
                    },
                    technical_specs: {
                        pa_specs: {
                            technology: "0.15μm GaN HEMT",
                            topology: "2-stage Doherty",
                            gain: "25 dB",
                            p1db: "35 dBm",
                            psat: "38 dBm",
                            pae_peak: "42%",
                            bandwidth: "2 GHz"
                        },
                        lna_specs: {
                            technology: "65nm CMOS",
                            stages: 3,
                            gain: "20 dB",
                            nf: "1.2 dB",
                            iip3: "-5 dBm",
                            current: "8 mA"
                        },
                        beamforming: {
                            elements: "8x8 array",
                            spacing: "λ/2",
                            scan_angle: "±60°",
                            sidelobe: "-13 dB",
                            beam_width: "12°",
                            gain_variation: "< 3 dB"
                        }
                    },
                    challenges_solved: [
                        "High-frequency parasitic effects",
                        "Thermal management at high power",
                        "Phase synchronization across array",
                        "Wideband impedance matching",
                        "Digital predistortion implementation"
                    ],
                    tools_used: ["Keysight ADS", "ANSYS HFSS", "Cadence", "MATLAB", "Python"],
                    outcomes: [
                        "Successfully demonstrated 5G NR compliance",
                        "Achieved industry-leading efficiency",
                        "Published in IEEE conference",
                        "Patent pending on DPD algorithm"
                    ]
                },
                
                "risc-v-soc": {
                    title: "RISC-V SoC with Custom Accelerators",
                    category: "Hardware/Embedded Systems",
                    description: "64-bit RISC-V processor with ML and crypto accelerators",
                    details: {
                        architecture: "RV64IMAFDC",
                        pipeline: "5-stage, out-of-order",
                        frequency: "1.2 GHz @ 28nm",
                        cache: "32KB L1I/D, 256KB L2",
                        accelerators: ["ML inference", "AES-256", "SHA-3"],
                        noc: "2D mesh, 4x4",
                        memory: "DDR4-3200 controller"
                    },
                    technical_specs: {
                        core: {
                            isa: "RV64IMAFDC",
                            pipeline_stages: 5,
                            issue_width: 2,
                            rob_size: 64,
                            branch_predictor: "TAGE, 4K entries",
                            btb_size: "2K entries",
                            ras_size: 16,
                            performance: "2.3 DMIPS/MHz"
                        },
                        ml_accelerator: {
                            type: "Systolic array",
                            size: "16x16 MACs",
                            precision: "INT8/INT16/FP16",
                            peak_ops: "512 GOPS",
                            supported_ops: ["Conv2D", "MatMul", "Pooling", "ReLU"],
                            memory: "128KB scratchpad"
                        },
                        security: {
                            crypto: ["AES-256-GCM", "SHA-3", "RSA-2048"],
                            features: ["Secure boot", "TrustZone", "PUF"],
                            side_channel: "Power analysis resistant",
                            certifications: "FIPS 140-2 Level 2"
                        }
                    },
                    verification: {
                        methodology: "UVM",
                        coverage: "98% functional, 95% code",
                        tests: "10,000+ directed, 1M+ random",
                        formal: "Property checking for critical paths",
                        emulation: "Xilinx VCU128 FPGA"
                    },
                    performance: {
                        coremark: "5.2 CoreMark/MHz",
                        dhrystone: "2.3 DMIPS/MHz",
                        ml_benchmarks: {
                            resnet50: "120 fps @ INT8",
                            mobilenet: "450 fps @ INT8",
                            bert: "30 sequences/sec"
                        },
                        power: "2.5W typical, 4W peak"
                    }
                },
                
                "cryptography": {
                    title: "Post-Quantum Cryptography Research",
                    category: "Security Research",
                    description: "Implementation and optimization of NIST PQC algorithms",
                    details: {
                        algorithms: ["CRYSTALS-Kyber", "CRYSTALS-Dilithium", "FALCON", "SPHINCS+"],
                        platforms: ["FPGA", "ASIC", "Software"],
                        attacks_studied: ["Side-channel", "Fault injection", "Quantum"],
                        optimizations: ["Hardware acceleration", "Constant-time", "Memory-efficient"]
                    },
                    implementations: {
                        kyber: {
                            security_level: [512, 768, 1024],
                            key_gen: "0.05ms",
                            encapsulation: "0.06ms",
                            decapsulation: "0.07ms",
                            hw_resources: "15K LUTs, 8 DSPs",
                            sw_performance: "100K ops/sec"
                        },
                        dilithium: {
                            security_level: [2, 3, 5],
                            key_gen: "0.1ms",
                            sign: "0.3ms",
                            verify: "0.1ms",
                            signature_size: "2420 bytes",
                            hw_acceleration: "5x speedup"
                        }
                    },
                    research_contributions: [
                        "Novel side-channel countermeasures",
                        "Optimized NTT implementation",
                        "Hybrid classical-PQC protocols",
                        "Lightweight PQC for IoT"
                    ]
                },
                
                "pcm": {
                    title: "Phase Change Memory Research",
                    category: "Semiconductor Research",
                    description: "Next-generation non-volatile memory technology",
                    details: {
                        material: "Ge2Sb2Te5 (GST)",
                        cell_size: "4F²",
                        endurance: "10^9 cycles",
                        retention: "10 years @ 85°C",
                        switching_time: "50ns",
                        multi_level: "2 bits/cell"
                    },
                    research_areas: {
                        materials: [
                            "Doped GST optimization",
                            "Alternative chalcogenides",
                            "Superlattice structures",
                            "Interfacial engineering"
                        ],
                        device_physics: [
                            "Threshold switching mechanisms",
                            "Crystallization dynamics",
                            "Thermal modeling",
                            "Reliability physics"
                        ],
                        circuit_design: [
                            "Write driver optimization",
                            "Sense amplifier design",
                            "Wear leveling algorithms",
                            "Error correction codes"
                        ]
                    },
                    achievements: [
                        "10x improvement in write endurance",
                        "50% reduction in reset current",
                        "Demonstrated 3D crosspoint array",
                        "Published in Nature Electronics"
                    ]
                }
            },
            
            skills: {
                programming: {
                    expert: ["Python", "Verilog", "SystemVerilog", "C++"],
                    proficient: ["JavaScript", "MATLAB", "TCL", "Bash"],
                    familiar: ["Rust", "Julia", "Scala", "VHDL"]
                },
                
                tools: {
                    eda: ["Cadence Virtuoso", "Synopsys Design Compiler", "Mentor Calibre"],
                    rf: ["Keysight ADS", "ANSYS HFSS", "CST Studio", "AWR"],
                    fpga: ["Vivado", "Quartus", "ModelSim", "VCS"],
                    software: ["Git", "Docker", "TensorFlow", "PyTorch"]
                },
                
                domains: {
                    hardware: ["Digital Design", "Analog/RF", "VLSI", "PCB Design"],
                    software: ["Full-Stack Web", "Embedded", "ML/AI", "DevOps"],
                    research: ["Device Physics", "Cryptography", "Signal Processing", "Computer Architecture"]
                }
            },
            
            experience: {
                current: {
                    position: "RF Design Engineer",
                    company: "Leading Tech Company",
                    duration: "2022 - Present",
                    responsibilities: [
                        "Lead 5G mmWave frontend development",
                        "Design high-efficiency PAs and LNAs",
                        "Implement beamforming algorithms",
                        "Collaborate with system architects"
                    ],
                    achievements: [
                        "Reduced PA power consumption by 30%",
                        "Improved beamforming accuracy to ±1°",
                        "Led team of 5 engineers",
                        "Filed 3 patents"
                    ]
                },
                
                previous: [
                    {
                        position: "Hardware Engineer Intern",
                        company: "Semiconductor Startup",
                        duration: "Summer 2021",
                        highlights: [
                            "Developed RISC-V peripherals",
                            "Implemented UVM testbenches",
                            "Optimized critical paths for timing"
                        ]
                    },
                    {
                        position: "Research Assistant",
                        company: "University Lab",
                        duration: "2020 - 2022",
                        highlights: [
                            "PCM device characterization",
                            "Published 2 papers",
                            "Mentored 3 undergraduates"
                        ]
                    }
                ]
            },
            
            education: {
                graduate: {
                    degree: "M.S. Electrical Engineering",
                    school: "Top Engineering University",
                    gpa: "3.9/4.0",
                    thesis: "Adaptive Beamforming for 5G mmWave Systems",
                    courses: [
                        "Advanced RF Circuit Design",
                        "Digital Signal Processing",
                        "Computer Architecture",
                        "Quantum Computing"
                    ]
                },
                
                undergraduate: {
                    degree: "B.S. Computer Engineering",
                    school: "Engineering University",
                    gpa: "3.8/4.0",
                    honors: ["Summa Cum Laude", "Tau Beta Pi", "IEEE Student Member"],
                    projects: [
                        "FPGA-based Bitcoin Miner",
                        "Autonomous Drone Navigation",
                        "IoT Security Framework"
                    ]
                }
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
            if (lower.includes(key.replace('-', ' ')) || 
                lower.includes(project.title.toLowerCase()) ||
                project.details.applications?.some(app => lower.includes(app.toLowerCase()))) {
                entities.projects.push(key);
            }
        });

        // Check for technology mentions
        const allTechs = [
            ...this.knowledgeBase.skills.programming.expert,
            ...this.knowledgeBase.skills.programming.proficient,
            ...Object.values(this.knowledgeBase.skills.tools).flat()
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

    // Generate project-specific response
    generateProjectResponse(projectKey, knowledge) {
        if (!projectKey || !this.knowledgeBase.projects[projectKey]) {
            return "I can tell you about several projects including the mmWave RF Frontend, RISC-V SoC, Cryptography Research, and Phase Change Memory. Which one interests you most?";
        }
        
        const project = this.knowledgeBase.projects[projectKey];
        
        let response = `Let me tell you about the ${project.title}.\n\n`;
        response += `This ${project.category} project ${project.description}. `;
        
        // Add technical details
        if (project.details) {
            response += `\n\nKey technical aspects include:\n`;
            Object.entries(project.details).forEach(([key, value]) => {
                if (Array.isArray(value)) {
                    response += `• ${key.replace(/_/g, ' ')}: ${value.join(', ')}\n`;
                } else {
                    response += `• ${key.replace(/_/g, ' ')}: ${value}\n`;
                }
            });
        }
        
        // Add challenges and outcomes
        if (project.challenges_solved) {
            response += `\n\nSome of the main challenges I solved were:\n`;
            project.challenges_solved.forEach(challenge => {
                response += `• ${challenge}\n`;
            });
        }
        
        if (project.outcomes) {
            response += `\n\nProject outcomes:\n`;
            project.outcomes.forEach(outcome => {
                response += `• ${outcome}\n`;
            });
        }
        
        response += `\n\nWould you like to know more about the technical implementation, see code examples, or explore the interactive demo?`;
        
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

    // Explain beamforming
    explainBeamforming() {
        return `Beamforming is a signal processing technique I implemented in the mmWave RF Frontend project. Here's how it works:

**Fundamental Principle:**
Beamforming uses an array of antennas to create a directional radiation pattern. By controlling the phase and amplitude of signals at each antenna element, we can steer the beam electronically without mechanical movement.

**My Implementation:**
• 8x8 phased array with λ/2 spacing
• Digital beamforming with 6-bit phase shifters
• Adaptive algorithms for optimal beam steering
• ±60° scanning range in both azimuth and elevation

**Technical Details:**
The array factor is given by: AF(θ,φ) = Σ w_n * exp(j*k*r_n·û)
where w_n are the complex weights, k is the wave number, r_n is the element position, and û is the direction vector.

**Performance Achieved:**
• Beam width: 12° at 28 GHz
• Sidelobe level: -13 dB
• Gain variation: < 3 dB across scan range
• Steering accuracy: ±1°

The key challenge was maintaining phase coherence across all 64 elements while managing mutual coupling effects. I solved this using a calibration algorithm that compensates for element-to-element variations.

Would you like me to explain the calibration process, show you the beamforming code, or discuss the RF frontend architecture?`;
    }

    // Explain pipeline
    explainPipeline() {
        return `The RISC-V processor pipeline is a crucial part of the SoC design. Let me explain the implementation:

**5-Stage Pipeline Architecture:**

1. **IF (Instruction Fetch):**
   - Fetch from I-cache (32KB, 2-way set-associative)
   - Branch prediction using TAGE predictor
   - 4K-entry BTB for target prediction

2. **ID (Instruction Decode):**
   - Decode RV64IMAFDC instructions
   - Register file read (32 integer + 32 floating-point)
   - Dependency checking and scoreboarding

3. **EX (Execute):**
   - ALU operations (ADD, SUB, logical, shifts)
   - Address calculation for loads/stores
   - Branch resolution and misprediction recovery

4. **MEM (Memory Access):**
   - D-cache access (32KB, 4-way set-associative)
   - Store buffer for write combining
   - Load-store unit with disambiguation

5. **WB (Write Back):**
   - Write results to register file
   - Commit to architectural state
   - Exception handling

**Advanced Features:**
• Out-of-order execution with 64-entry ROB
• Register renaming with 96 physical registers
• Speculative execution with precise exceptions
• Forwarding paths to minimize stalls

**Hazard Handling:**
• Data hazards: Forwarding from EX/MEM/WB stages
• Control hazards: Branch prediction + fast recovery
• Structural hazards: Dual-ported register file

**Performance:**
• 2.3 DMIPS/MHz
• 85% branch prediction accuracy
• Average CPI: 1.2 for typical workloads

The most challenging aspect was implementing precise exception handling with out-of-order execution. I used a reorder buffer (ROB) to maintain program order for commits.

Would you like to see the Verilog implementation, understand the hazard detection logic, or explore the branch predictor design?`;
    }

    // Explain cryptography
    explainCryptography() {
        return `My cryptography research focuses on post-quantum cryptography (PQC) - algorithms that remain secure against quantum computers. Here's an overview:

**Why Post-Quantum Cryptography?**
Quantum computers can break current public-key cryptography (RSA, ECC) using Shor's algorithm. We need new mathematical problems that are hard even for quantum computers.

**Algorithms I've Implemented:**

**1. CRYSTALS-Kyber (Key Exchange):**
• Based on Module-LWE problem
• Security levels: 512, 768, 1024 bits
• My optimization: 5x speedup using NTT acceleration
• Hardware: 15K LUTs, 100K ops/sec

**2. CRYSTALS-Dilithium (Digital Signatures):**
• Also Module-LWE based
• Signature size: 2.4KB (compact for PQC)
• Signing: 0.3ms, Verification: 0.1ms
• Implemented constant-time for side-channel resistance

**3. FALCON (Signatures):**
• Based on NTRU lattices
• Smallest signatures but complex implementation
• Uses fast Fourier sampling over lattices

**4. SPHINCS+ (Hash-based):**
• Stateless, based only on hash functions
• Large signatures but minimal assumptions
• Quantum-resistant by design

**Key Innovations:**
• Novel masking scheme for side-channel protection
• Optimized NTT with lazy reduction
• Hardware/software co-design for acceleration
• Hybrid protocols combining classical and PQC

**Security Analysis:**
• Resistance to timing attacks through constant-time implementation
• Power analysis countermeasures using masking
• Fault injection protection with redundancy
• Formal verification of critical components

**Real-world Applications:**
• Secure firmware updates for IoT devices
• Post-quantum TLS for web security
• Blockchain with quantum-resistant signatures
• Hardware security modules (HSM) integration

The main challenge is balancing security, performance, and resource usage. PQC algorithms typically have larger keys and signatures than classical crypto.

Would you like to dive deeper into the mathematical foundations, see implementation code, or discuss practical deployment strategies?`;
    }

    // Explain PCM
    explainPCM() {
        return `Phase Change Memory (PCM) is an emerging non-volatile memory technology I've been researching. It's fascinating because it could replace both DRAM and flash storage:

**How PCM Works:**
PCM uses chalcogenide materials (typically Ge₂Sb₂Te₅ or GST) that can switch between crystalline (low resistance) and amorphous (high resistance) states.

**Physical Mechanism:**
• **SET Operation (Write '1'):** Apply moderate current pulse (~100μA for 100ns) to crystallize
• **RESET Operation (Write '0'):** Apply high current pulse (~500μA for 50ns) to melt and quench
• **READ:** Apply small current (~10μA) to sense resistance

**My Research Contributions:**

**1. Material Optimization:**
• Doped GST with nitrogen for better retention
• Achieved 10 years retention at 85°C
• Reduced reset current by 50% through confined structures

**2. Device Architecture:**
• 4F² cell size using vertical structure
• 3D crosspoint array demonstration
• Multi-level cell (2 bits/cell) with resistance tuning

**3. Circuit Innovations:**
• Adaptive write algorithm reducing energy by 40%
• Wear leveling for 10⁹ cycle endurance
• Error correction for reliability

**Performance Metrics Achieved:**
• Write speed: 50ns (10x faster than NAND)
• Read speed: 10ns (comparable to DRAM)
• Endurance: 10⁹ cycles (1000x better than NAND)
• Density: 4 Gb chip demonstrated

**Key Challenges Solved:**

**1. Thermal Crosstalk:**
Used thermal barriers between cells to prevent adjacent cell disturb

**2. Resistance Drift:**
Developed compensation algorithm for long-term stability

**3. Write Energy:**
Optimized pulse shapes for minimum energy switching

**4. Variability:**
Statistical write-verify algorithm for consistent operation

**Applications:**
• Storage-class memory bridging DRAM-SSD gap
• Neuromorphic computing using analog resistance
• Radiation-hard memory for space applications
• Embedded memory for AI accelerators

The most exciting aspect is PCM's potential for in-memory computing, where we can perform computations directly in the memory array without moving data.

Would you like to explore the device physics in more detail, see the characterization data, or discuss the neuromorphic computing applications?`;
    }

    // Generate comparison response
    generateComparisonResponse(entities, knowledge) {
        if (entities.projects.length >= 2) {
            const proj1 = this.knowledgeBase.projects[entities.projects[0]];
            const proj2 = this.knowledgeBase.projects[entities.projects[1]];
            
            return `Let me compare ${proj1.title} and ${proj2.title}:

**Project Scope:**
• ${proj1.title}: ${proj1.description}
• ${proj2.title}: ${proj2.description}

**Technical Complexity:**
• ${proj1.title}: ${proj1.category} requiring expertise in ${proj1.details.technology || proj1.details.architecture}
• ${proj2.title}: ${proj2.category} requiring expertise in ${proj2.details.architecture || proj2.details.algorithms}

**Key Differentiators:**
• ${proj1.title} focuses on ${Object.keys(proj1.details)[0]}, while ${proj2.title} emphasizes ${Object.keys(proj2.details)[0]}
• Different domains: ${proj1.category} vs ${proj2.category}
• Different skill sets required

**Common Aspects:**
• Both involve complex system design
• Both required extensive verification/validation
• Both resulted in tangible outcomes and publications

Which specific aspects would you like me to compare in more detail?`;
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

    // Generate skills response
    generateSkillsResponse(technologies, knowledge) {
        let response = "I have extensive experience across multiple domains:\n\n";
        
        response += "**Programming Languages:**\n";
        response += `• Expert: ${this.knowledgeBase.skills.programming.expert.join(', ')}\n`;
        response += `• Proficient: ${this.knowledgeBase.skills.programming.proficient.join(', ')}\n`;
        response += `• Familiar: ${this.knowledgeBase.skills.programming.familiar.join(', ')}\n\n`;
        
        response += "**Technical Domains:**\n";
        Object.entries(this.knowledgeBase.skills.domains).forEach(([category, skills]) => {
            response += `• ${category.charAt(0).toUpperCase() + category.slice(1)}: ${skills.join(', ')}\n`;
        });
        
        response += "\n**Tools & Software:**\n";
        Object.entries(this.knowledgeBase.skills.tools).forEach(([category, tools]) => {
            response += `• ${category.toUpperCase()}: ${tools.join(', ')}\n`;
        });
        
        if (technologies && technologies.length > 0) {
            response += `\n\nRegarding ${technologies.join(' and ')}, I have hands-on experience through various projects. Would you like specific examples of how I've used these technologies?`;
        }
        
        return response;
    }

    // Generate contact response
    generateContactResponse() {
        return `I'd be happy to connect with you! Here are the best ways to reach Louis Antoine:

**Professional Contact:**
📧 Email: louis@portfolio.com
💼 LinkedIn: linkedin.com/in/louisantoine
🐙 GitHub: github.com/louisantoine

**Availability:**
• Open to freelance projects in RF design and hardware development
• Available for consulting on cryptography and security implementations
• Interested in collaborative research opportunities
• Happy to discuss full-time positions in cutting-edge technology

**Areas of Interest:**
• 5G/6G wireless systems
• Post-quantum cryptography
• AI hardware acceleration
• Emerging memory technologies

**Response Time:**
Typically respond within 24-48 hours. For urgent matters, please mention it in the subject line.

Feel free to reach out with project proposals, technical questions, or collaboration ideas. I'm always excited to work on challenging problems and innovative solutions!

Would you like to know more about my availability for specific types of projects?`;
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
                text: 'GitHub Repository',
                url: 'https://github.com/louisantoine'
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

    // Provide more details
    provideMoreDetails(topic, knowledge) {
        if (this.knowledgeBase.projects[topic]) {
            const project = this.knowledgeBase.projects[topic];
            let details = `here are additional details about ${project.title}:\n\n`;
            
            if (project.technical_specs) {
                details += "**Technical Specifications:**\n";
                Object.entries(project.technical_specs).forEach(([key, specs]) => {
                    details += `\n${key.replace(/_/g, ' ').toUpperCase()}:\n`;
                    Object.entries(specs).forEach(([spec, value]) => {
                        details += `• ${spec.replace(/_/g, ' ')}: ${value}\n`;
                    });
                });
            }
            
            if (project.verification) {
                details += "\n**Verification & Validation:**\n";
                Object.entries(project.verification).forEach(([key, value]) => {
                    details += `• ${key.replace(/_/g, ' ')}: ${value}\n`;
                });
            }
            
            return details;
        }
        
        return "I can provide more specific details. What aspect would you like me to elaborate on?";
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