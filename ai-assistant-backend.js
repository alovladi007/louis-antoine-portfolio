// AI Assistant Backend - Complete Implementation
// This simulates a full backend with NLP, context management, and knowledge base

class AIAssistantBackend {
    constructor() {
        this.knowledgeBase = this.initializeKnowledgeBase();
        this.conversationHistory = this.loadConversationHistory();
        this.userProfiles = this.loadUserProfiles();
        this.contextMemory = new Map();
        this.activeConnections = new Map();
        this.initializeNLP();
    }

    // Initialize NLP and response generation
    initializeNLP() {
        this.intents = {
            greeting: ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            farewell: ['bye', 'goodbye', 'see you', 'farewell', 'later', 'see ya'],
            thanks: ['thank', 'thanks', 'appreciate', 'grateful'],
            help: ['help', 'assist', 'support', 'guide', 'how to', 'what is', 'explain'],
            projects: ['project', 'portfolio', 'work', 'mmwave', 'risc-v', 'cryptography', 'pcm', 'rf', 'hardware'],
            skills: ['skill', 'technology', 'experience', 'proficiency', 'expertise', 'language', 'framework'],
            contact: ['contact', 'email', 'reach', 'connect', 'hire', 'collaborate'],
            education: ['education', 'degree', 'university', 'study', 'course', 'masters', 'bachelors'],
            experience: ['experience', 'work', 'job', 'position', 'role', 'company'],
            code: ['code', 'example', 'snippet', 'implementation', 'algorithm', 'function'],
            technical: ['technical', 'how does', 'architecture', 'design', 'performance', 'optimization'],
            comparison: ['compare', 'versus', 'vs', 'difference', 'better', 'choose'],
            timeline: ['when', 'timeline', 'schedule', 'deadline', 'duration', 'how long']
        };

        this.sentimentAnalyzer = {
            positive: ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best'],
            negative: ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'disappointing'],
            neutral: ['okay', 'fine', 'alright', 'normal', 'average']
        };
    }

    // Initialize comprehensive knowledge base
    initializeKnowledgeBase() {
        return {
            owner: {
                name: "Louis Antoine",
                role: "Full-Stack Developer & Hardware Engineer",
                email: "louis@portfolio.com",
                location: "United States",
                specializations: ["RF Engineering", "Hardware Design", "Cryptography", "Full-Stack Development"],
                languages: ["Python", "JavaScript", "Verilog", "C++", "MATLAB", "SystemVerilog"],
                clearance: "Security Clearance (if applicable)"
            },
            
            projects: {
                "mmwave-rf": {
                    title: "mmWave RF Frontend Design",
                    category: "RF Semiconductor Design",
                    description: "28 GHz 5G NR n257 band phased array system with beamforming",
                    technologies: ["GaN", "PA", "LNA", "Beamforming", "ADS", "HFSS"],
                    highlights: [
                        "8x8 phased array with ±60° steering",
                        "38 dBm output power with 42% PAE",
                        "1.2 dB noise figure LNA",
                        "Digital predistortion for linearity"
                    ],
                    complexity: "Expert",
                    duration: "6 months",
                    status: "Completed",
                    links: {
                        demo: "mmwave-rf-complete.html",
                        paper: "resources/mmwave/research-paper.pdf",
                        github: "https://github.com/yourusername/mmwave-rf"
                    }
                },
                
                "risc-v-soc": {
                    title: "RISC-V SoC with Custom Accelerators",
                    category: "Hardware/Embedded Systems",
                    description: "64-bit RISC-V processor with ML and crypto accelerators",
                    technologies: ["Verilog", "SystemVerilog", "UVM", "FPGA", "ASIC"],
                    highlights: [
                        "5-stage pipeline with out-of-order execution",
                        "Custom ML accelerator for CNN inference",
                        "Hardware security module with AES/SHA",
                        "2D mesh NoC for scalability"
                    ],
                    complexity: "Expert",
                    duration: "8 months",
                    status: "Completed",
                    links: {
                        demo: "riscv-soc-complete.html",
                        rtl: "resources/riscv/riscv_core.v",
                        github: "https://github.com/yourusername/riscv-soc"
                    }
                },
                
                "cryptography": {
                    title: "Cryptography Research",
                    category: "Security",
                    description: "Post-quantum cryptography and hardware security modules",
                    technologies: ["CRYSTALS-Kyber", "Dilithium", "HSM", "FPGA", "Python"],
                    highlights: [
                        "Implemented NIST PQC algorithms",
                        "Hardware security module design",
                        "Side-channel attack protection",
                        "FIPS 140-3 compliance"
                    ],
                    complexity: "Advanced",
                    duration: "12 months",
                    status: "Ongoing",
                    links: {
                        demo: "cryptography-research.html",
                        paper: "resources/crypto/pqc-research.pdf"
                    }
                },
                
                "pcm": {
                    title: "Phase Change Memory Research",
                    category: "Semiconductor Research",
                    description: "Next-generation non-volatile memory technology",
                    technologies: ["GST", "Material Science", "Device Physics", "TCAD"],
                    highlights: [
                        "10x faster than NAND Flash",
                        "1M write cycles endurance",
                        "Multi-level cell capability",
                        "3D crosspoint architecture"
                    ],
                    complexity: "Research",
                    duration: "18 months",
                    status: "Published"
                }
            },
            
            skills: {
                languages: {
                    expert: ["Python", "JavaScript", "Verilog"],
                    advanced: ["C++", "SystemVerilog", "MATLAB"],
                    intermediate: ["Java", "TypeScript", "R"]
                },
                
                tools: {
                    rf: ["ADS", "HFSS", "CST", "Cadence", "AWR"],
                    hardware: ["Vivado", "Quartus", "ModelSim", "VCS", "Design Compiler"],
                    software: ["React", "Node.js", "TensorFlow", "Docker", "Git"],
                    simulation: ["SPICE", "TCAD", "SystemC", "Simulink"]
                },
                
                domains: {
                    expert: ["RF Design", "Digital Design", "Cryptography"],
                    advanced: ["Machine Learning", "Embedded Systems", "PCB Design"],
                    intermediate: ["Cloud Computing", "DevOps", "Data Science"]
                }
            },
            
            education: [
                {
                    degree: "Master of Science in Electrical Engineering",
                    institution: "Top University",
                    year: "2020-2022",
                    gpa: "3.9/4.0",
                    focus: "RF and Microwave Engineering"
                },
                {
                    degree: "Bachelor of Science in Computer Engineering",
                    institution: "Engineering University",
                    year: "2016-2020",
                    gpa: "3.8/4.0",
                    focus: "Digital Systems and VLSI"
                }
            ],
            
            experience: [
                {
                    position: "RF Design Engineer",
                    company: "Tech Company",
                    duration: "2022-Present",
                    responsibilities: [
                        "5G mmWave frontend design",
                        "Phased array systems",
                        "Power amplifier optimization"
                    ]
                },
                {
                    position: "Hardware Engineer Intern",
                    company: "Semiconductor Corp",
                    duration: "Summer 2021",
                    responsibilities: [
                        "FPGA development",
                        "RTL design and verification",
                        "Hardware acceleration"
                    ]
                }
            ],
            
            faq: {
                "What technologies do you specialize in?": "I specialize in RF engineering (particularly mmWave and 5G), hardware design (RISC-V, FPGA), cryptography (post-quantum), and full-stack development.",
                "Are you available for freelance work?": "Yes, I'm available for consulting and freelance projects in RF design, hardware development, and security implementations.",
                "What's your most complex project?": "The RISC-V SoC with custom accelerators is my most complex project, involving processor design, verification, and hardware acceleration.",
                "Do you have experience with AI/ML?": "Yes, I've implemented ML accelerators in hardware and have experience with TensorFlow and PyTorch for model development.",
                "Can you help with RF simulations?": "Absolutely! I have extensive experience with ADS, HFSS, and CST for RF and microwave simulations."
            },

            codeExamples: {
                python: {
                    rf_calc: `# RF Link Budget Calculator
import numpy as np

def calculate_link_budget(pt_dbm, gt_db, gr_db, freq_ghz, dist_km):
    """Calculate RF link budget for wireless communication"""
    # Free space path loss
    fspl = 20 * np.log10(dist_km) + 20 * np.log10(freq_ghz) + 92.45
    
    # Received power
    pr_dbm = pt_dbm + gt_db + gr_db - fspl
    
    return {
        'transmitted_power': pt_dbm,
        'path_loss': fspl,
        'received_power': pr_dbm,
        'link_margin': pr_dbm + 174 - 10 * np.log10(1e6) - 10  # Assuming 1MHz BW, NF=10dB
    }`,
                    
                    crypto: `# Post-Quantum Key Exchange (simplified)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import secrets

def generate_kyber_keypair(n=256, k=3, q=3329):
    """Generate CRYSTALS-Kyber keypair (simplified)"""
    # Generate random polynomial
    sk = [secrets.randbelow(q) for _ in range(n * k)]
    
    # Public key generation (simplified)
    A = [[secrets.randbelow(q) for _ in range(n)] for _ in range(k * k)]
    pk = matrix_multiply(A, sk, q)
    
    return pk, sk

def encapsulate(pk, n=256, k=3, q=3329):
    """Encapsulate shared secret"""
    m = secrets.token_bytes(32)
    r = [secrets.randbelow(q) for _ in range(n * k)]
    
    # Encryption (simplified)
    c = encrypt_message(pk, m, r, q)
    return c, m`
                },
                
                verilog: {
                    riscv_alu: `// RISC-V ALU Module
module riscv_alu #(
    parameter WIDTH = 64
)(
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    input  [3:0]       alu_op,
    output reg [WIDTH-1:0] result,
    output zero
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
            default: result = 0;
        endcase
    end
    
    assign zero = (result == 0);
endmodule`,
                    
                    cache: `// Cache Controller Module
module cache_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter CACHE_SIZE = 8192,
    parameter LINE_SIZE = 64
)(
    input clk,
    input rst_n,
    input [ADDR_WIDTH-1:0] addr,
    input [DATA_WIDTH-1:0] write_data,
    input read_en,
    input write_en,
    output reg [DATA_WIDTH-1:0] read_data,
    output reg hit,
    output reg miss
);
    // Cache implementation
    localparam NUM_LINES = CACHE_SIZE / LINE_SIZE;
    localparam INDEX_BITS = $clog2(NUM_LINES);
    
    reg [DATA_WIDTH-1:0] cache_data [NUM_LINES-1:0];
    reg [ADDR_WIDTH-INDEX_BITS-1:0] cache_tags [NUM_LINES-1:0];
    reg valid [NUM_LINES-1:0];
    
    wire [INDEX_BITS-1:0] index = addr[INDEX_BITS+1:2];
    wire [ADDR_WIDTH-INDEX_BITS-1:0] tag = addr[ADDR_WIDTH-1:INDEX_BITS+2];
    
    always @(posedge clk) begin
        if (!rst_n) begin
            hit <= 0;
            miss <= 0;
        end else begin
            if (read_en || write_en) begin
                hit <= valid[index] && (cache_tags[index] == tag);
                miss <= !hit;
            end
        end
    end
endmodule`
                }
            }
        };
    }

    // Load conversation history
    loadConversationHistory() {
        const saved = localStorage.getItem('aiConversations');
        return saved ? JSON.parse(saved) : {};
    }

    // Load user profiles
    loadUserProfiles() {
        const saved = localStorage.getItem('aiUserProfiles');
        return saved ? JSON.parse(saved) : {};
    }

    // Save conversation
    saveConversation(userId, conversation) {
        if (!this.conversationHistory[userId]) {
            this.conversationHistory[userId] = [];
        }
        this.conversationHistory[userId].push(conversation);
        
        // Keep only last 100 messages per user
        if (this.conversationHistory[userId].length > 100) {
            this.conversationHistory[userId] = this.conversationHistory[userId].slice(-100);
        }
        
        localStorage.setItem('aiConversations', JSON.stringify(this.conversationHistory));
    }

    // Process user message
    async processMessage(userId, message, context = {}) {
        // Update user profile
        this.updateUserProfile(userId, message);
        
        // Analyze message
        const analysis = this.analyzeMessage(message);
        
        // Get context from history
        const userContext = this.getUserContext(userId);
        
        // Generate response
        const response = await this.generateResponse(message, analysis, userContext, context);
        
        // Save to history
        this.saveConversation(userId, {
            timestamp: new Date().toISOString(),
            user: message,
            assistant: response.text,
            intent: analysis.intent,
            sentiment: analysis.sentiment,
            context: context
        });
        
        return response;
    }

    // Analyze message for intent and entities
    analyzeMessage(message) {
        const lower = message.toLowerCase();
        const words = lower.split(/\s+/);
        
        // Detect intent
        let detectedIntent = 'general';
        let confidence = 0;
        
        for (const [intent, keywords] of Object.entries(this.intents)) {
            const matches = keywords.filter(keyword => lower.includes(keyword)).length;
            const score = matches / keywords.length;
            
            if (score > confidence) {
                confidence = score;
                detectedIntent = intent;
            }
        }
        
        // Detect sentiment
        let sentiment = 'neutral';
        const positiveCount = this.sentimentAnalyzer.positive.filter(word => lower.includes(word)).length;
        const negativeCount = this.sentimentAnalyzer.negative.filter(word => lower.includes(word)).length;
        
        if (positiveCount > negativeCount) sentiment = 'positive';
        else if (negativeCount > positiveCount) sentiment = 'negative';
        
        // Extract entities
        const entities = this.extractEntities(message);
        
        return {
            intent: detectedIntent,
            confidence: confidence,
            sentiment: sentiment,
            entities: entities,
            keywords: words.filter(w => w.length > 3)
        };
    }

    // Extract entities from message
    extractEntities(message) {
        const entities = {
            projects: [],
            technologies: [],
            skills: [],
            timeframes: []
        };
        
        const lower = message.toLowerCase();
        
        // Check for project mentions
        for (const [key, project] of Object.entries(this.knowledgeBase.projects)) {
            if (lower.includes(project.title.toLowerCase()) || 
                lower.includes(key.replace('-', ' '))) {
                entities.projects.push(key);
            }
        }
        
        // Check for technology mentions
        const allTechs = [
            ...this.knowledgeBase.skills.languages.expert,
            ...this.knowledgeBase.skills.languages.advanced,
            ...this.knowledgeBase.skills.languages.intermediate,
            ...Object.values(this.knowledgeBase.skills.tools).flat()
        ];
        
        for (const tech of allTechs) {
            if (lower.includes(tech.toLowerCase())) {
                entities.technologies.push(tech);
            }
        }
        
        // Extract timeframes
        const timePattern = /\d+\s*(days?|weeks?|months?|years?)/gi;
        const matches = message.match(timePattern);
        if (matches) {
            entities.timeframes = matches;
        }
        
        return entities;
    }

    // Get user context from history
    getUserContext(userId) {
        const history = this.conversationHistory[userId] || [];
        const recentHistory = history.slice(-5); // Last 5 messages
        
        const context = {
            previousTopics: [],
            userPreferences: {},
            conversationFlow: [],
            lastIntent: null
        };
        
        // Analyze recent history
        for (const conv of recentHistory) {
            if (conv.intent) {
                context.previousTopics.push(conv.intent);
                context.lastIntent = conv.intent;
            }
            context.conversationFlow.push({
                user: conv.user.substring(0, 50),
                assistant: conv.assistant.substring(0, 50)
            });
        }
        
        // Get user profile
        const profile = this.userProfiles[userId] || {};
        context.userPreferences = profile.preferences || {};
        
        return context;
    }

    // Generate response based on analysis
    async generateResponse(message, analysis, userContext, requestContext) {
        let response = {
            text: '',
            suggestions: [],
            actions: [],
            code: null,
            links: [],
            emotion: 'friendly'
        };
        
        // Handle different intents
        switch(analysis.intent) {
            case 'greeting':
                response = this.handleGreeting(message, userContext);
                break;
                
            case 'farewell':
                response = this.handleFarewell(message, userContext);
                break;
                
            case 'thanks':
                response = this.handleThanks(message, userContext);
                break;
                
            case 'help':
                response = this.handleHelp(message, analysis, userContext);
                break;
                
            case 'projects':
                response = this.handleProjects(message, analysis, userContext);
                break;
                
            case 'skills':
                response = this.handleSkills(message, analysis, userContext);
                break;
                
            case 'contact':
                response = this.handleContact(message, userContext);
                break;
                
            case 'education':
                response = this.handleEducation(message, userContext);
                break;
                
            case 'experience':
                response = this.handleExperience(message, userContext);
                break;
                
            case 'code':
                response = this.handleCode(message, analysis, userContext);
                break;
                
            case 'technical':
                response = this.handleTechnical(message, analysis, userContext);
                break;
                
            case 'comparison':
                response = this.handleComparison(message, analysis, userContext);
                break;
                
            case 'timeline':
                response = this.handleTimeline(message, analysis, userContext);
                break;
                
            default:
                response = this.handleGeneral(message, analysis, userContext);
        }
        
        // Add contextual suggestions
        response.suggestions = this.generateSuggestions(analysis, userContext);
        
        // Add personality to response
        response = this.addPersonality(response, analysis.sentiment);
        
        return response;
    }

    // Handle greeting intent
    handleGreeting(message, context) {
        const greetings = [
            "Hello! Welcome to Louis Antoine's portfolio. How can I help you explore the projects and skills?",
            "Hi there! I'm here to guide you through the portfolio. What would you like to know about?",
            "Greetings! Ready to discover some amazing projects in RF, hardware, and cryptography?",
            "Hello! Great to see you here. Would you like to learn about the mmWave RF frontend, RISC-V SoC, or cryptography research?"
        ];
        
        const timeOfDay = new Date().getHours();
        let greeting = greetings[Math.floor(Math.random() * greetings.length)];
        
        if (timeOfDay < 12) {
            greeting = "Good morning! " + greeting;
        } else if (timeOfDay < 17) {
            greeting = "Good afternoon! " + greeting;
        } else {
            greeting = "Good evening! " + greeting;
        }
        
        return {
            text: greeting,
            suggestions: [
                "Tell me about the projects",
                "What skills does Louis have?",
                "Show me the mmWave RF design",
                "Explain the RISC-V SoC"
            ],
            emotion: 'welcoming'
        };
    }

    // Handle farewell intent
    handleFarewell(message, context) {
        const farewells = [
            "Thank you for visiting! Feel free to return anytime if you have more questions.",
            "Goodbye! Don't hesitate to reach out if you need any information about the portfolio.",
            "See you later! Remember, you can always explore the projects in detail or contact Louis directly.",
            "Farewell! Hope you found the portfolio interesting. Come back anytime!"
        ];
        
        return {
            text: farewells[Math.floor(Math.random() * farewells.length)],
            suggestions: [],
            emotion: 'friendly'
        };
    }

    // Handle thanks intent
    handleThanks(message, context) {
        const responses = [
            "You're welcome! Is there anything else you'd like to know?",
            "My pleasure! Feel free to ask more questions about the projects or skills.",
            "Glad I could help! Would you like to explore any specific project in detail?",
            "Happy to assist! Don't hesitate to ask if you need more information."
        ];
        
        return {
            text: responses[Math.floor(Math.random() * responses.length)],
            suggestions: [
                "Show me another project",
                "How can I contact Louis?",
                "View code examples",
                "Explore skills matrix"
            ],
            emotion: 'appreciative'
        };
    }

    // Handle help intent
    handleHelp(message, analysis, context) {
        const lower = message.toLowerCase();
        
        if (lower.includes('navigate') || lower.includes('use')) {
            return {
                text: `I can help you navigate the portfolio! Here's what you can do:

📂 **Explore Projects**: Ask about mmWave RF, RISC-V SoC, Cryptography, or PCM research
💻 **View Code**: Request code examples in Python, Verilog, or other languages
🎓 **Learn About Skills**: Discover technical skills and expertise areas
📧 **Contact Info**: Get contact details for collaboration
🔍 **Search Topics**: Ask about specific technologies or concepts

What would you like to explore first?`,
                suggestions: [
                    "Show me all projects",
                    "What's the most complex project?",
                    "View RF design code",
                    "Contact information"
                ],
                emotion: 'helpful'
            };
        }
        
        // Check for specific help topics
        if (analysis.entities.projects.length > 0) {
            const project = this.knowledgeBase.projects[analysis.entities.projects[0]];
            return {
                text: `I'll help you with ${project.title}. This project involves ${project.description}. 

Key highlights:
${project.highlights.map(h => `• ${h}`).join('\n')}

Would you like to see the demo, read the documentation, or view the code?`,
                links: Object.entries(project.links).map(([key, url]) => ({
                    text: key.charAt(0).toUpperCase() + key.slice(1),
                    url: url
                })),
                suggestions: [
                    "Show me the demo",
                    "View source code",
                    "Technical details",
                    "Similar projects"
                ],
                emotion: 'informative'
            };
        }
        
        return {
            text: "I'm here to help! You can ask me about projects, skills, experience, or any technical topics. What specific information are you looking for?",
            suggestions: [
                "Overview of all projects",
                "Technical skills",
                "How to get started",
                "Contact Louis"
            ],
            emotion: 'supportive'
        };
    }

    // Handle projects intent
    handleProjects(message, analysis, context) {
        const lower = message.toLowerCase();
        
        // Check for specific project
        if (analysis.entities.projects.length > 0) {
            const projectKey = analysis.entities.projects[0];
            const project = this.knowledgeBase.projects[projectKey];
            
            return {
                text: `## ${project.title}

${project.description}

**Category**: ${project.category}
**Complexity**: ${project.complexity}
**Duration**: ${project.duration}
**Status**: ${project.status}

**Technologies Used**:
${project.technologies.map(t => `• ${t}`).join('\n')}

**Key Achievements**:
${project.highlights.map(h => `• ${h}`).join('\n')}

Would you like to explore the interactive demo or see code examples?`,
                links: project.links ? Object.entries(project.links).map(([key, url]) => ({
                    text: key.charAt(0).toUpperCase() + key.slice(1),
                    url: url
                })) : [],
                suggestions: [
                    "Show me the demo",
                    "View source code",
                    "Compare with other projects",
                    "Technical deep-dive"
                ],
                emotion: 'enthusiastic'
            };
        }
        
        // Show all projects
        return {
            text: `Here are the main projects in the portfolio:

🔬 **mmWave RF Frontend Design**
28 GHz 5G phased array system with beamforming capabilities

💻 **RISC-V SoC with Custom Accelerators**
64-bit processor with ML and crypto acceleration

🔐 **Cryptography Research**
Post-quantum algorithms and hardware security modules

🧪 **Phase Change Memory Research**
Next-generation non-volatile memory technology

Which project would you like to explore in detail?`,
            suggestions: [
                "Tell me about mmWave RF",
                "Explain RISC-V SoC",
                "Cryptography details",
                "PCM research"
            ],
            emotion: 'informative'
        };
    }

    // Handle skills intent
    handleSkills(message, analysis, context) {
        const lower = message.toLowerCase();
        
        // Check for specific technology
        if (analysis.entities.technologies.length > 0) {
            const tech = analysis.entities.technologies[0];
            let level = 'intermediate';
            let category = 'general';
            
            // Find skill level
            for (const [lvl, techs] of Object.entries(this.knowledgeBase.skills.languages)) {
                if (techs.includes(tech)) {
                    level = lvl;
                    category = 'programming';
                    break;
                }
            }
            
            return {
                text: `**${tech}** - ${level.charAt(0).toUpperCase() + level.slice(1)} Level

I have extensive experience with ${tech} in ${category} applications. This skill has been applied in various projects including:

${this.findProjectsUsingTech(tech).map(p => `• ${p.title}`).join('\n')}

Would you like to see code examples or learn about related technologies?`,
                code: this.getCodeExample(tech),
                suggestions: [
                    `Show ${tech} code examples`,
                    "Related technologies",
                    "Projects using this",
                    "Learning resources"
                ],
                emotion: 'confident'
            };
        }
        
        return {
            text: `## Technical Skills Overview

**Programming Languages**:
• Expert: ${this.knowledgeBase.skills.languages.expert.join(', ')}
• Advanced: ${this.knowledgeBase.skills.languages.advanced.join(', ')}

**Domains**:
• RF & Microwave Design
• Digital Hardware Design
• Cryptography & Security
• Machine Learning
• Embedded Systems

**Tools & Software**:
• RF: ADS, HFSS, CST
• Hardware: Vivado, ModelSim, Design Compiler
• Software: React, Node.js, Docker

What specific skill area interests you?`,
            suggestions: [
                "RF design expertise",
                "Hardware skills",
                "Programming languages",
                "View skills matrix"
            ],
            emotion: 'professional'
        };
    }

    // Handle code examples
    handleCode(message, analysis, context) {
        const lower = message.toLowerCase();
        let language = 'python'; // default
        let example = null;
        
        if (lower.includes('verilog') || lower.includes('rtl') || lower.includes('hardware')) {
            language = 'verilog';
            if (lower.includes('cache')) {
                example = this.knowledgeBase.codeExamples.verilog.cache;
            } else {
                example = this.knowledgeBase.codeExamples.verilog.riscv_alu;
            }
        } else if (lower.includes('rf') || lower.includes('link')) {
            example = this.knowledgeBase.codeExamples.python.rf_calc;
        } else if (lower.includes('crypto') || lower.includes('quantum')) {
            example = this.knowledgeBase.codeExamples.python.crypto;
        }
        
        if (example) {
            return {
                text: `Here's a code example that might help:`,
                code: {
                    language: language,
                    content: example
                },
                suggestions: [
                    "Explain this code",
                    "Show another example",
                    "Related projects",
                    "Download full source"
                ],
                emotion: 'helpful'
            };
        }
        
        return {
            text: "I can show you code examples in Python, Verilog, JavaScript, and more. What specific functionality or language are you interested in?",
            suggestions: [
                "Python RF calculations",
                "Verilog ALU design",
                "Cryptography implementation",
                "Cache controller code"
            ],
            emotion: 'eager'
        };
    }

    // Handle technical questions
    handleTechnical(message, analysis, context) {
        const lower = message.toLowerCase();
        
        if (lower.includes('beamform')) {
            return {
                text: `## Beamforming Technology

Beamforming is a signal processing technique used in the mmWave RF project for directional signal transmission/reception.

**How it works**:
1. **Phase Array**: 8x8 antenna elements work together
2. **Phase Shifting**: Each element's phase is adjusted to steer the beam
3. **Constructive Interference**: Signals combine to focus energy in desired direction
4. **Steering Range**: ±60° in both azimuth and elevation

**Benefits**:
• Increased signal strength in target direction
• Reduced interference
• Better range and data rates
• Multiple user support (MU-MIMO)

The implementation uses digital beamforming with adaptive algorithms for optimal performance.`,
                links: [{
                    text: "View Interactive Demo",
                    url: "mmwave-rf-complete.html#beamforming"
                }],
                suggestions: [
                    "Show beamforming code",
                    "RF frontend architecture",
                    "Performance metrics",
                    "5G applications"
                ],
                emotion: 'educational'
            };
        }
        
        if (lower.includes('risc-v') || lower.includes('pipeline')) {
            return {
                text: `## RISC-V Pipeline Architecture

The RISC-V SoC implements a sophisticated 5-stage pipeline:

**Pipeline Stages**:
1. **IF (Instruction Fetch)**: Fetch instructions from memory
2. **ID (Instruction Decode)**: Decode and read registers
3. **EX (Execute)**: ALU operations and address calculation
4. **MEM (Memory)**: Load/store operations
5. **WB (Write Back)**: Write results to registers

**Advanced Features**:
• **Out-of-Order Execution**: Maximize instruction throughput
• **Branch Prediction**: 2-bit saturating counter predictor
• **Hazard Detection**: Data and control hazard handling
• **Forwarding Paths**: Reduce pipeline stalls

The design achieves 1.2 GHz operation on 28nm technology.`,
                code: {
                    language: 'verilog',
                    content: this.knowledgeBase.codeExamples.verilog.riscv_alu
                },
                suggestions: [
                    "View RTL code",
                    "Pipeline visualization",
                    "Performance benchmarks",
                    "Custom accelerators"
                ],
                emotion: 'technical'
            };
        }
        
        return {
            text: "I can explain technical concepts from any of the projects. What specific technology or concept would you like to understand better?",
            suggestions: [
                "How does beamforming work?",
                "RISC-V pipeline explained",
                "Post-quantum cryptography",
                "Phase change memory physics"
            ],
            emotion: 'curious'
        };
    }

    // Handle comparison requests
    handleComparison(message, analysis, context) {
        if (analysis.entities.projects.length >= 2) {
            const proj1 = this.knowledgeBase.projects[analysis.entities.projects[0]];
            const proj2 = this.knowledgeBase.projects[analysis.entities.projects[1]];
            
            return {
                text: `## Comparing Projects

**${proj1.title}** vs **${proj2.title}**

| Aspect | ${proj1.title} | ${proj2.title} |
|--------|---------------|---------------|
| Category | ${proj1.category} | ${proj2.category} |
| Complexity | ${proj1.complexity} | ${proj2.complexity} |
| Duration | ${proj1.duration} | ${proj2.duration} |
| Technologies | ${proj1.technologies.slice(0,3).join(', ')} | ${proj2.technologies.slice(0,3).join(', ')} |
| Status | ${proj1.status} | ${proj2.status} |

Both projects demonstrate different aspects of expertise - ${proj1.title} focuses on ${proj1.category.toLowerCase()}, while ${proj2.title} showcases ${proj2.category.toLowerCase()} skills.`,
                links: [{
                    text: "Interactive Comparison Tool",
                    url: "project-comparison.html"
                }],
                suggestions: [
                    "Detailed comparison",
                    "Technical differences",
                    "Which is more complex?",
                    "View both demos"
                ],
                emotion: 'analytical'
            };
        }
        
        return {
            text: "I can compare different projects, technologies, or approaches. What would you like to compare?",
            suggestions: [
                "Compare RF vs Hardware projects",
                "Python vs Verilog skills",
                "Project complexities",
                "Technology stacks"
            ],
            emotion: 'helpful'
        };
    }

    // Handle contact requests
    handleContact(message, context) {
        return {
            text: `## Contact Information

📧 **Email**: louis@portfolio.com
💼 **LinkedIn**: [Connect on LinkedIn](https://linkedin.com/in/louisantoine)
🐙 **GitHub**: [View GitHub Profile](https://github.com/louisantoine)
🌐 **Portfolio**: You're already here!

**Available for**:
• Freelance RF design projects
• Hardware development consulting
• Cryptography implementations
• Technical collaborations

Feel free to reach out for project discussions, collaborations, or opportunities!`,
            actions: [{
                type: 'email',
                data: 'louis@portfolio.com'
            }],
            suggestions: [
                "Schedule a meeting",
                "View resume",
                "Collaboration areas",
                "Download contact card"
            ],
            emotion: 'professional'
        };
    }

    // Handle education queries
    handleEducation(message, context) {
        const education = this.knowledgeBase.education;
        
        return {
            text: `## Educational Background

${education.map(edu => `**${edu.degree}**
${edu.institution} | ${edu.year}
GPA: ${edu.gpa}
Focus: ${edu.focus}`).join('\n\n')}

**Key Coursework**:
• Advanced RF/Microwave Engineering
• VLSI Design
• Computer Architecture
• Cryptography & Network Security
• Machine Learning

The combination of electrical and computer engineering provides a unique perspective for hardware-software co-design.`,
            links: [{
                text: "View Full Academic Record",
                url: "masters-coursework.html"
            }],
            suggestions: [
                "Research projects",
                "Relevant coursework",
                "Academic achievements",
                "Thesis work"
            ],
            emotion: 'proud'
        };
    }

    // Handle experience queries
    handleExperience(message, context) {
        const experience = this.knowledgeBase.experience;
        
        return {
            text: `## Professional Experience

${experience.map(exp => `**${exp.position}**
${exp.company} | ${exp.duration}

Key Responsibilities:
${exp.responsibilities.map(r => `• ${r}`).join('\n')}`).join('\n\n')}

**Industry Expertise**:
• 5G/6G wireless systems
• High-frequency circuit design
• FPGA development
• Hardware acceleration
• Security implementations

Each role has contributed to the diverse skill set reflected in the portfolio projects.`,
            suggestions: [
                "Specific role details",
                "Key achievements",
                "Technologies used",
                "Industry projects"
            ],
            emotion: 'accomplished'
        };
    }

    // Handle timeline queries
    handleTimeline(message, analysis, context) {
        const lower = message.toLowerCase();
        
        if (analysis.entities.projects.length > 0) {
            const project = this.knowledgeBase.projects[analysis.entities.projects[0]];
            return {
                text: `**${project.title}** was completed over ${project.duration} and is currently ${project.status.toLowerCase()}.

For new projects, typical timelines are:
• Small projects: 1-2 months
• Medium complexity: 3-6 months
• Large/Research projects: 6-12+ months

The timeline depends on scope, complexity, and resource availability.`,
                suggestions: [
                    "Project availability",
                    "Freelance timeline",
                    "Consultation schedule",
                    "Project phases"
                ],
                emotion: 'informative'
            };
        }
        
        return {
            text: "Timeline depends on the project scope. Could you specify what you're interested in - project completion times, availability for new work, or development schedules?",
            suggestions: [
                "Project durations",
                "Current availability",
                "Development timeline",
                "Milestone planning"
            ],
            emotion: 'helpful'
        };
    }

    // Handle general queries
    handleGeneral(message, analysis, context) {
        // Check FAQ
        for (const [question, answer] of Object.entries(this.knowledgeBase.faq)) {
            if (this.similarity(message.toLowerCase(), question.toLowerCase()) > 0.6) {
                return {
                    text: answer,
                    suggestions: [
                        "Tell me more",
                        "Related projects",
                        "Other questions",
                        "Contact info"
                    ],
                    emotion: 'informative'
                };
            }
        }
        
        // Default response with guidance
        return {
            text: `I understand you're asking about "${message}". Let me help you find the right information.

Here are some areas I can assist with:
• **Projects**: mmWave RF, RISC-V SoC, Cryptography, PCM
• **Technical Skills**: Programming, hardware design, RF engineering
• **Code Examples**: Python, Verilog, algorithms
• **Career**: Experience, education, availability

What specific aspect would you like to explore?`,
            suggestions: [
                "Show all projects",
                "Technical skills",
                "View demos",
                "Contact Louis"
            ],
            emotion: 'thoughtful'
        };
    }

    // Generate contextual suggestions
    generateSuggestions(analysis, context) {
        const suggestions = [];
        
        // Based on intent
        switch(analysis.intent) {
            case 'projects':
                suggestions.push("View interactive demos", "Compare projects", "Technical details");
                break;
            case 'skills':
                suggestions.push("Code examples", "Skill matrix", "Certifications");
                break;
            case 'technical':
                suggestions.push("Deep dive", "Implementation details", "Performance metrics");
                break;
            default:
                suggestions.push("Explore projects", "View skills", "Contact info");
        }
        
        // Based on conversation history
        if (context.lastIntent && context.lastIntent !== analysis.intent) {
            suggestions.push(`Back to ${context.lastIntent}`);
        }
        
        return suggestions.slice(0, 4); // Max 4 suggestions
    }

    // Add personality to responses
    addPersonality(response, sentiment) {
        // Add emojis based on emotion
        const emojis = {
            'friendly': '😊',
            'helpful': '💡',
            'enthusiastic': '🚀',
            'professional': '💼',
            'technical': '⚙️',
            'proud': '🎓',
            'welcoming': '👋',
            'analytical': '📊',
            'educational': '📚',
            'thoughtful': '🤔'
        };
        
        if (response.emotion && emojis[response.emotion]) {
            response.text = emojis[response.emotion] + ' ' + response.text;
        }
        
        // Adjust tone based on sentiment
        if (sentiment === 'positive') {
            response.text += "\n\nI'm glad you're interested! Let me know if you need anything else.";
        } else if (sentiment === 'negative') {
            response.text += "\n\nI'm here to help resolve any concerns. What can I clarify for you?";
        }
        
        return response;
    }

    // Helper: Find projects using a technology
    findProjectsUsingTech(tech) {
        const projects = [];
        for (const [key, project] of Object.entries(this.knowledgeBase.projects)) {
            if (project.technologies.some(t => t.toLowerCase().includes(tech.toLowerCase()))) {
                projects.push(project);
            }
        }
        return projects;
    }

    // Helper: Get code example for technology
    getCodeExample(tech) {
        const lower = tech.toLowerCase();
        
        if (lower.includes('python')) {
            return {
                language: 'python',
                content: this.knowledgeBase.codeExamples.python.rf_calc
            };
        } else if (lower.includes('verilog')) {
            return {
                language: 'verilog',
                content: this.knowledgeBase.codeExamples.verilog.riscv_alu
            };
        }
        
        return null;
    }

    // Helper: Calculate string similarity
    similarity(str1, str2) {
        const longer = str1.length > str2.length ? str1 : str2;
        const shorter = str1.length > str2.length ? str2 : str1;
        
        if (longer.length === 0) return 1.0;
        
        const editDistance = this.levenshteinDistance(longer, shorter);
        return (longer.length - editDistance) / longer.length;
    }

    // Helper: Levenshtein distance
    levenshteinDistance(str1, str2) {
        const matrix = [];
        
        for (let i = 0; i <= str2.length; i++) {
            matrix[i] = [i];
        }
        
        for (let j = 0; j <= str1.length; j++) {
            matrix[0][j] = j;
        }
        
        for (let i = 1; i <= str2.length; i++) {
            for (let j = 1; j <= str1.length; j++) {
                if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }
        
        return matrix[str2.length][str1.length];
    }

    // Update user profile based on interactions
    updateUserProfile(userId, message) {
        if (!this.userProfiles[userId]) {
            this.userProfiles[userId] = {
                firstSeen: new Date().toISOString(),
                messageCount: 0,
                interests: [],
                preferences: {}
            };
        }
        
        const profile = this.userProfiles[userId];
        profile.messageCount++;
        profile.lastSeen = new Date().toISOString();
        
        // Track interests
        const analysis = this.analyzeMessage(message);
        if (analysis.intent !== 'general') {
            if (!profile.interests.includes(analysis.intent)) {
                profile.interests.push(analysis.intent);
            }
        }
        
        // Save profile
        localStorage.setItem('aiUserProfiles', JSON.stringify(this.userProfiles));
    }

    // Export conversation history
    exportConversation(userId, format = 'json') {
        const history = this.conversationHistory[userId] || [];
        
        if (format === 'json') {
            return JSON.stringify(history, null, 2);
        } else if (format === 'text') {
            return history.map(conv => 
                `[${conv.timestamp}]\nUser: ${conv.user}\nAssistant: ${conv.assistant}\n`
            ).join('\n---\n');
        }
        
        return history;
    }

    // Clear conversation history
    clearHistory(userId) {
        if (userId) {
            delete this.conversationHistory[userId];
        } else {
            this.conversationHistory = {};
        }
        localStorage.setItem('aiConversations', JSON.stringify(this.conversationHistory));
    }

    // Get statistics
    getStatistics(userId) {
        const history = this.conversationHistory[userId] || [];
        const profile = this.userProfiles[userId] || {};
        
        return {
            totalMessages: history.length,
            firstInteraction: profile.firstSeen,
            lastInteraction: profile.lastSeen,
            topInterests: profile.interests || [],
            averageResponseTime: '< 1 second',
            satisfactionScore: '98%' // Simulated
        };
    }
}

// Initialize and export
if (typeof window !== 'undefined') {
    window.AIAssistantBackend = AIAssistantBackend;
}