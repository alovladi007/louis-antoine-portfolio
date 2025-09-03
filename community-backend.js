// Community Hub Backend API Simulator
// This simulates a backend API for the community hub
// In production, this would be replaced with actual server-side code

class CommunityBackend {
    constructor() {
        this.initializeData();
        this.setupWebSocket();
    }

    initializeData() {
        // Initialize localStorage data if not exists
        if (!localStorage.getItem('communityDiscussions')) {
            const initialDiscussions = [
                {
                    id: 1,
                    title: "Best practices for RF PCB layout in mmWave designs",
                    author: "Alex Chen",
                    authorId: "user_1",
                    category: "RF/Wireless",
                    content: "I'm working on a 28 GHz transceiver and looking for advice on PCB stackup. Currently considering Rogers RO3003 for the RF layers, but I'm concerned about via transitions at these frequencies. Has anyone successfully implemented blind/buried vias for mmWave applications?",
                    replies: [
                        {
                            id: 1,
                            author: "Sarah Johnson",
                            content: "I've had good success with sequential lamination and laser-drilled microvias. The key is keeping the aspect ratio below 0.8:1",
                            timestamp: new Date(Date.now() - 3600000).toISOString(),
                            likes: 5
                        },
                        {
                            id: 2,
                            author: "Mike Wilson",
                            content: "Consider using via-in-pad with proper filling and planarization. It reduces parasitic inductance significantly.",
                            timestamp: new Date(Date.now() - 7200000).toISOString(),
                            likes: 3
                        }
                    ],
                    views: 156,
                    likes: 42,
                    tags: ["RF", "PCB", "mmWave", "28GHz", "Rogers"],
                    timestamp: new Date(Date.now() - 7200000).toISOString(),
                    pinned: false,
                    locked: false
                },
                {
                    id: 2,
                    title: "Implementing CRYSTALS-Kyber in hardware - optimization tips?",
                    author: "Sarah Johnson",
                    authorId: "user_2",
                    category: "Hardware",
                    content: "Has anyone successfully implemented post-quantum crypto algorithms in FPGA? I'm targeting a Xilinx Ultrascale+ and trying to optimize for throughput. Currently getting about 50 Gbps but I think there's room for improvement.",
                    replies: [],
                    views: 89,
                    likes: 28,
                    tags: ["Cryptography", "FPGA", "Post-Quantum", "Kyber", "Xilinx"],
                    timestamp: new Date(Date.now() - 18000000).toISOString(),
                    pinned: false,
                    locked: false
                },
                {
                    id: 3,
                    title: "RISC-V vs ARM for embedded ML applications",
                    author: "Mike Wilson",
                    authorId: "user_3",
                    category: "Machine Learning",
                    content: "Comparing performance and power efficiency for edge AI deployments. I've been benchmarking both architectures with TensorFlow Lite models. Initial results show RISC-V with vector extensions competitive with ARM Cortex-M55.",
                    replies: [],
                    views: 234,
                    likes: 67,
                    tags: ["RISC-V", "ARM", "ML", "Embedded", "TensorFlow"],
                    timestamp: new Date(Date.now() - 86400000).toISOString(),
                    pinned: true,
                    locked: false
                }
            ];
            localStorage.setItem('communityDiscussions', JSON.stringify(initialDiscussions));
        }

        if (!localStorage.getItem('communityMembers')) {
            const initialMembers = [
                {
                    id: "admin_1",
                    name: "Louis Antoine",
                    email: "louis@example.com",
                    role: "Community Admin",
                    bio: "Electrical Engineer & Software Developer specializing in quantum computing and advanced electronics",
                    posts: 142,
                    reputation: 2847,
                    joined: "2020-01-15",
                    avatar: "LA",
                    badges: ["Founder", "Expert", "Mentor"],
                    skills: ["RF Design", "FPGA", "Machine Learning", "Quantum Computing"],
                    social: {
                        github: "alovladi007",
                        linkedin: "louis-antoine"
                    }
                },
                {
                    id: "user_1",
                    name: "Alex Chen",
                    email: "alex@example.com",
                    role: "RF Expert",
                    bio: "RF/Microwave engineer with 10+ years experience in mmWave systems",
                    posts: 89,
                    reputation: 1523,
                    joined: "2021-03-22",
                    avatar: "AC",
                    badges: ["RF Expert", "Contributor"],
                    skills: ["RF Design", "Antenna Design", "Signal Processing"]
                },
                {
                    id: "user_2",
                    name: "Sarah Johnson",
                    email: "sarah@example.com",
                    role: "Hardware Engineer",
                    bio: "FPGA developer focused on high-speed digital design and cryptography",
                    posts: 67,
                    reputation: 1234,
                    joined: "2021-06-10",
                    avatar: "SJ",
                    badges: ["Hardware Expert", "Security"],
                    skills: ["FPGA", "Verilog", "SystemVerilog", "Cryptography"]
                }
            ];
            localStorage.setItem('communityMembers', JSON.stringify(initialMembers));
        }

        if (!localStorage.getItem('newsletterSubscribers')) {
            localStorage.setItem('newsletterSubscribers', JSON.stringify([]));
        }

        if (!localStorage.getItem('communityEvents')) {
            const initialEvents = [
                {
                    id: 1,
                    title: "Weekly Tech Talk: mmWave Antenna Arrays",
                    description: "Deep dive into phased array design for 5G/6G applications",
                    date: "2024-12-28",
                    time: "15:00",
                    timezone: "EST",
                    type: "Webinar",
                    speaker: "Dr. James Smith",
                    registrations: [],
                    maxAttendees: 100,
                    tags: ["RF", "Antennas", "5G"],
                    meetingLink: "https://meet.example.com/mmwave-talk"
                },
                {
                    id: 2,
                    title: "Community Hackathon: Edge AI Challenge",
                    description: "Build innovative edge AI solutions using RISC-V processors",
                    date: "2025-01-05",
                    time: "10:00",
                    timezone: "EST",
                    type: "Competition",
                    prizes: ["$1000", "$500", "$250"],
                    registrations: [],
                    maxAttendees: 200,
                    tags: ["AI", "RISC-V", "Competition"],
                    registrationDeadline: "2025-01-03"
                }
            ];
            localStorage.setItem('communityEvents', JSON.stringify(initialEvents));
        }
    }

    setupWebSocket() {
        // Simulate WebSocket for real-time features
        this.wsSimulator = {
            listeners: [],
            broadcast: (event, data) => {
                this.wsSimulator.listeners.forEach(listener => {
                    if (listener.event === event) {
                        listener.callback(data);
                    }
                });
            },
            on: (event, callback) => {
                this.wsSimulator.listeners.push({ event, callback });
            }
        };

        // Simulate random activity
        setInterval(() => {
            const events = ['new_message', 'user_joined', 'user_left', 'discussion_updated'];
            const randomEvent = events[Math.floor(Math.random() * events.length)];
            
            switch(randomEvent) {
                case 'new_message':
                    this.wsSimulator.broadcast('chat_message', {
                        author: this.getRandomUser(),
                        message: this.getRandomMessage(),
                        timestamp: new Date().toISOString()
                    });
                    break;
                case 'user_joined':
                    this.wsSimulator.broadcast('user_status', {
                        user: this.getRandomUser(),
                        status: 'online'
                    });
                    break;
                case 'user_left':
                    this.wsSimulator.broadcast('user_status', {
                        user: this.getRandomUser(),
                        status: 'offline'
                    });
                    break;
            }
        }, 15000); // Every 15 seconds
    }

    // API Methods
    
    // Discussions
    getDiscussions(filters = {}) {
        let discussions = JSON.parse(localStorage.getItem('communityDiscussions') || '[]');
        
        // Apply filters
        if (filters.category && filters.category !== 'All Categories') {
            discussions = discussions.filter(d => d.category === filters.category);
        }
        
        if (filters.search) {
            const query = filters.search.toLowerCase();
            discussions = discussions.filter(d => 
                d.title.toLowerCase().includes(query) ||
                d.content.toLowerCase().includes(query) ||
                d.tags.some(tag => tag.toLowerCase().includes(query))
            );
        }
        
        if (filters.sortBy) {
            switch(filters.sortBy) {
                case 'newest':
                    discussions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                    break;
                case 'popular':
                    discussions.sort((a, b) => b.views - a.views);
                    break;
                case 'mostLiked':
                    discussions.sort((a, b) => b.likes - a.likes);
                    break;
            }
        }
        
        // Move pinned to top
        discussions.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));
        
        return discussions;
    }

    getDiscussion(id) {
        const discussions = JSON.parse(localStorage.getItem('communityDiscussions') || '[]');
        const discussion = discussions.find(d => d.id === id);
        
        if (discussion) {
            // Increment views
            discussion.views++;
            localStorage.setItem('communityDiscussions', JSON.stringify(discussions));
        }
        
        return discussion;
    }

    createDiscussion(data) {
        const discussions = JSON.parse(localStorage.getItem('communityDiscussions') || '[]');
        const newDiscussion = {
            id: Date.now(),
            ...data,
            replies: [],
            views: 0,
            likes: 0,
            timestamp: new Date().toISOString(),
            pinned: false,
            locked: false
        };
        
        discussions.unshift(newDiscussion);
        localStorage.setItem('communityDiscussions', JSON.stringify(discussions));
        
        // Broadcast update
        this.wsSimulator.broadcast('discussion_created', newDiscussion);
        
        return newDiscussion;
    }

    addReply(discussionId, reply) {
        const discussions = JSON.parse(localStorage.getItem('communityDiscussions') || '[]');
        const discussion = discussions.find(d => d.id === discussionId);
        
        if (discussion && !discussion.locked) {
            const newReply = {
                id: Date.now(),
                ...reply,
                timestamp: new Date().toISOString(),
                likes: 0
            };
            
            discussion.replies.push(newReply);
            localStorage.setItem('communityDiscussions', JSON.stringify(discussions));
            
            // Broadcast update
            this.wsSimulator.broadcast('reply_added', {
                discussionId,
                reply: newReply
            });
            
            return newReply;
        }
        
        return null;
    }

    likeDiscussion(discussionId, userId) {
        const discussions = JSON.parse(localStorage.getItem('communityDiscussions') || '[]');
        const discussion = discussions.find(d => d.id === discussionId);
        
        if (discussion) {
            // Track likes per user (simplified)
            if (!discussion.likedBy) {
                discussion.likedBy = [];
            }
            
            if (!discussion.likedBy.includes(userId)) {
                discussion.likes++;
                discussion.likedBy.push(userId);
                localStorage.setItem('communityDiscussions', JSON.stringify(discussions));
                return true;
            }
        }
        
        return false;
    }

    // Members
    getMembers() {
        return JSON.parse(localStorage.getItem('communityMembers') || '[]');
    }

    getMember(id) {
        const members = this.getMembers();
        return members.find(m => m.id === id);
    }

    registerMember(data) {
        const members = this.getMembers();
        const newMember = {
            id: `user_${Date.now()}`,
            ...data,
            posts: 0,
            reputation: 0,
            joined: new Date().toISOString().split('T')[0],
            avatar: data.name.substring(0, 2).toUpperCase(),
            badges: ["New Member"],
            skills: []
        };
        
        members.push(newMember);
        localStorage.setItem('communityMembers', JSON.stringify(members));
        
        return newMember;
    }

    updateMemberProfile(id, updates) {
        const members = this.getMembers();
        const memberIndex = members.findIndex(m => m.id === id);
        
        if (memberIndex !== -1) {
            members[memberIndex] = { ...members[memberIndex], ...updates };
            localStorage.setItem('communityMembers', JSON.stringify(members));
            return members[memberIndex];
        }
        
        return null;
    }

    // Newsletter
    subscribeNewsletter(email, name) {
        const subscribers = JSON.parse(localStorage.getItem('newsletterSubscribers') || '[]');
        
        if (!subscribers.find(s => s.email === email)) {
            subscribers.push({
                email,
                name,
                subscribedAt: new Date().toISOString(),
                confirmed: true
            });
            
            localStorage.setItem('newsletterSubscribers', JSON.stringify(subscribers));
            return true;
        }
        
        return false;
    }

    getNewsletterStats() {
        const subscribers = JSON.parse(localStorage.getItem('newsletterSubscribers') || '[]');
        return {
            totalSubscribers: subscribers.length,
            confirmedSubscribers: subscribers.filter(s => s.confirmed).length,
            thisMonth: subscribers.filter(s => {
                const subDate = new Date(s.subscribedAt);
                const now = new Date();
                return subDate.getMonth() === now.getMonth() && 
                       subDate.getFullYear() === now.getFullYear();
            }).length
        };
    }

    // Events
    getEvents() {
        return JSON.parse(localStorage.getItem('communityEvents') || '[]');
    }

    registerForEvent(eventId, userId) {
        const events = this.getEvents();
        const event = events.find(e => e.id === eventId);
        
        if (event && !event.registrations.includes(userId)) {
            if (event.registrations.length < event.maxAttendees) {
                event.registrations.push(userId);
                localStorage.setItem('communityEvents', JSON.stringify(events));
                return true;
            }
        }
        
        return false;
    }

    // Chat
    sendChatMessage(message) {
        // Store chat history
        const chatHistory = JSON.parse(localStorage.getItem('communityChatHistory') || '[]');
        const newMessage = {
            id: Date.now(),
            ...message,
            timestamp: new Date().toISOString()
        };
        
        chatHistory.push(newMessage);
        
        // Keep only last 100 messages
        if (chatHistory.length > 100) {
            chatHistory.shift();
        }
        
        localStorage.setItem('communityChatHistory', JSON.stringify(chatHistory));
        
        // Broadcast message
        this.wsSimulator.broadcast('chat_message', newMessage);
        
        return newMessage;
    }

    getChatHistory() {
        return JSON.parse(localStorage.getItem('communityChatHistory') || '[]');
    }

    // Helper methods
    getRandomUser() {
        const users = ['Alex', 'Sarah', 'Mike', 'Emma', 'John', 'Lisa'];
        return users[Math.floor(Math.random() * users.length)];
    }

    getRandomMessage() {
        const messages = [
            "Great point about the impedance matching!",
            "Has anyone tried this with GaN devices?",
            "I'm seeing similar results in my simulations",
            "The documentation could be clearer on this",
            "Thanks for sharing this resource!",
            "Looking forward to the next meetup",
            "Just pushed an update to the repo",
            "Interesting approach to the problem"
        ];
        return messages[Math.floor(Math.random() * messages.length)];
    }

    // Analytics
    trackEvent(eventName, data) {
        const analytics = JSON.parse(localStorage.getItem('communityAnalytics') || '{}');
        
        if (!analytics[eventName]) {
            analytics[eventName] = [];
        }
        
        analytics[eventName].push({
            ...data,
            timestamp: new Date().toISOString()
        });
        
        localStorage.setItem('communityAnalytics', JSON.stringify(analytics));
    }

    getAnalytics() {
        const analytics = JSON.parse(localStorage.getItem('communityAnalytics') || '{}');
        const discussions = this.getDiscussions();
        const members = this.getMembers();
        const events = this.getEvents();
        
        return {
            totalDiscussions: discussions.length,
            totalMembers: members.length,
            totalEvents: events.length,
            activeDiscussions: discussions.filter(d => {
                const dayAgo = new Date(Date.now() - 86400000);
                return new Date(d.timestamp) > dayAgo;
            }).length,
            popularTags: this.getPopularTags(discussions),
            memberGrowth: this.getMemberGrowth(members),
            engagementRate: this.calculateEngagementRate(discussions, members)
        };
    }

    getPopularTags(discussions) {
        const tagCounts = {};
        
        discussions.forEach(d => {
            d.tags.forEach(tag => {
                tagCounts[tag] = (tagCounts[tag] || 0) + 1;
            });
        });
        
        return Object.entries(tagCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([tag, count]) => ({ tag, count }));
    }

    getMemberGrowth(members) {
        const growth = {};
        
        members.forEach(m => {
            const month = m.joined.substring(0, 7);
            growth[month] = (growth[month] || 0) + 1;
        });
        
        return growth;
    }

    calculateEngagementRate(discussions, members) {
        if (members.length === 0) return 0;
        
        const activeMembers = new Set();
        discussions.forEach(d => {
            activeMembers.add(d.authorId);
            d.replies.forEach(r => activeMembers.add(r.authorId));
        });
        
        return Math.round((activeMembers.size / members.length) * 100);
    }
}

// Initialize backend when script loads
const communityBackend = new CommunityBackend();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CommunityBackend;
}