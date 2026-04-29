#!/bin/bash
cd louis-antoine-portfolio

# Create a backup
cp index.html index_contact_backup.html

# Replace the contact section with a new one that includes a contact form
sed -i.bak3 '/<section id="contact" class="contact">/,/<\/section>/c\
    <!-- Contact Section -->\
    <section id="contact" class="contact">\
        <div class="container">\
            <h2 class="section-title">Get In Touch</h2>\
            <div class="contact-content">\
                <div class="contact-info">\
                    <div class="contact-item">\
                        <i class="fas fa-envelope"></i>\
                        <div>\
                            <h3>Email</h3>\
                            <p>alovladi@gmail.com</p>\
                        </div>\
                    </div>\
                    <div class="contact-item">\
                        <i class="fas fa-phone"></i>\
                        <div>\
                            <h3>Phone</h3>\
                            <p>(203) 360-5619</p>\
                        </div>\
                    </div>\
                    <div class="contact-item">\
                        <i class="fas fa-map-marker-alt"></i>\
                        <div>\
                            <h3>Location</h3>\
                            <p>Connecticut, USA</p>\
                        </div>\
                    </div>\
                </div>\
                <div class="contact-form-container">\
                    <h3>Send me a message</h3>\
                    <form class="contact-form" action="mailto:alovladi@gmail.com" method="post" enctype="text/plain">\
                        <div class="form-group">\
                            <label for="name">Name *</label>\
                            <input type="text" id="name" name="name" required>\
                        </div>\
                        <div class="form-group">\
                            <label for="email">Email *</label>\
                            <input type="email" id="email" name="email" required>\
                        </div>\
                        <div class="form-group">\
                            <label for="company">Company/Organization</label>\
                            <input type="text" id="company" name="company">\
                        </div>\
                        <div class="form-group">\
                            <label for="subject">Subject *</label>\
                            <input type="text" id="subject" name="subject" required>\
                        </div>\
                        <div class="form-group">\
                            <label for="message">Message *</label>\
                            <textarea id="message" name="message" rows="5" required></textarea>\
                        </div>\
                        <button type="submit" class="submit-btn">\
                            <i class="fas fa-paper-plane"></i> Send Message\
                        </button>\
                    </form>\
                </div>\
                <div class="social-links">\
                    <a href="https://www.linkedin.com/in/louis-antoine-333199a0" class="social-link" target="_blank" rel="noopener noreferrer">\
                        <i class="fab fa-linkedin"></i>\
                    </a>\
                    <a href="https://github.com/alovladi007" class="social-link" target="_blank" rel="noopener noreferrer">\
                        <i class="fab fa-github"></i>\
                    </a>\
                    <a href="mailto:alovladi@gmail.com" class="social-link">\
                        <i class="fas fa-envelope"></i>\
                    </a>\
                    <a href="https://www.youtube.com/@AuroraBorealisPhotonics" class="social-link" target="_blank" rel="noopener noreferrer">\
                        <i class="fab fa-youtube"></i>\
                    </a>\
                </div>\
            </div>\
        </div>\
    </section>' index.html

echo "Contact section updated with form"
