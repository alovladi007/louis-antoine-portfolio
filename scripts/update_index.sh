#!/bin/bash
cd louis-antoine-portfolio

# Add Graduate Research Assistant link
sed -i.bak1 's|<p>Supported semiconductor and optoelectronics labs with precision measurements, front-end and back-end processing technologies. Provided guidance to undergraduate students on independent research projects.</p>|<p>Supported semiconductor and optoelectronics labs with precision measurements, front-end and back-end processing technologies. Provided guidance to undergraduate students on independent research projects.</p>\
                        <a href="graduate-research-experience.html" class="learn-more-btn">\
                            <i class="fas fa-flask"></i> Learn More About This Role\
                        </a>|' index.html

# Add Patient Administration Specialist link
sed -i.bak2 's|<p>Utilized computerized RPMS and EHR systems for patient records management. Maintained strict HIPAA compliance procedures and prevented information breaches.</p>|<p>Utilized computerized RPMS and EHR systems for patient records management. Maintained strict HIPAA compliance procedures and prevented information breaches.</p>\
                        <a href="patient-admin-experience.html" class="learn-more-btn">\
                            <i class="fas fa-user-md"></i> Learn More About This Role\
                        </a>|' index.html

echo "Index.html updated successfully"
