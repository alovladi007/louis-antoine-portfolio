import re

# Read the current index.html file
with open('index.html', 'r') as f:
    content = f.read()

# Define the new project card HTML
new_project = '''                <div class="project-card">
                    <div class="project-icon">
                        <i class="fas fa-microscope"></i>
                    </div>
                    <h3>193nm DUV Lithography Optimization</h3>
                    <p>Graduate research project applying Six Sigma methodologies to optimize deep ultraviolet lithography processes. Achieved 42% defect reduction and $14,700 cost savings through advanced statistical analysis and process control.</p>
                    <a href="duv-lithography-project.html" class="project-link">
                        <i class="fas fa-external-link-alt"></i> View Project Details
                    </a>
                </div>

'''

# Find the location to insert the new project (after the last project-card and before the closing </div>)
pattern = r'(.*<div class="project-card">.*?</div>\s*)(</div>\s*<div class="projects-learn-more">)'
match = re.search(pattern, content, re.DOTALL)

if match:
    # Insert the new project before the closing div
    new_content = match.group(1) + new_project + '            ' + match.group(2)
    content = content.replace(match.group(0), new_content)
    
    # Write the updated content back to the file
    with open('index.html', 'w') as f:
        f.write(content)
    
    print("Successfully added DUV project to index.html")
else:
    print("Could not find the insertion point in index.html")
