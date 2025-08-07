# GitHub Push Permission Setup Guide

## Option 1: Using Personal Access Token (PAT)

### Step 1: Create a Personal Access Token on GitHub

1. Go to GitHub and log in to your account
2. Click on your profile picture (top right) → **Settings**
3. Scroll down to **Developer settings** (bottom of left sidebar)
4. Click on **Personal access tokens** → **Tokens (classic)**
5. Click **Generate new token** → **Generate new token (classic)**
6. Give your token a descriptive name (e.g., "Cursor Bot Access")
7. Set expiration (90 days is reasonable for security)
8. Select scopes:
   - ✅ **repo** (Full control of private repositories)
   - This automatically includes:
     - repo:status
     - repo_deployment
     - public_repo
     - repo:invite
     - security_events
9. Click **Generate token**
10. **IMPORTANT**: Copy the token immediately! You won't be able to see it again.

### Step 2: Configure Git to Use the Token

In the terminal, run these commands:

```bash
cd /workspace/louis-antoine-portfolio

# Set the remote URL with the token
git remote set-url origin https://<YOUR_GITHUB_USERNAME>:<YOUR_TOKEN>@github.com/alovladi007/louis-antoine-portfolio.git

# Example (replace with your actual username and token):
# git remote set-url origin https://alovladi007:ghp_xxxxxxxxxxxxxxxxxxxx@github.com/alovladi007/louis-antoine-portfolio.git
```

### Step 3: Test the Connection

```bash
# This should now work without asking for password
git push origin main
```

## Option 2: Deploy Keys (Repository-Specific)

### Step 1: Generate an SSH Key

```bash
# Generate a new SSH key
ssh-keygen -t ed25519 -C "cursor-bot@louis-antoine-portfolio" -f ~/.ssh/louis_portfolio_key

# Don't set a passphrase (just press Enter twice)
```

### Step 2: Add the Deploy Key to GitHub

1. Copy the public key:
   ```bash
   cat ~/.ssh/louis_portfolio_key.pub
   ```

2. Go to your repository: https://github.com/alovladi007/louis-antoine-portfolio
3. Click **Settings** (repository settings, not profile settings)
4. Click **Deploy keys** (left sidebar)
5. Click **Add deploy key**
6. Title: "Cursor Bot Deploy Key"
7. Paste the public key
8. ✅ Check **Allow write access**
9. Click **Add key**

### Step 3: Configure Git to Use SSH

```bash
# Add SSH key to agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/louis_portfolio_key

# Change remote to use SSH
git remote set-url origin git@github.com:alovladi007/louis-antoine-portfolio.git

# Test the connection
ssh -T git@github.com
```

## Option 3: GitHub App or OAuth (Advanced)

For more complex automation, you can create a GitHub App, but this is overkill for simple push access.

## Security Best Practices

1. **Never commit tokens to the repository**
2. **Use environment variables** for tokens:
   ```bash
   export GITHUB_TOKEN="your_token_here"
   git remote set-url origin https://alovladi007:${GITHUB_TOKEN}@github.com/alovladi007/louis-antoine-portfolio.git
   ```

3. **Rotate tokens regularly**
4. **Use minimal permissions** (only 'repo' scope needed)
5. **Delete tokens when no longer needed**

## Troubleshooting

If you get a 403 error:
- Check token permissions
- Ensure token hasn't expired
- Verify you're using the correct username
- Make sure the repository name is correct

If you get "Support for password authentication was removed":
- You're trying to use your GitHub password instead of a token
- Follow the steps above to create and use a PAT

## Revoking Access

To revoke access later:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click **Delete** next to the token you want to revoke

Or for deploy keys:
1. Go to Repository Settings → Deploy keys
2. Click **Delete** next to the key