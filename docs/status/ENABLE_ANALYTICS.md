# How to Enable Analytics on Your Portfolio

I've added analytics tracking code to your website footer. The code is currently commented out. Here's how to activate it:

## Quick Setup Guide

### Option 1: Google Analytics (Recommended - Most Features)

1. **Create a Google Analytics Account:**
   - Go to https://analytics.google.com/
   - Click "Start measuring"
   - Create an account and property for your portfolio
   - Select "Web" as platform
   - Enter your website URL: https://alovladi007.github.io/louis-antoine-portfolio/

2. **Get Your Measurement ID:**
   - After setup, you'll get a Measurement ID like: `G-ABC123XYZ`
   - Copy this ID

3. **Enable in Your Code:**
   - Open `index.html`
   - Find the Google Analytics section (around line 1397-1407)
   - Remove the `<!--` and `-->` comment tags
   - Replace `G-XXXXXXXXXX` with your actual Measurement ID (both occurrences)

### Option 2: Microsoft Clarity (Free Heatmaps & Session Recordings)

1. **Create a Clarity Account:**
   - Go to https://clarity.microsoft.com/
   - Sign up with Microsoft account
   - Create a new project
   - Add your website URL

2. **Get Your Clarity ID:**
   - In your project settings, copy the Project ID

3. **Enable in Your Code:**
   - Open `index.html`
   - Find the Microsoft Clarity section (around line 1409-1419)
   - Remove the `<!--` and `-->` comment tags
   - Replace `YOUR_CLARITY_ID` with your actual Clarity ID

### Option 3: GoatCounter (Privacy-Friendly, Free)

1. **Create GoatCounter Account:**
   - Go to https://www.goatcounter.com/
   - Sign up for free account
   - Choose a subdomain (e.g., louisantoine)

2. **Enable in Your Code:**
   - Open `index.html`
   - Find the GoatCounter section (around line 1421-1426)
   - Remove the `<!--` and `-->` comment tags
   - Replace `YOURUSERNAME` with your GoatCounter subdomain

## What Gets Tracked

The code I've added tracks:
- Page views and unique visitors
- Which projects people click on
- Contact form submissions
- Social media link clicks
- How far people scroll down your page
- Geographic location of visitors
- Device types (mobile/desktop)
- Traffic sources

## After Enabling

1. **Commit and Push Changes:**
   ```bash
   git add index.html
   git commit -m "Add analytics tracking"
   git push origin main
   ```

2. **Wait 5-10 minutes** for GitHub Pages to update

3. **Test It:**
   - Visit your site
   - Check your analytics dashboard
   - You should see your own visit recorded

## Privacy Considerations

- Consider adding a privacy notice to your footer
- GoatCounter is the most privacy-friendly option
- Google Analytics provides the most detailed insights
- You can use multiple analytics tools simultaneously

## Need Help?

If you need assistance setting up any of these services, each platform has excellent documentation:
- [Google Analytics Help](https://support.google.com/analytics)
- [Microsoft Clarity Docs](https://docs.microsoft.com/en-us/clarity/)
- [GoatCounter Docs](https://www.goatcounter.com/help)