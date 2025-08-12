# How to Add Interstellar Main Theme to Your Portfolio

## Option 1: YouTube Embed (Currently Implemented)
The portfolio now uses YouTube's API to play the official Interstellar Main Theme. Click the play button in the bottom right corner.

## Option 2: Host Your Own MP3
If you want better control and no dependency on YouTube:

1. **Obtain the MP3 file legally**:
   - Purchase from iTunes, Amazon Music, or Google Play
   - Or use YouTube to MP3 converter for personal use

2. **Add to your repository**:
   ```bash
   # In your local repository
   git add interstellar-theme.mp3
   git commit -m "Add Interstellar theme audio"
   git push origin main
   ```

3. **Update the HTML**:
   Replace the YouTube player code with:
   ```html
   <audio id="background-audio" loop preload="auto">
       <source src="interstellar-theme.mp3" type="audio/mpeg">
   </audio>
   ```

## Current Implementation
The site now uses YouTube's official upload of "Interstellar Main Theme - Hans Zimmer" (video ID: UDVtMYqUAyw).