# Analytics Setup Guide for Portfolio Website

## Option 1: Google Analytics 4 (Most Popular)

### Pros:
- Free for most use cases
- Detailed visitor insights (location, device, behavior)
- Real-time tracking
- Integration with Google Search Console

### Cons:
- Requires Google account
- Some visitors may block it
- Privacy concerns for some users

### Setup Steps:
1. Go to [Google Analytics](https://analytics.google.com/)
2. Create a new property for your portfolio
3. Get your Measurement ID (looks like: G-XXXXXXXXXX)
4. Add the tracking code to your website (see implementation below)

## Option 2: Plausible Analytics (Privacy-Focused)

### Pros:
- Privacy-friendly (no cookies, GDPR compliant)
- Simple, clean interface
- Lightweight (< 1KB script)
- Less likely to be blocked

### Cons:
- Free trial, then $9/month
- Less detailed than Google Analytics

## Option 3: Umami (Self-Hosted or Cloud)

### Pros:
- Open source and free if self-hosted
- Privacy-focused
- Simple interface
- GDPR compliant

### Cons:
- Requires setup if self-hosting
- Cloud version is paid

## Option 4: GoatCounter (Free & Privacy-Friendly)

### Pros:
- Completely free
- Privacy-focused
- No cookies
- Open source

### Cons:
- Basic features compared to GA
- Simple interface (may be a pro for some)

## Option 5: Microsoft Clarity (Free with Heatmaps)

### Pros:
- Completely free
- Session recordings
- Heatmaps
- No traffic limits

### Cons:
- Microsoft account required
- Can impact performance slightly

## Implementation Code

I'll add the tracking codes to your website. Choose which one(s) you want to use:

### For Google Analytics 4:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### For GoatCounter:
```html
<script data-goatcounter="https://YOURSITENAME.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
```

### For Plausible:
```html
<script defer data-domain="yourdomain.com" src="https://plausible.io/js/script.js"></script>
```

### For Microsoft Clarity:
```html
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "YOUR_CLARITY_ID");
</script>
```

## What You Can Track:

1. **Basic Metrics:**
   - Page views
   - Unique visitors
   - Session duration
   - Bounce rate

2. **User Information:**
   - Geographic location (country/city)
   - Device type (mobile/desktop)
   - Browser and OS
   - Screen resolution

3. **Behavior:**
   - Most visited pages
   - Traffic sources (direct, search, social)
   - User flow through site
   - Click tracking (with additional setup)

4. **Custom Events:**
   - Button clicks
   - Form submissions
   - Download tracking
   - Scroll depth

## Privacy & Legal Considerations:

1. Add a privacy policy if collecting data
2. Consider GDPR compliance if you have EU visitors
3. Some organizations block analytics scripts
4. Be transparent about tracking

## Recommended Setup for Your Portfolio:

I recommend using **both**:
1. **Google Analytics 4** - For detailed insights
2. **GoatCounter or Microsoft Clarity** - As a backup and for additional features

This gives you comprehensive tracking while ensuring you capture data even from privacy-conscious visitors.