/**
 * Floating "Ask the AI Assistant" chat bubble.
 * Auto-injected near the bottom-right of any page that loads this script.
 * Hides itself on the chatbot pages so it doesn't loop.
 *
 * Usage: <script src="<relative-path>/assets/js/chatbot-bubble.js"></script>
 *
 * Path resolution: the script reads its own <script src=...> attribute
 * to figure out where the repo root is, so the same file works from any
 * folder depth. No build step needed.
 */
(function () {
  if (window.__cbBubbleLoaded) return;
  window.__cbBubbleLoaded = true;

  // Skip on the chatbot pages themselves (and on the redirect stub)
  var here = (location.pathname || '').toLowerCase();
  if (
    here.endsWith('/ai-assistant-enhanced.html') ||
    here.endsWith('/chatbot.html')
  ) {
    return;
  }

  function init() {
    // Find this script tag to derive the repo root prefix.
    var src = '';
    var scripts = document.getElementsByTagName('script');
    for (var i = 0; i < scripts.length; i++) {
      var s = scripts[i].getAttribute('src') || '';
      if (s.indexOf('chatbot-bubble.js') !== -1) {
        src = s;
        break;
      }
    }
    // src looks like '../../assets/js/chatbot-bubble.js' or 'assets/js/chatbot-bubble.js'
    var prefix = src.replace(/assets\/js\/chatbot-bubble\.js.*$/, '');
    var target = prefix + 'projects/ml-ai/ai-assistant-enhanced.html';

    // Detect a live-chat widget (Tawk.to in this site, but we also check
    // a few other common vendors so the bubble plays nicely if any of them
    // get added later). When detected, the AI bubble lifts above the live
    // chat button so the two don't overlap.
    function liveChatPresent() {
      return Boolean(
        window.Tawk_API ||
        window.$crisp || window.CRISP_WEBSITE_ID ||
        window.intercomSettings || window.Intercom ||
        window.tidioChatApi ||
        window.HubSpotConversations ||
        window.zE
      );
    }

    // Inject styles once. We use CSS classes to switch position so we don't
    // have to rebuild the element when the live-chat widget loads asynchronously.
    var style = document.createElement('style');
    style.textContent =
      '.cb-bubble{position:fixed;right:24px;bottom:24px;z-index:99998;' +
      'display:inline-flex;align-items:center;gap:.6rem;padding:.85rem 1.15rem;' +
      'background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;' +
      "font:600 .95rem/1 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;" +
      'border-radius:999px;text-decoration:none;border:0;cursor:pointer;' +
      'box-shadow:0 10px 30px rgba(102,126,234,.45),0 4px 10px rgba(0,0,0,.3);' +
      'transition:transform .2s ease,box-shadow .2s ease,opacity .2s ease,bottom .25s ease;' +
      'opacity:0;transform:translateY(12px) scale(.96);pointer-events:auto;}' +
      '.cb-bubble.cb-show{opacity:1;transform:translateY(0) scale(1);}' +
      // When a live-chat widget is present, lift the AI bubble above its
      // ~70px floating button (Tawk default) so the two stack vertically
      // with a small gap. On narrow screens a smaller offset still clears.
      '.cb-bubble.cb-with-livechat{bottom:100px;}' +
      '@media (max-width:480px){.cb-bubble.cb-with-livechat{bottom:88px;}}' +
      '.cb-bubble:hover{transform:translateY(-2px) scale(1.03);' +
      'box-shadow:0 14px 40px rgba(102,126,234,.55),0 4px 12px rgba(0,0,0,.35);}' +
      '.cb-bubble:focus-visible{outline:3px solid #fff;outline-offset:3px;}' +
      '.cb-bubble i{font-size:1.05rem;}' +
      '.cb-bubble .cb-pulse{position:absolute;top:-2px;right:-2px;width:10px;height:10px;' +
      'background:#22c55e;border-radius:50%;border:2px solid #0a0a0a;}' +
      '@media (max-width:480px){.cb-bubble{padding:.85rem;border-radius:50%;}' +
      '.cb-bubble .cb-label{display:none;}}';
    document.head.appendChild(style);

    // Build the link
    var a = document.createElement('a');
    a.className = 'cb-bubble';
    a.href = target;
    a.setAttribute('aria-label', 'Ask the AI Assistant');
    a.title = 'Ask the AI Assistant';
    a.innerHTML =
      '<span class="cb-pulse" aria-hidden="true"></span>' +
      '<i class="fas fa-robot" aria-hidden="true"></i>' +
      '<span class="cb-label" data-i18n="chatbot.bubble">Ask the AI Assistant</span>';

    document.body.appendChild(a);
    requestAnimationFrame(function () {
      a.classList.add('cb-show');
    });

    // Apply or remove the lift class based on whether a live-chat widget
    // is loaded. Live-chat scripts (Tawk, Crisp, etc.) are async, so we
    // poll briefly. Once detected we stop polling.
    function syncPosition() {
      a.classList.toggle('cb-with-livechat', liveChatPresent());
    }
    syncPosition();
    var attempts = 0;
    var poll = setInterval(function () {
      attempts++;
      if (liveChatPresent()) {
        a.classList.add('cb-with-livechat');
        clearInterval(poll);
      } else if (attempts > 20) {
        // ~10s with 500ms cadence — give up; chat probably isn't on this page.
        clearInterval(poll);
      }
    }, 500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
