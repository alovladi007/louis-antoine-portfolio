/**
 * Tiny non-blocking toast helper. Replaces blocking alert() calls.
 * Usage: csToast('Your message here');
 */
(function () {
  if (window.csToast) return; // idempotent

  // Inject styles once
  var style = document.createElement('style');
  style.textContent =
    '.cs-toast{position:fixed;left:50%;bottom:32px;transform:translateX(-50%) translateY(20px);' +
    'background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:.85rem 1.25rem;' +
    'border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.4);z-index:99999;' +
    "font:500 .95rem/1.4 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;max-width:min(92vw,440px);" +
    'opacity:0;pointer-events:none;transition:opacity .25s ease,transform .25s ease}' +
    '.cs-toast.show{opacity:1;transform:translateX(-50%) translateY(0)}' +
    '.cs-toast i{margin-right:.5rem}';
  document.head.appendChild(style);

  window.csToast = function (msg) {
    var t = document.createElement('div');
    t.className = 'cs-toast';
    t.setAttribute('role', 'status');
    t.setAttribute('aria-live', 'polite');
    t.innerHTML = '<i class="fas fa-rocket"></i>' + msg;
    document.body.appendChild(t);
    requestAnimationFrame(function () {
      t.classList.add('show');
    });
    setTimeout(function () {
      t.classList.remove('show');
      setTimeout(function () {
        t.remove();
      }, 300);
    }, 3500);
  };
})();
