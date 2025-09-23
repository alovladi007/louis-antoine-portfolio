/* global fetch */

(function () {
  const backendUrlInput = document.getElementById('backendUrl');
  const imageInput = document.getElementById('imageInput');
  const runBtn = document.getElementById('runBtn');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const jsonOut = document.getElementById('jsonOut');

  let imageBitmap = null;

  imageInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    imageBitmap = await createImageBitmap(file);
    drawBaseImage();
  });

  runBtn.addEventListener('click', async () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
      alert('Please select an image first.');
      return;
    }
    const endpoint = backendUrlInput.value || 'http://localhost:5000/process-image';
    const formData = new FormData();
    formData.append('image', file, file.name);
    try {
      const resp = await fetch(endpoint, { method: 'POST', body: formData });
      const data = await resp.json();
      jsonOut.textContent = JSON.stringify(data, null, 2);
      drawDetections(data);
    } catch (err) {
      jsonOut.textContent = 'Request failed: ' + (err && err.message ? err.message : String(err));
    }
  });

  function drawBaseImage() {
    if (!imageBitmap) return;
    const scale = Math.min(canvas.width / imageBitmap.width, canvas.height / imageBitmap.height);
    const drawWidth = Math.floor(imageBitmap.width * scale);
    const drawHeight = Math.floor(imageBitmap.height * scale);
    const offsetX = Math.floor((canvas.width - drawWidth) / 2);
    const offsetY = Math.floor((canvas.height - drawHeight) / 2);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageBitmap, offsetX, offsetY, drawWidth, drawHeight);
    return { scale, offsetX, offsetY };
  }

  function drawDetections(data) {
    const base = drawBaseImage();
    if (!base) return;

    // Draw lanes in blue
    ctx.strokeStyle = 'rgba(0, 102, 204, 1)';
    ctx.lineWidth = 2;
    if (data && Array.isArray(data.lanes)) {
      for (const seg of data.lanes) {
        const x1 = base.offsetX + Math.floor(seg.x1 * base.scale);
        const y1 = base.offsetY + Math.floor(seg.y1 * base.scale);
        const x2 = base.offsetX + Math.floor(seg.x2 * base.scale);
        const y2 = base.offsetY + Math.floor(seg.y2 * base.scale);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // Draw objects; red if high risk, otherwise green
    if (data && Array.isArray(data.objects)) {
      for (const obj of data.objects) {
        const [x, y, w, h] = obj.bbox;
        const rx = base.offsetX + Math.floor(x * base.scale);
        const ry = base.offsetY + Math.floor(y * base.scale);
        const rw = Math.floor(w * base.scale);
        const rh = Math.floor(h * base.scale);
        ctx.strokeStyle = obj.high_risk ? 'rgba(220, 53, 69, 1)' : 'rgba(40, 167, 69, 1)';
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);
        // label + confidence
        const label = `${obj.label || 'obj'} ${((obj.confidence || 0) * 100).toFixed(1)}%`;
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        const pad = 4;
        ctx.font = '12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto';
        const textWidth = ctx.measureText(label).width;
        const textHeight = 14;
        ctx.fillRect(rx, Math.max(ry - textHeight - pad * 2, 0), textWidth + pad * 2, textHeight + pad * 2);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, rx + pad, Math.max(ry - pad, textHeight));
      }
    }
  }
})();

