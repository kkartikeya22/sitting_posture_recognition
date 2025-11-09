
const $ = (q)=>document.querySelector(q);

// --------------------------
// Single Image Upload
// --------------------------
const form = $("#uploadForm");
const fileInput = $("#fileInput");
const inputPreview = $("#inputPreview");
const outputPreview = $("#outputPreview");
const imageResult = $("#imageResult");
const postureBadge = $("#postureBadge");
const notesBadge = $("#notesBadge");

form?.addEventListener("submit", async (e)=>{
  e.preventDefault();
  if(!fileInput.files?.length){ alert("Choose an image first."); return; }
  const file = fileInput.files[0];
  inputPreview.src = URL.createObjectURL(file);

  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/analyze-image", { method:"POST", body: fd });
  const data = await res.json();
  if (data.error) {
    alert("Error: " + data.error);
    return;
  }
  outputPreview.src = "data:image/png;base64," + data.image_b64;
  postureBadge.textContent = "Posture: " + (data.posture || "—");
  notesBadge.textContent = "Notes: " + ((data.notes && data.notes.join(" · ")) || "—");
  imageResult.classList.remove("hidden");
});

// --------------------------
/* Live Webcam Side-by-Side */
// --------------------------
const video = $("#video");
const canvasOut = $("#canvasOut");
const startBtn = $("#startBtn");
const stopBtn = $("#stopBtn");
const livePosture = $("#livePosture");
const liveNotes = $("#liveNotes");

let stream = null;
let running = false;
let sendTimer = null;

async function startWebcam(){
  if (running) return;
  try{
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  }catch(err){
    alert("Webcam permission denied or not available.");
    return;
  }
  video.srcObject = stream;
  running = true;
  pumpFrames();
}

function stopWebcam(){
  running = false;
  if (sendTimer) clearTimeout(sendTimer);
  if (stream){
    stream.getTracks().forEach(t=>t.stop());
    stream = null;
  }
}

async function pumpFrames(){
  if (!running) return;
  // Draw current video frame to a temp canvas and send to server
  const tcanvas = document.createElement("canvas");
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  tcanvas.width = w; tcanvas.height = h;
  const tctx = tcanvas.getContext("2d");
  tctx.drawImage(video, 0, 0, w, h);
  const dataURL = tcanvas.toDataURL("image/jpeg", 0.7);

  try{
    const res = await fetch("/api/analyze-frame", {
      method:"POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ frame_b64: dataURL })
    });
    const data = await res.json();
    if (!data.error){
      const img = new Image();
      img.onload = ()=>{
        canvasOut.width = img.width;
        canvasOut.height = img.height;
        const ctx = canvasOut.getContext("2d");
        ctx.drawImage(img, 0, 0);
      };
      img.src = "data:image/png;base64," + data.image_b64;
      livePosture.textContent = "Posture: " + (data.posture || "—");
      liveNotes.textContent = "Notes: " + ((data.notes && data.notes.join(" · ")) || "—");
    }
  }catch(e){
    console.error(e);
  }

  // Throttle a bit (adjust as needed)
  sendTimer = setTimeout(pumpFrames, 600);
}

startBtn?.addEventListener("click", startWebcam);
stopBtn?.addEventListener("click", stopWebcam);
