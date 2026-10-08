
function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadReport(results, mode) {
  const report = {
    project: "TerraVision",
    analysis_type: mode,
    model: mode === "buildings" ? "Temporal U-Net" : "U-Net",
    generated_at: new Date().toISOString(),
    changed_pixels: results.changed_pixels,
    total_pixels: results.total_pixels,
    predicted_change_percentage: results.change_percentage,
    ...(mode === "land-cover"
      ? {
          land_cover: results.land_cover,
          transitions: results.transitions,
        }
      : {
          threshold: 0.5,
          test_f1: 0.8316,
          test_iou: 0.7118,
        }),
    note:
      "Results represent model predictions, not verified geographical changes.",
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], {
    type: "application/json",
  });

  triggerDownload(blob, "terravision_analysis_report.json");
}

export function downloadMask(base64Mask) {
  const binary = atob(base64Mask);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));

  triggerDownload(
    new Blob([bytes], { type: "image/png" }),
    "terravision_building_change_mask.png"
  );
}

export async function downloadOverlay(afterImageUrl, base64Mask) {
  const loadImage = (src) =>
    new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Could not load image"));
      image.src = src;
    });

  const [afterImage, maskImage] = await Promise.all([
    loadImage(afterImageUrl),
    loadImage(`data:image/png;base64,${base64Mask}`),
  ]);

  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;

  const ctx = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  if (!ctx) {
    throw new Error("Canvas is unavailable.");
  }

  ctx.drawImage(afterImage, 0, 0, 256, 256);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = 256;
  maskCanvas.height = 256;

  const maskCtx = maskCanvas.getContext("2d", {
    willReadFrequently: true,
  });

  maskCtx.drawImage(maskImage, 0, 0, 256, 256);

  const pixels = maskCtx.getImageData(0, 0, 256, 256).data;
  const overlay = ctx.createImageData(256, 256);

  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] > 127) {
      overlay.data[i] = 255;
      overlay.data[i + 1] = 0;
      overlay.data[i + 2] = 0;
      overlay.data[i + 3] = 165;
    }
  }

  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = 256;
  overlayCanvas.height = 256;

  overlayCanvas.getContext("2d").putImageData(overlay, 0, 0);

  ctx.drawImage(overlayCanvas, 0, 0);

  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/png")
  );

  if (!blob) {
    throw new Error("Could not generate overlay PNG.");
  }

  triggerDownload(blob, "terravision_building_change_overlay.png");
}
