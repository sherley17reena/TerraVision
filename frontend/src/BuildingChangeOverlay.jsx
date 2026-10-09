
import { useEffect, useState } from "react";

export default function BuildingChangeOverlay({
  afterImage,
  base64Mask,
}) {
  const [overlayUrl, setOverlayUrl] = useState(null);

  useEffect(() => {
    if (!base64Mask) {
      setOverlayUrl(null);
      return;
    }

    let cancelled = false;
    setOverlayUrl(null);

    const mask = new Image();

    mask.onload = () => {
      const canvas = document.createElement("canvas");

      canvas.width = mask.naturalWidth;
      canvas.height = mask.naturalHeight;

      const ctx = canvas.getContext("2d", {
        willReadFrequently: true,
      });

      if (!ctx) return;

      ctx.drawImage(mask, 0, 0);

      const imageData = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
      );

      const pixels = imageData.data;

      for (let i = 0; i < pixels.length; i += 4) {
        const changed = pixels[i] > 127;

        pixels[i] = 255;
        pixels[i + 1] = 0;
        pixels[i + 2] = 0;
        pixels[i + 3] = changed ? 165 : 0;
      }

      ctx.putImageData(imageData, 0, 0);

      if (!cancelled) {
        setOverlayUrl(canvas.toDataURL("image/png"));
      }
    };

    mask.src = `data:image/png;base64,${base64Mask}`;

    return () => {
      cancelled = true;
    };
  }, [base64Mask]);

  return (
    <div className="overlay-container">
      <img
        src={afterImage}
        alt="Satellite image after changes"
        className="overlay-base"
      />

      {overlayUrl && (
        <img
          src={overlayUrl}
          alt="Predicted building changes highlighted in red"
          className="overlay-mask"
        />
      )}
    </div>
  );
}
