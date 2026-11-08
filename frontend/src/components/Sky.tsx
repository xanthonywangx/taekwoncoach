import { useState } from 'react';

/**
 * Background: a still gradient (CSS on .sky) with whole cloud images drifting down over it.
 * Each cloud is always fully on screen side-to-side, and picks a new random position every time it
 * starts a new pass, so the arrangement keeps shuffling.
 *
 * Images live in frontend/public/sky/. Edit the list to add, remove, resize or speed up clouds.
 *   width: % of screen width (100 = full width; the whole image always fits)
 *   speed: seconds to fall from above the top to below the bottom (bigger = slower)
 *   delay: negative = start partway down, so the sky isn't empty when the page opens
 */
const SCALE = 1; // shrink/grow every cloud at once, e.g. 0.8
const OVERHANG = 0.2
const CLOUDS = [
  // back (drawn first): smaller, slower, fainter
  { src: '/sky/3.png', width: 45, speed: 70, delay: -10, opacity: 0.7 },
  { src: '/sky/3.png', width: 40, speed: 70, delay: -45, opacity: 0.7 },
  { src: '/sky/2.png', width: 55, speed: 50, delay: -5, opacity: 0.85 },
  { src: '/sky/2.png', width: 50, speed: 50, delay: -30, opacity: 0.85 },
  // front (drawn last): bigger, faster
  { src: '/sky/4.png', width: 65, speed: 36, delay: -12, opacity: 1 },
  { src: '/sky/4.png', width: 60, speed: 36, delay: -30, opacity: 1 },
];


function randomLeft(width: number) {
  const min = -width * OVERHANG;
  const max = 100 - width * (1 - OVERHANG);
  return min + Math.random() * (max - min);
}

function Cloud({ src, width, speed, delay, opacity }: (typeof CLOUDS)[number]) {
  const w = Math.min(100, width * SCALE);
  const [left, setLeft] = useState(() => randomLeft(w));
  return (
    <img
      className="cloud"
      src={src}
      alt=""
      draggable={false}
      style={{
        width: `${w}vw`,
        left: `${left}vw`,
        opacity,
        animationDuration: `${speed}s`,
        animationDelay: `${delay}s`,
      }}
      onAnimationIteration={() => setLeft(randomLeft(w))}
    />
  );
}

export default function Sky() {
  return (
    <div className="sky" aria-hidden="true">
      {CLOUDS.map((c, i) => (
        <Cloud key={i} {...c} />
      ))}
    </div>
  );
}
