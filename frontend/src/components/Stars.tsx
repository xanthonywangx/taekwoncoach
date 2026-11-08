
const PIXELS = [
  '....#....',
  '....#....',
  '...###...',
  '#########',
  '.#######.',
  '..#####..',
  '..#####..',
  '.##...##.',
  '##.....##',
];

// Turn each row into horizontal runs of filled pixels (fewer SVG rects).
const RUNS = PIXELS.flatMap((row, y) => {
  const runs: { x: number; y: number; w: number }[] = [];
  let start = -1;
  for (let x = 0; x <= row.length; x++) {
    const filled = row[x] === '#';
    if (filled && start < 0) start = x;
    if (!filled && start >= 0) {
      runs.push({ x: start, y, w: x - start });
      start = -1;
    }
  }
  return runs;
});

function PixelStar({ on }: { on: boolean }) {
  return (
    <svg className={on ? 'star on' : 'star off'} viewBox="0 0 9 9" shapeRendering="crispEdges" aria-hidden="true">
      {RUNS.map((r, i) => (
        <rect key={i} x={r.x} y={r.y} width={r.w} height={1} fill="currentColor" />
      ))}
    </svg>
  );
}

export default function Stars({ passed, size = 'md' }: { passed: number; size?: 'sm' | 'md' | 'lg' }) {
  return (
    <span className={`stars stars-${size}`} role="img" aria-label={`${passed} of 4 checks passed`} title={`${passed}/4 checks`}>
      {[0, 1, 2, 3].map((i) => (
        <PixelStar key={i} on={i < passed} />
      ))}
    </span>
  );
}
