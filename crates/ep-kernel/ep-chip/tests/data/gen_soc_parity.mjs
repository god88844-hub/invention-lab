// WP-4.4 — deterministic golden-parity generator for the SoC interposer
// composer. Runs `packages/chip-engine/src/accelerators.mjs :: compareArchitectures`
// on a fixed workload grid and freezes every output field (floats as exact
// IEEE bit patterns) into `soc_parity_vectors.json`, which the Rust test
// `tests/soc_oracle.rs` replays bit-exactly. A `_provenance` block pins the
// source, the engine and the JS semantics. Run: `node gen_soc_parity.mjs`.

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { compareArchitectures } from '../../../../../packages/chip-engine/src/accelerators.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const f64bits = (x) => Buffer.from(new Float64Array([x]).buffer).toString('hex');

// Fixed workload grid: the JS defaults plus declared-variant points that
// exercise every branch of the compare engine (both radices, zero-copy vs
// DMA, non-default clocks/voltages/bandwidths, odd tile edges).
const cases = [
  { id: 'js-defaults', workload: {}, opts: {} },
  { id: 'default-128', workload: { M: 128, N: 128, K: 128 }, opts: {} },
  { id: 'int8-clock-100', workload: { M: 64, N: 64, K: 64 }, opts: { clockMHz: 100 } },
  { id: 'q4-vdd-07', workload: { M: 96, N: 64, K: 96 }, opts: { vdd: 0.7 } },
  { id: 'edge-tiles', workload: { M: 70, N: 45, K: 33 }, opts: { gridSize: 8 } },
  { id: 'grid-4', workload: { M: 32, N: 32, K: 64 }, opts: { gridSize: 4 } },
  { id: 'hbm-bus', workload: { M: 64, N: 64, K: 128 }, opts: { hbmGBps: 600, pcieGBps: 32, lpddrGBps: 68 } },
  { id: 'm2-bus', workload: { M: 64, N: 64, K: 64 }, opts: { lpddrGBps: 100, pcieGBps: 64 } },
];

const vectors = cases.map(({ id, workload, opts }) => {
  const r = compareArchitectures(workload, opts);
  const arch = (a) => ({
    architecture: a.architecture,
    radix: a.radix,
    totalCycles: a.totalCycles,
    bandwidthGBpsBits: f64bits(a.bandwidthGBps),
    energyPerOpPJBits: f64bits(a.energyPerOpPJ),
    dynamicPowerWBits: f64bits(a.dynamicPowerW),
    hotspotCBits: f64bits(a.hotspotC),
    bytesOnBus: a.bytesOnBus,
  });
  return {
    id,
    workload,
    opts,
    discreteInt8: arch(r.discreteInt8),
    discreteQ4: arch(r.discreteQ4),
    unifiedInt8: arch(r.unifiedInt8),
    unifiedQ4: arch(r.unifiedQ4),
    delta: {
      cycleReductionPctBits: f64bits(r.delta.cycleReductionPct),
      energyReductionPctBits: f64bits(r.delta.energyReductionPct),
      q4BandwidthGainPctBits: f64bits(r.delta.q4BandwidthGainPct),
    },
    badge: r.badge,
  };
});

const out = {
  _provenance: {
    format: 'invention-lab/ep-chip-soc-parity/1',
    source: 'packages/chip-engine/src/accelerators.mjs compareArchitectures (JS oracle)',
    generator: 'tests/data/gen_soc_parity.mjs (deterministic, committed)',
    semantics:
      'bytesPerElement int8=1 q4=0.5; unified zeroCopy => weightsMoved=0; bytesPerCycle = BW*1e9/(clockMHz*1e6); dmaCycles = bytesOnBus/bytesPerCycle; dequantCycles(q4) = ceil(M/P)*ceil(N/P)*P; totalCycles = compute + ceil(dma) + dequant; bandwidthGBps = bytesOnBus/(totalCycles/clockMHz/1e6)/1e9; energyPerOpPJ = alpha*V^2*1e12 (alpha int8 1.2 / q4 0.9); dynamicPowerW = ops*E*1e-12/seconds; hotspotC = 45 + P*0.8*10; deltas = (discrete-unified)/discrete*100. Floats frozen as IEEE-754 bit hex for bit-exact replay.',
    badge: 'SIMULATED_ARCHITECTURAL_MODEL',
    note: 'Architectural-model parity fixture — no silicon measurements.',
  },
  cases: vectors,
};

writeFileSync(path.join(here, 'soc_parity_vectors.json'), JSON.stringify(out, null, 1) + '\n');
console.log(`soc_parity_vectors.json written: ${vectors.length} cases`);
