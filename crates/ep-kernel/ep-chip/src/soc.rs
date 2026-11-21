//! WP-4.4 — Apple-M-class unified-memory SoC interposer composer.
//!
//! Every number this module emits is an architectural model derived from the
//! SHA-256-verified offline vault
//! (`data/reference-library/07_soc_interposer_memory_baselines.json`, parsed
//! verbatim via [`SocBaselines::from_json`], and
//! `data/cad-specs/02_pcb_bga_soc_packages.json` `soc_floorplan`, bound by
//! [`SocFloorplan::from_cad_specs_json`]) — badged
//! `SIMULATED_ARCHITECTURAL_MODEL`. It is NOT a cycle-accurate RTL NoC, a
//! GDS floorplan, or a silicon measurement (Builder Rule 5).
//!
//! Surfaces (PHASE_04 WP-4.4 checklist + `03_PHYSICS_MATH_FUNDAMENTALS.md` §3):
//! - [`NoCFabric`]: interposer routes modelled as **1 cycle per hop** (07
//!   `interposer_reference.rc_delay_note`), graph built from the 5 declared
//!   tile→block routes of the sealed `02` floorplan; shortest-path latency
//!   over the route graph, fail-closed on unknown endpoints.
//! - [`SocBaselines`] / [`MemoryStack`]: LPDDR5 unified (68 GB/s Intel 155U
//!   class / 68.25 M1 / 100 M2; 90 ns random access), HBM (600 GB/s TPU v1),
//!   discrete PCIe 4.0/5.0 x16 (32 / 64 GB/s) — all JEDEC/vendor-published
//!   interface maximums, `bytes_per_cycle` and `copy_time_cycles` exactly the
//!   07-JSON equations.
//! - [`compare_architectures`]: bit-exact Rust port of
//!   `packages/chip-engine/src/accelerators.mjs :: compareArchitectures`
//!   (§5 oracle discipline: golden fixture `tests/data/soc_parity_vectors.json`
//!   captured by the committed generator + live Node differential with loud
//!   SKIP) — unified zero-copy vs discrete-PCIe DMA cycle/bandwidth/energy
//!   deltas for INT8 and Q4 radices.
//! - [`roofline`]: `achievable_FLOPs = min(PeakFLOPs, AI·PeakBW)` (§3) with
//!   the declared bottleneck.
//! - [`littles_law`]: `L = λ·W` — in-flight memory = bandwidth × latency
//!   over the 07 cache hierarchy (L1 3 / L2 12 / L3 26 / DRAM 200 cycles).
//! - [`DvfsController`]: closed-loop DVFS thermal throttling —
//!   `T_junction > 125 °C` (05 envelope `riscv_rv32im.temp_range_c` upper
//!   bound) reduces the frequency stepwise with a ±5 °C hysteresis band
//!   (PHASE_04 failure modes); `T ≥ 200 °C` latches a `si_thermal_runaway`
//!   event (06 JSON `si_thermal_runaway.threshold_c = 200`) and refuses
//!   further operation until explicitly reset. The thermal feed is a
//!   closure, so the loop couples to **real `ep-thermal` tile temps**
//!   (demonstrated in `tests/soc_oracle.rs` with a live `ThermalGrid2D`).

use ep_core::ProvenanceBadge;
use serde::{Deserialize, Serialize};

/// Provenance badge carried by every output surface of this module.
pub const SOC_BADGE: ProvenanceBadge = ProvenanceBadge::SimulatedArchitecturalModel;

/// DVFS throttle threshold — `05 → valid_operating_envelope.riscv_rv32im
/// .temp_range_c` upper bound (§3: "if T_junction > 125 °C (05 envelope)").
pub const THROTTLE_THRESHOLD_C: f64 = 125.0;
/// Thermal-runaway threshold — `06 → parameters.si_thermal_runaway
/// .threshold_c` (leakage doubles every ~10 °C above ~200 °C).
pub const RUNAWAY_THRESHOLD_C: f64 = 200.0;
/// DVFS hysteresis band (±5 °C) — PHASE_04 failure modes: "DVFS oscillation
/// → hysteresis band (±5 °C) on the throttle loop".
pub const DVFS_HYSTERESIS_C: f64 = 5.0;
/// Declared stepwise frequency change per control cycle (% of the current
/// frequency). A declared model parameter — not a measured DVFS table.
pub const DVFS_STEP_PCT: f64 = 10.0;

// ---------------------------------------------------------------------------
// Errors — fail-closed, never silent.
// ---------------------------------------------------------------------------

/// Fail-closed SoC composer errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocError {
    /// A sealed vault payload is malformed or tampered.
    Vault(String),
    /// Unknown block / endpoint / route.
    Unknown(String),
    /// Non-physical input (negative bandwidth, non-finite temps, …).
    NonPhysicalInput(String),
    /// The DVFS loop latched a `si_thermal_runaway` and refuses to operate.
    RunawayLatched,
}

impl std::fmt::Display for SocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vault(what) => write!(f, "SoC vault payload invalid: {what}"),
            Self::Unknown(what) => write!(f, "unknown SoC element: {what}"),
            Self::NonPhysicalInput(what) => write!(f, "non-physical SoC input: {what}"),
            Self::RunawayLatched => write!(
                f,
                "si_thermal_runaway latched: junction ≥ 200 °C — the model refuses further operation until reset"
            ),
        }
    }
}

impl std::error::Error for SocError {}

// ---------------------------------------------------------------------------
// 07 JSON baselines — parsed verbatim from the sealed vault payload.
// ---------------------------------------------------------------------------

/// Parsed `07_soc_interposer_memory_baselines.json` (all figures verbatim —
/// JEDEC/vendor-published interface maximums; no invented constants).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SocBaselines {
    /// LPDDR5/LPDDR5x unified-memory reference (Apple-M / Intel 155U class).
    pub lpddr5: Lpddr5,
    /// HBM2/HBM2E interposer reference (TPU v1 topology).
    pub hbm: Hbm,
    /// Discrete accelerator over PCIe (copy-penalty baseline).
    pub pcie: Pcie,
    /// Cache hierarchy latencies (cycles) + capacities.
    pub caches: CacheHierarchy,
    /// Interposer dimensions (mm) — TPU v1 + the v3 target scene.
    pub interposer: InterposerReference,
}

/// LPDDR5 unified-memory block of the 07 payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lpddr5 {
    pub per_channel_16bit_4800mts: f64,
    pub per_channel_16bit_7467mts: f64,
    pub apple_m1_total: f64,
    pub apple_m2_total: f64,
    pub intel_iris_xe_155u_total: f64,
    /// End-to-end SoC load-to-use (published analyses range 80–120 ns).
    pub typical_random_access_ns: f64,
    pub apple_m1_base_gb: f64,
    pub apple_m2_base_gb: f64,
    pub target_platform_gb: f64,
    /// The 07 zero-copy contract, verbatim.
    pub zero_copy_note: String,
}

/// HBM block of the 07 payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hbm {
    pub tpu_v1_total: f64,
    pub hbm2e_per_stack: f64,
}

/// Discrete PCIe block of the 07 payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pcie {
    pub pcie4_x16_directional: f64,
    pub pcie5_x16_directional: f64,
}

/// Cache hierarchy of the 07 payload (latencies in cycles).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheHierarchy {
    pub l1_latency_cycles: f64,
    pub l2_latency_cycles: f64,
    pub l3_latency_cycles: f64,
    pub dram_latency_cycles: f64,
}

/// Interposer reference dimensions (mm strings in the payload, kept verbatim).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterposerReference {
    pub tpu_v1_interposer: String,
    pub target_soc_interposer: String,
}

impl SocBaselines {
    /// Parse the sealed 07 payload verbatim (fail-closed on tamper/shape).
    ///
    /// # Errors
    /// [`SocError::Vault`] when any required figure is missing or not the
    /// sealed value type.
    pub fn from_json(json: &str) -> Result<Self, SocError> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| SocError::Vault(format!("JSON parse: {e}")))?;
        let p = |path: &[&str]| -> Result<f64, SocError> {
            let mut node = &v;
            for key in path {
                node = node.get(key).ok_or_else(|| {
                    SocError::Vault(format!("missing 07 field '{}'", path.join(".")))
                })?;
            }
            node.as_f64().ok_or_else(|| {
                SocError::Vault(format!("non-numeric 07 field '{}'", path.join(".")))
            })
        };
        let s = |path: &[&str]| -> Result<String, SocError> {
            let mut node = &v;
            for key in path {
                node = node.get(key).ok_or_else(|| {
                    SocError::Vault(format!("missing 07 field '{}'", path.join(".")))
                })?;
            }
            node.as_str()
                .map(str::to_string)
                .ok_or_else(|| SocError::Vault(format!("non-string 07 field '{}'", path.join("."))))
        };
        let lp = &v["parameters"]["lpddr5_unified_memory"];
        if !lp.is_object() {
            return Err(SocError::Vault("missing lpddr5_unified_memory".into()));
        }
        Ok(Self {
            lpddr5: Lpddr5 {
                per_channel_16bit_4800mts: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "bandwidth_gbps",
                    "per_channel_16bit_4800mts",
                ])?,
                per_channel_16bit_7467mts: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "bandwidth_gbps",
                    "per_channel_16bit_7467mts",
                ])?,
                apple_m1_total: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "bandwidth_gbps",
                    "apple_m1_total",
                ])?,
                apple_m2_total: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "bandwidth_gbps",
                    "apple_m2_total",
                ])?,
                intel_iris_xe_155u_total: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "bandwidth_gbps",
                    "intel_iris_xe_155u_total",
                ])?,
                typical_random_access_ns: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "latency_ns",
                    "typical_random_access_ns",
                ])?,
                apple_m1_base_gb: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "capacity_gb",
                    "apple_m1_base",
                ])?,
                apple_m2_base_gb: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "capacity_gb",
                    "apple_m2_base",
                ])?,
                target_platform_gb: p(&[
                    "parameters",
                    "lpddr5_unified_memory",
                    "capacity_gb",
                    "target_platform_gb",
                ])?,
                zero_copy_note: s(&["parameters", "lpddr5_unified_memory", "zero_copy_note"])?,
            },
            hbm: Hbm {
                tpu_v1_total: p(&[
                    "parameters",
                    "hbm_reference",
                    "bandwidth_gbps",
                    "tpu_v1_total",
                ])?,
                hbm2e_per_stack: p(&[
                    "parameters",
                    "hbm_reference",
                    "bandwidth_gbps",
                    "hbm2e_per_stack",
                ])?,
            },
            pcie: Pcie {
                pcie4_x16_directional: p(&[
                    "parameters",
                    "discrete_pcie_reference",
                    "bandwidth_gbps",
                    "pcie4_x16_directional",
                ])?,
                pcie5_x16_directional: p(&[
                    "parameters",
                    "discrete_pcie_reference",
                    "bandwidth_gbps",
                    "pcie5_x16_directional",
                ])?,
            },
            caches: CacheHierarchy {
                l1_latency_cycles: p(&[
                    "parameters",
                    "cache_hierarchy_reference",
                    "levels",
                    "l1_latency_cycles",
                ])?,
                l2_latency_cycles: p(&[
                    "parameters",
                    "cache_hierarchy_reference",
                    "levels",
                    "l2_latency_cycles",
                ])?,
                l3_latency_cycles: p(&[
                    "parameters",
                    "cache_hierarchy_reference",
                    "levels",
                    "l3_latency_cycles",
                ])?,
                dram_latency_cycles: p(&[
                    "parameters",
                    "cache_hierarchy_reference",
                    "levels",
                    "dram_latency_cycles",
                ])?,
            },
            interposer: InterposerReference {
                tpu_v1_interposer: s(&[
                    "parameters",
                    "interposer_reference",
                    "dimensions_mm",
                    "tpu_v1_interposer",
                ])?,
                target_soc_interposer: s(&[
                    "parameters",
                    "interposer_reference",
                    "dimensions_mm",
                    "target_soc_interposer",
                ])?,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Memory stacks + the 07 bus equations.
// ---------------------------------------------------------------------------

/// One memory/bus interface of the composed SoC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Apple-M-class unified LPDDR5 (zero-copy shared address space).
    UnifiedLpddr5,
    /// HBM stacks on a silicon interposer (TPU v1 topology).
    Hbm,
    /// Discrete accelerator behind PCIe (full DMA copy per move).
    DiscretePcie4,
    /// Discrete accelerator behind PCIe 5.0.
    DiscretePcie5,
}

impl MemoryKind {
    /// Interface bandwidth (GB/s) from the 07 payload figures.
    #[must_use]
    pub fn bandwidth_gbps(self, b: &SocBaselines) -> f64 {
        match self {
            Self::UnifiedLpddr5 => b.lpddr5.intel_iris_xe_155u_total,
            Self::Hbm => b.hbm.tpu_v1_total,
            Self::DiscretePcie4 => b.pcie.pcie4_x16_directional,
            Self::DiscretePcie5 => b.pcie.pcie5_x16_directional,
        }
    }

    /// Whether this interface pays a DMA copy (discrete) or touches memory
    /// once (Apple-M-class unified zero-copy, 07 `zero_copy_note`).
    #[must_use]
    pub const fn is_zero_copy(self) -> bool {
        matches!(self, Self::UnifiedLpddr5 | Self::Hbm)
    }
}

/// `bytes_per_cycle = BW_GBps × 1e9 / clock_Hz` (07 `equations`, verbatim).
///
/// # Errors
/// [`SocError::NonPhysicalInput`] for non-physical bandwidth/clock.
pub fn bytes_per_cycle(bw_gbps: f64, clock_mhz: f64) -> Result<f64, SocError> {
    if !(bw_gbps.is_finite() && bw_gbps >= 0.0) || !(clock_mhz.is_finite() && clock_mhz > 0.0) {
        return Err(SocError::NonPhysicalInput(format!(
            "bandwidth {bw_gbps} GB/s at {clock_mhz} MHz"
        )));
    }
    Ok(bw_gbps * 1e9 / (clock_mhz * 1e6))
}

/// `copy_time_cycles = bytes / bytes_per_cycle` (07 `equations`, verbatim).
///
/// # Errors
/// [`SocError::NonPhysicalInput`] for non-physical inputs.
pub fn copy_time_cycles(bytes: f64, bw_gbps: f64, clock_mhz: f64) -> Result<f64, SocError> {
    if !(bytes.is_finite() && bytes >= 0.0) {
        return Err(SocError::NonPhysicalInput(format!(
            "copy size {bytes} bytes"
        )));
    }
    Ok(bytes / bytes_per_cycle(bw_gbps, clock_mhz)?)
}

/// Explicit driver submission round-trip in cycles (discrete PCIe only).
///
/// The 07 payload cites a published driver round-trip of ~5–10 µs per
/// submission; the caller supplies the µs figure and clock — nothing is
/// assumed inside the bit-exact compare path.
///
/// # Errors
/// [`SocError::NonPhysicalInput`] for non-physical inputs.
pub fn submission_latency_cycles(driver_latency_us: f64, clock_mhz: f64) -> Result<f64, SocError> {
    if !(driver_latency_us.is_finite() && driver_latency_us >= 0.0) {
        return Err(SocError::NonPhysicalInput(format!(
            "driver latency {driver_latency_us} µs"
        )));
    }
    Ok(driver_latency_us * 1e-6 * clock_mhz * 1e6)
}

// ---------------------------------------------------------------------------
// Tile→block floorplan (cad-specs 02 soc_floorplan) + NoC fabric.
// ---------------------------------------------------------------------------

/// A floorplan block parsed verbatim from `02 → soc_floorplan.blocks`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorplanBlock {
    pub id: String,
    pub name: String,
    /// Tile count (cpu 8, tpu 64) or compute units (gpu 96); 0 for
    /// non-compute blocks (memory stacks, NoC fabric).
    pub tiles: u32,
    /// Block position on the die (mm, corner).
    pub position_mm: (f64, f64),
    /// Block size (mm).
    pub size_mm: (f64, f64),
}

/// The die/interposer floorplan bound to the sealed `02` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SocFloorplan {
    /// Die size (mm) — `soc_floorplan.die_size_mm`.
    pub die_mm: (f64, f64),
    /// Interposer size (mm) — `interposer.dimensions_mm` (10×7 target scene).
    pub interposer_mm: (f64, f64),
    /// Blocks in sealed order (cpu, tpu, gpu, mem0, mem1, noc).
    pub blocks: Vec<FloorplanBlock>,
    /// The 5 declared NoC route legs (undirected pairs).
    pub routes: Vec<(String, String)>,
}

impl SocFloorplan {
    /// Parse `data/cad-specs/02_pcb_bga_soc_packages.json` `soc_floorplan`
    /// (+ `interposer.dimensions_mm`) verbatim, fail-closed.
    ///
    /// # Errors
    /// [`SocError::Vault`] when the sealed payload is malformed or the five
    /// required blocks / five declared routes are missing.
    pub fn from_cad_specs_json(json: &str) -> Result<Self, SocError> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| SocError::Vault(format!("JSON parse: {e}")))?;
        let floor = &v["soc_floorplan"];
        if !floor.is_object() {
            return Err(SocError::Vault("missing soc_floorplan".into()));
        }
        let dims = |obj: &serde_json::Value, key: &str| -> Result<(f64, f64), SocError> {
            let d = &obj[key];
            let w = d["w"]
                .as_f64()
                .ok_or_else(|| SocError::Vault(format!("{key}.w")))?;
            let h = d["h"]
                .as_f64()
                .ok_or_else(|| SocError::Vault(format!("{key}.h")))?;
            Ok((w, h))
        };
        let die_mm = dims(floor, "die_size_mm")?;
        let interposer_mm = dims(&v["interposer"], "dimensions_mm")?;
        let mut blocks = Vec::new();
        for b in floor["blocks"]
            .as_array()
            .ok_or_else(|| SocError::Vault("soc_floorplan.blocks missing".into()))?
        {
            let id = b["id"]
                .as_str()
                .ok_or_else(|| SocError::Vault("block id missing".into()))?
                .to_string();
            let name = b["name"]
                .as_str()
                .ok_or_else(|| SocError::Vault("block name missing".into()))?
                .to_string();
            // The gpu block declares `units` (96 EU), cpu/tpu declare `tiles`.
            let tiles = b["tiles"]
                .as_u64()
                .or_else(|| b["units"].as_u64())
                .unwrap_or(0) as u32;
            let pos = (
                b["position_mm"]["x"].as_f64().unwrap_or(0.0),
                b["position_mm"]["y"].as_f64().unwrap_or(0.0),
            );
            let size = dims(b, "size_mm").unwrap_or((0.0, 0.0));
            blocks.push(FloorplanBlock {
                id,
                name,
                tiles,
                position_mm: pos,
                size_mm: size,
            });
        }
        for required in ["cpu", "tpu", "gpu", "mem0", "mem1"] {
            if !blocks.iter().any(|b| b.id == required) {
                return Err(SocError::Vault(format!(
                    "required block '{required}' missing"
                )));
            }
        }
        // The NoC fabric declares its route legs in the sealed note
        // ("mem↔cpu, mem↔tpu, mem↔gpu, cpu↔tpu, tpu↔gpu") — parsed from the
        // payload text, never hardcoded twice.
        if !blocks.iter().any(|b| b.id == "noc") {
            return Err(SocError::Vault("noc block missing".into()));
        }
        let noc_json = floor["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["id"] == "noc")
            .ok_or_else(|| SocError::Vault("noc block missing".into()))?;
        let note = noc_json["note"]
            .as_str()
            .ok_or_else(|| SocError::Vault("noc note missing".into()))?;
        let mut routes = Vec::new();
        for leg in note.split(',') {
            let leg = leg.trim();
            let (a, b) = leg
                .split_once("↔")
                .ok_or_else(|| SocError::Vault(format!("unparsable NoC leg '{leg}'")))?;
            routes.push((a.trim().to_string(), b.trim().to_string()));
        }
        let declared_count = noc_json["routes"].as_u64().unwrap_or(0) as usize;
        if routes.len() != declared_count {
            return Err(SocError::Vault(format!(
                "noc declares {} routes but the note lists {}",
                declared_count,
                routes.len()
            )));
        }
        // The sealed note names the logical memory hub ("mem") — the
        // floorplan instantiates it as the concrete unified stacks mem0/mem1:
        // a "mem↔X" leg is a 1-hop route from EVERY stack to X.
        let stack_ids: Vec<String> = blocks
            .iter()
            .filter(|b| b.id.starts_with("mem") && b.id != "noc")
            .map(|b| b.id.clone())
            .collect();
        let mut expanded = Vec::new();
        for (a, b) in &routes {
            if a == "mem" {
                for stack in &stack_ids {
                    expanded.push((stack.clone(), b.clone()));
                }
            } else if b == "mem" {
                for stack in &stack_ids {
                    expanded.push((a.clone(), stack.clone()));
                }
            } else {
                expanded.push((a.clone(), b.clone()));
            }
        }
        routes = expanded;
        Ok(Self {
            die_mm,
            interposer_mm,
            blocks,
            routes,
        })
    }

    /// Block lookup by id.
    #[must_use]
    pub fn block(&self, id: &str) -> Option<&FloorplanBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }
}

/// The interposer NoC fabric: **1 cycle per hop** (07
/// `interposer_reference`), routed over the sealed floorplan's route graph.
#[derive(Debug, Clone)]
pub struct NoCFabric<'a> {
    floorplan: &'a SocFloorplan,
}

impl<'a> NoCFabric<'a> {
    /// Build the fabric over a parsed floorplan.
    #[must_use]
    pub const fn new(floorplan: &'a SocFloorplan) -> Self {
        Self { floorplan }
    }

    /// Cycles per hop — the 07 contract, as a named constant.
    pub const CYCLES_PER_HOP: u64 = 1;

    fn adjacency(&self) -> std::collections::HashMap<&str, Vec<&str>> {
        let mut adj: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
        for (a, b) in &self.floorplan.routes {
            adj.entry(a.as_str()).or_default().push(b.as_str());
            adj.entry(b.as_str()).or_default().push(a.as_str());
        }
        adj
    }

    /// Shortest route path between two blocks (BFS over the declared legs).
    ///
    /// # Errors
    /// [`SocError::Unknown`] when either endpoint (or the path itself) is not
    /// part of the declared fabric.
    pub fn route_path<'b>(&'b self, from: &'b str, to: &'b str) -> Result<Vec<&'b str>, SocError> {
        if from == to {
            if self.floorplan.block(from).is_some() {
                return Ok(vec![from]);
            }
            return Err(SocError::Unknown(format!("block '{from}'")));
        }
        let adj = self.adjacency();
        if !adj.contains_key(from) || !adj.contains_key(to) {
            return Err(SocError::Unknown(format!(
                "'{from}' → '{to}' is not routed by the declared NoC fabric"
            )));
        }
        let mut prev: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from);
        let mut visited: std::collections::HashMap<&str, bool> = std::collections::HashMap::new();
        visited.insert(from, true);
        while let Some(node) = queue.pop_front() {
            if node == to {
                break;
            }
            if let Some(neighbors) = adj.get(node) {
                for n in neighbors.clone() {
                    if !visited.contains_key(n) {
                        visited.insert(n, true);
                        prev.insert(n, node);
                        queue.push_back(n);
                    }
                }
            }
        }
        if !visited.contains_key(to) {
            return Err(SocError::Unknown(format!(
                "no route from '{from}' to '{to}'"
            )));
        }
        let mut path = vec![to];
        let mut cur = to;
        while cur != from {
            cur = prev[cur];
            path.push(cur);
        }
        path.reverse();
        Ok(path)
    }

    /// Route latency in cycles: **1 cycle per hop** over the shortest path.
    ///
    /// # Errors
    /// [`SocError::Unknown`] for unrouted endpoints.
    pub fn route_latency_cycles(&self, from: &str, to: &str) -> Result<u64, SocError> {
        let path = self.route_path(from, to)?;
        Ok((path.len() as u64 - 1).saturating_mul(Self::CYCLES_PER_HOP))
    }
}

// ---------------------------------------------------------------------------
// Roofline + Little's Law (§3).
// ---------------------------------------------------------------------------

/// The declared bottleneck of a Roofline evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RooflineBound {
    /// `AI·PeakBW ≥ PeakFLOPs` — the compute roof binds.
    Compute,
    /// `AI·PeakBW < PeakFLOPs` — the memory roof binds.
    Memory,
}

/// Roofline result: `achievable_FLOPs = min(PeakFLOPs, AI·PeakBW)` (§3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RooflineResult {
    pub achievable_flops: f64,
    pub bound: RooflineBound,
    /// The arithmetic intensity the evaluation used (ops/byte).
    pub arithmetic_intensity: f64,
    pub peak_flops: f64,
    pub peak_bandwidth_bytes_s: f64,
    /// Declared provenance of the evaluation.
    pub badge: ProvenanceBadge,
}

/// `achievable_FLOPs = min(PeakFLOPs, AI·PeakBW)` (§3, first Roofline bullet).
///
/// # Errors
/// [`SocError::NonPhysicalInput`] for non-physical inputs.
pub fn roofline(
    peak_flops: f64,
    peak_bandwidth_bytes_s: f64,
    arithmetic_intensity: f64,
) -> Result<RooflineResult, SocError> {
    for (name, v) in [
        ("peak FLOPs", peak_flops),
        ("peak bandwidth", peak_bandwidth_bytes_s),
        ("arithmetic intensity", arithmetic_intensity),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(SocError::NonPhysicalInput(format!("{name} = {v}")));
        }
    }
    let memory_roof = arithmetic_intensity * peak_bandwidth_bytes_s;
    let (achievable, bound) = if memory_roof < peak_flops {
        (memory_roof, RooflineBound::Memory)
    } else {
        (peak_flops, RooflineBound::Compute)
    };
    Ok(RooflineResult {
        achievable_flops: achievable,
        bound,
        arithmetic_intensity,
        peak_flops,
        peak_bandwidth_bytes_s,
        badge: SOC_BADGE,
    })
}

/// Little's Law: `L = λ·W` — the concurrency bound (§3): the number of
/// memory accesses in flight equals the bandwidth (λ, accesses/s) times the
/// latency (W, s). With the 07 cache hierarchy this yields the classic
/// per-level in-flight bounds (L1 3 / L2 12 / L3 26 / DRAM 200 cycles).
///
/// # Errors
/// [`SocError::NonPhysicalInput`] for non-physical inputs.
pub fn littles_law(accesses_per_cycle: f64, latency_cycles: f64) -> Result<f64, SocError> {
    for (name, v) in [
        ("access rate", accesses_per_cycle),
        ("latency", latency_cycles),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(SocError::NonPhysicalInput(format!("{name} = {v}")));
        }
    }
    Ok(accesses_per_cycle * latency_cycles)
}

// ---------------------------------------------------------------------------
// Unified vs discrete comparison — bit-exact port of compareArchitectures.
// ---------------------------------------------------------------------------

/// Compare-engine options; the defaults are the JS oracle's defaults
/// (`compareArchitectures(workload, opts = {})`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareOptions {
    pub grid_size: usize,
    pub clock_mhz: f64,
    pub vdd: f64,
    pub pcie_gbps: f64,
    pub lpddr_gbps: f64,
    pub hbm_gbps: f64,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            grid_size: 8,
            clock_mhz: 700.0,
            vdd: 0.8,
            pcie_gbps: 32.0,  // PCIe 4.0 x16 ≈ 32 GB/s
            lpddr_gbps: 68.0, // 05/07 JSON: Intel Iris Xe LPDDR5
            hbm_gbps: 600.0,  // 05/07 JSON: TPU v1 HBM
        }
    }
}

/// One evaluated architecture arm of the compare engine (fields mirror the
/// JS object exactly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchResult {
    pub architecture: String,
    pub radix: String,
    pub total_cycles: u64,
    pub bandwidth_gbps: f64,
    pub energy_per_op_pj: f64,
    pub dynamic_power_w: f64,
    pub hotspot_c: f64,
    pub bytes_on_bus: f64,
}

/// Unified-vs-discrete delta block (percent reductions, JS formulas).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompareDelta {
    pub cycle_reduction_pct: f64,
    pub energy_reduction_pct: f64,
    pub q4_bandwidth_gain_pct: f64,
}

/// Full compare-engine output (mirrors the JS return object).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareResult {
    pub discrete_int8: ArchResult,
    pub discrete_q4: ArchResult,
    pub unified_int8: ArchResult,
    pub unified_q4: ArchResult,
    pub delta: CompareDelta,
    pub badge: ProvenanceBadge,
}

/// Bit-exact Rust port of
/// `packages/chip-engine/src/accelerators.mjs :: compareArchitectures`.
///
/// Every float expression follows the JS statement order exactly (IEEE-754
/// determinism → bit-identical results; asserted to `f64::to_bits()` in
/// `tests/soc_oracle.rs`).
///
/// # Errors
/// [`SocError::NonPhysicalInput`] for non-physical options.
pub fn compare_architectures(
    m: u64,
    n: u64,
    k: u64,
    opts: &CompareOptions,
) -> Result<CompareResult, SocError> {
    if !(opts.clock_mhz.is_finite() && opts.clock_mhz > 0.0) {
        return Err(SocError::NonPhysicalInput(format!(
            "clock {} MHz",
            opts.clock_mhz
        )));
    }
    if !(opts.vdd.is_finite() && opts.vdd >= 0.0) {
        return Err(SocError::NonPhysicalInput(format!("vdd {} V", opts.vdd)));
    }
    for (name, v) in [
        ("pcie", opts.pcie_gbps),
        ("lpddr", opts.lpddr_gbps),
        ("hbm", opts.hbm_gbps),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(SocError::NonPhysicalInput(format!(
                "{name} bandwidth {v} GB/s"
            )));
        }
    }
    if opts.grid_size == 0 {
        return Err(SocError::NonPhysicalInput("grid size 0".into()));
    }
    let (mf, nf, kf) = (m as f64, n as f64, k as f64);
    let p = opts.grid_size as f64;

    // Bytes moved per inference: weights K×N INT8 + activations M×K + outputs M×N.
    let weight_bytes = kf * nf;
    let act_bytes = mf * kf;
    let out_bytes = mf * nf;

    // Systolic compute cycles (weight-stationary: weights loaded once per tile).
    let tiles = (mf / p).ceil() * (nf / p).ceil();
    let compute_cycles = tiles * (kf + 2.0 * p - 2.0);

    let evaluate =
        |memory_gbps: f64, zero_copy: bool, radix: &'static str| -> Result<ArchResult, SocError> {
            let bytes_per_element = if radix == "q4" { 0.5 } else { 1.0 }; // Q4 packs 2 digits/byte
            let weights_moved = if zero_copy {
                0.0
            } else {
                weight_bytes * bytes_per_element
            }; // unified: no host copy
            let acts_moved = act_bytes * bytes_per_element;
            let out_moved = out_bytes * bytes_per_element;
            let bytes_on_bus = weights_moved + acts_moved + out_moved;
            let bytes_per_cycle = memory_gbps * 1e9 / (opts.clock_mhz * 1e6); // bus width × clock
            let dma_cycles = bytes_on_bus / bytes_per_cycle;
            // Q4 dequant overhead: one extra cycle per output tile row pass.
            let dequant_cycles = if radix == "q4" {
                (mf / p).ceil() * (nf / p).ceil() * p
            } else {
                0.0
            };
            let total_cycles = compute_cycles + dma_cycles.ceil() + dequant_cycles;
            let bandwidth_gbps = bytes_on_bus / (total_cycles / opts.clock_mhz / 1e6) / 1e9;
            let energy_per_op_pj = alpha_energy(radix) * opts.vdd * opts.vdd * 1e12; // pJ per MAC-ish op
            let ops = mf * nf * kf;
            let dynamic_power_w =
                (ops * energy_per_op_pj * 1e-12) / (total_cycles / opts.clock_mhz / 1e6);
            // Hotspot: share of compute power into one tile thermal resistance (0.8 K/W).
            let hotspot_c = 45.0 + dynamic_power_w * 0.8 * 10.0;
            Ok(ArchResult {
                architecture: if zero_copy {
                    "unified-memory".to_string()
                } else {
                    "discrete-pcie".to_string()
                },
                radix: radix.to_string(),
                total_cycles: total_cycles as u64,
                bandwidth_gbps,
                energy_per_op_pj,
                dynamic_power_w,
                hotspot_c,
                bytes_on_bus,
            })
        };

    let discrete_int8 = evaluate(opts.pcie_gbps, false, "int8")?;
    let discrete_q4 = evaluate(opts.pcie_gbps, false, "q4")?;
    let unified_int8 = evaluate(opts.lpddr_gbps, true, "int8")?;
    let unified_q4 = evaluate(opts.lpddr_gbps, true, "q4")?;
    let delta = CompareDelta {
        cycle_reduction_pct: ((discrete_int8.total_cycles as f64
            - unified_int8.total_cycles as f64)
            / discrete_int8.total_cycles as f64)
            * 100.0,
        energy_reduction_pct: ((discrete_int8.energy_per_op_pj - unified_int8.energy_per_op_pj)
            / discrete_int8.energy_per_op_pj)
            * 100.0,
        q4_bandwidth_gain_pct: ((unified_q4.bandwidth_gbps - unified_int8.bandwidth_gbps)
            / unified_int8.bandwidth_gbps)
            * 100.0,
    };
    Ok(CompareResult {
        discrete_int8,
        discrete_q4,
        unified_int8,
        unified_q4,
        delta,
        badge: SOC_BADGE,
    })
}

/// Relative switching energy factor — JS `alphaEnergy(radix)`.
#[must_use]
pub fn alpha_energy(radix: &str) -> f64 {
    if radix == "q4" {
        0.9
    } else {
        1.2
    }
}

// ---------------------------------------------------------------------------
// Closed-loop DVFS thermal-throttling controller.
// ---------------------------------------------------------------------------

/// One DVFS control-cycle outcome.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DvfsAction {
    /// `T_junction ≤ 125 °C` and inside the hysteresis band — hold frequency.
    Hold,
    /// `T_junction > 125 °C` — one stepwise reduction applied.
    ThrottleDown,
    /// `T_junction < 125 °C − 5 °C` hysteresis band — one recovery step.
    ThrottleUp,
    /// `T_junction ≥ 200 °C` — `si_thermal_runaway` latched (06 JSON).
    Runaway,
}

/// A recorded DVFS control event (bounded ring).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DvfsEvent {
    pub cycle_index: u64,
    pub action: DvfsAction,
    /// Max tile junction temperature of the control cycle (°C).
    pub t_junction_c: f64,
    pub freq_before_mhz: f64,
    pub freq_after_mhz: f64,
}

/// Closed-loop DVFS thermal-throttling controller.
///
/// Contract (§3 + PHASE_04): `T_junction > 125 °C` ⇒ one **stepwise**
/// reduction per control cycle (`DVFS_STEP_PCT`, floored at `freq_min_mhz`);
/// `T_junction ≥ 200 °C` ⇒ a `si_thermal_runaway` event (the 06-JSON failure
/// name) **latches** the controller — it then refuses further operation
/// (fail-closed: a runaway destroyed the device) until `reset_runaway()`;
/// recovery steps only run once `T_junction` has fallen below the
/// `125 °C − 5 °C` hysteresis band (no DVFS oscillation).
#[derive(Debug, Clone)]
pub struct DvfsController {
    base_freq_mhz: f64,
    freq_min_mhz: f64,
    freq_mhz: f64,
    cycle_index: u64,
    runaway_latched: bool,
    events: Vec<DvfsEvent>,
    event_cursor: usize,
    event_capacity: usize,
}

impl DvfsController {
    /// New controller at `base_freq_mhz`, floored at `freq_min_mhz`.
    ///
    /// # Errors
    /// [`SocError::NonPhysicalInput`] for non-physical frequencies.
    pub fn new(base_freq_mhz: f64, freq_min_mhz: f64) -> Result<Self, SocError> {
        for (name, v) in [
            ("base frequency", base_freq_mhz),
            ("minimum frequency", freq_min_mhz),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(SocError::NonPhysicalInput(format!("{name} = {v} MHz")));
            }
        }
        if freq_min_mhz > base_freq_mhz {
            return Err(SocError::NonPhysicalInput(format!(
                "min frequency {freq_min_mhz} MHz above base {base_freq_mhz} MHz"
            )));
        }
        Ok(Self {
            base_freq_mhz,
            freq_min_mhz,
            freq_mhz: base_freq_mhz,
            cycle_index: 0,
            runaway_latched: false,
            events: Vec::new(),
            event_cursor: 0,
            event_capacity: 64,
        })
    }

    fn record(&mut self, event: DvfsEvent) {
        if self.events.len() < self.event_capacity {
            self.events.push(event);
        } else {
            self.events[self.event_cursor] = event;
            self.event_cursor = (self.event_cursor + 1) % self.event_capacity;
        }
    }

    /// One closed-loop control cycle over the tile junction temps (°C).
    /// The feed is a closure so the loop couples to real `ep-thermal` tile
    /// temps (see `tests/soc_oracle.rs` for the live-grid coupling).
    ///
    /// # Errors
    /// [`SocError::RunawayLatched`] once a runaway has been latched;
    /// [`SocError::NonPhysicalInput`] for non-physical tile temps.
    pub fn step(&mut self, tile_temps_c: &[f64]) -> Result<DvfsEvent, SocError> {
        if self.runaway_latched {
            return Err(SocError::RunawayLatched);
        }
        if tile_temps_c.is_empty() {
            return Err(SocError::NonPhysicalInput(
                "empty tile temperature feed".into(),
            ));
        }
        let t_junction = tile_temps_c.iter().fold(f64::NEG_INFINITY, |acc, t| {
            if t.is_finite() && *t > acc {
                *t
            } else {
                acc
            }
        });
        if !t_junction.is_finite() {
            return Err(SocError::NonPhysicalInput(
                "non-finite tile temperatures".into(),
            ));
        }
        self.cycle_index += 1;
        let before = self.freq_mhz;
        let (action, after) = if t_junction >= RUNAWAY_THRESHOLD_C {
            self.runaway_latched = true;
            (DvfsAction::Runaway, self.freq_mhz)
        } else if t_junction > THROTTLE_THRESHOLD_C {
            let stepped = (self.freq_mhz * (1.0 - DVFS_STEP_PCT / 100.0)).max(self.freq_min_mhz);
            self.freq_mhz = stepped;
            (DvfsAction::ThrottleDown, stepped)
        } else if t_junction < THROTTLE_THRESHOLD_C - DVFS_HYSTERESIS_C {
            let stepped = ((self.freq_mhz / (1.0 - DVFS_STEP_PCT / 100.0)).min(self.base_freq_mhz))
                .max(self.freq_min_mhz);
            if (stepped - self.freq_mhz).abs() < f64::EPSILON {
                (DvfsAction::Hold, self.freq_mhz)
            } else {
                self.freq_mhz = stepped;
                (DvfsAction::ThrottleUp, stepped)
            }
        } else {
            (DvfsAction::Hold, self.freq_mhz)
        };
        let event = DvfsEvent {
            cycle_index: self.cycle_index,
            action,
            t_junction_c: t_junction,
            freq_before_mhz: before,
            freq_after_mhz: after,
        };
        self.record(event);
        Ok(event)
    }

    /// Current frequency (MHz).
    #[must_use]
    pub const fn freq_mhz(&self) -> f64 {
        self.freq_mhz
    }

    /// Base frequency (MHz).
    #[must_use]
    pub const fn base_freq_mhz(&self) -> f64 {
        self.base_freq_mhz
    }

    /// Whether a `si_thermal_runaway` is latched (fail-closed halt).
    #[must_use]
    pub const fn runaway_latched(&self) -> bool {
        self.runaway_latched
    }

    /// Recorded events (bounded ring, oldest first).
    #[must_use]
    pub fn events(&self) -> &[DvfsEvent] {
        &self.events
    }

    /// Clear the runaway latch after an explicit operator reset.
    pub fn reset_runaway(&mut self) {
        self.runaway_latched = false;
        self.freq_mhz = self.base_freq_mhz;
    }
}
