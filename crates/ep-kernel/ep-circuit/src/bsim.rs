//! WP-2.2 — BSIM-level calibrated MOSFET evaluation for the kernel MNA solver.
//!
//! Id/gm/gds come from ngspice-characterized interpolation tables extracted from
//! the SKY130A public BSIM4 models (`nfet_01v8`/`pfet_01v8` at the pinned
//! SkyWater commit) and the FreePDK45 PTM nominal BSIM4 cards, both
//! byte-verified by SHA-256 (see `bsim_tables.rs` for the full provenance
//! ledger). Below/above the characterized envelope the evaluator falls back to
//! the extracted square-law core (the Shichman–Hodges parity model), exactly
//! as the phase playbook prescribes: "Kernel interpolates tables; fallback
//! square-law below table bounds".
//!
//! Gate capacitances `Cgs`/`Cgd` are **charge-conserving**: they are derived
//! from explicit terminal charge functions `Qg(vsg, vsd)`, `Qd`, `Qs` with
//! `Qg + Qd + Qs ≡ 0` by construction (Ward–Dutton style partition of the
//! channel charge between the channel end points), never from Meyer-type
//! per-capacitance derivatives. Conservation is proven in `bsim_tests`
//! (gauge invariance, closed-loop `∮ C·dV = 0`, row-sums of the
//! transcapacitance matrix).
//!
//! Drain/source symmetry mirrors the Shichman–Hodges operating point exactly:
//! the model is evaluated on the effective (drive, |Vds|) coordinates and the
//! roles swap continuously when the assigned terminals reverse
//! (`in_is_drain`), so there is no Newton oscillation at the reverse boundary.
//!
//! Honest capability boundary: the ASAP7 profile carries its envelope
//! parameters (nominal VDD 0.7 V, fin geometry from the `02`/`08` reference
//! data) but **no device table**: the local ngspice 47 build does not include
//! BSIM-CMG (`level=72`), the model class of the public ASAP7 cards, and
//! inventing FinFET parameters would violate the zero-invented-constants law.
//! Instantiating ASAP7 BSIM devices in a netlist is refused fail-closed at
//! admission time with an explicit reason (Level-1 `INSUFFICIENT_MODEL_DATA`
//! honesty ladder) instead of silently degrading.

use crate::bsim_tables::{
    BsimTable, FP45_NFET_PROBES, FP45_NFET_TABLE, FP45_PFET_PROBES, FP45_PFET_TABLE,
    SKY130A_NFET_PROBES, SKY130A_NFET_TABLE, SKY130A_PFET_PROBES, SKY130A_PFET_TABLE,
    VACUUM_PERMITTIVITY_F_PER_M,
};
use crate::mosfet::{MosfetParams, MosfetType};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

/// MOSFET compact-model selector carried by [`MosfetParams`](crate::mosfet::MosfetParams).
///
/// The default is `ShichmanHodges` — the WP-2.1 parity model — so every
/// pre-existing netlist solves bit-identically. `Bsim` selects the
/// ngspice-calibrated level of this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MosfetModel {
    /// Symmetric drain/source square law (parity oracle: `mna.mjs`).
    #[default]
    ShichmanHodges,
    /// BSIM-level calibrated evaluation for a characterized device profile.
    Bsim(BsimModelId),
}

/// Identifies a characterized (or envelope-only) device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BsimModelId {
    Sky130aNfet01v8,
    Sky130aPfet01v8,
    FreePdk45Nfet,
    FreePdk45Pfet,
    Asap7NfetRvt,
    Asap7PfetRvt,
}

/// Failure modes of the BSIM evaluator (honest, no silent degradation).
#[derive(Debug, Clone, PartialEq)]
pub enum BsimError {
    /// Profile has no characterized table (ASAP7: BSIM-CMG unavailable in
    /// ngspice 47 — see module docs). Carries the fail-closed reason.
    NotCharacterized(&'static str),
    /// Non-physical geometry (W/L must be finite and positive).
    BadGeometry(f64, f64),
}

impl std::fmt::Display for BsimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BsimError::NotCharacterized(r) => write!(f, "not characterized: {r}"),
            BsimError::BadGeometry(w, l) => write!(f, "bad geometry: w={w} l={l}"),
        }
    }
}

/// Operating region classification (telemetry only — indicative thresholds,
/// not the interpolation itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Cutoff,
    Subthreshold,
    Linear,
    Saturation,
}

/// Full BSIM operating point in **effective terminal coordinates** (drive
/// voltage `vgs_eff ≥ 0`, magnitude `vds_eff ≥ 0`). When the assigned
/// terminals were swapped, `in_is_drain` reports that the current enters at
/// the *physical* drain (mirroring the Shichman–Hodges convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsimOp {
    /// Conduction current (A), positive into the in-terminal.
    pub i: f64,
    /// ∂Id/∂Vgs in the effective frame (S).
    pub gm: f64,
    /// ∂Id/∂Vds in the effective frame (S).
    pub gds: f64,
    /// Gate–source terminal capacitance, charge-conserving (F).
    pub cgs: f64,
    /// Gate–drain terminal capacitance, charge-conserving (F).
    pub cgd: f64,
    pub in_is_drain: bool,
    pub region: Region,
}

/// SOA telemetry for one device bias (SKY130 core ceiling is 1.95 V per the
/// verified `02` reference data; other PDKs use their nominal-VDD envelope).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoaTelemetry {
    pub vgs_v: f64,
    pub ceiling_v: f64,
    pub margin_v: f64,
    pub violated: bool,
    pub warn: bool,
}

impl SoaTelemetry {
    pub fn status(&self) -> &'static str {
        if self.violated {
            "VIOLATED"
        } else if self.warn {
            "WARN"
        } else {
            "OK"
        }
    }
}

// ---------------------------------------------------------------------------
// Profile registry
// ---------------------------------------------------------------------------

/// Cached FC-derivatives of one characterization table.
///
/// Interpolation spaces (deliberate, and covered by the differential tests):
/// - **Vgs direction: log-domain** — Id spans decades across the subthreshold
///   transition, so the monotone cubic must run on ln(Id).
/// - **Vds direction: linear domain** — Id is smooth and near-linear in Vds
///   within every cell (a linear function is reproduced *exactly* by a
///   Hermite cubic with secant tangents), so linear-domain interpolation is
///   both more accurate and C1 here; log-domain Vds interpolation was
///   evaluated and rejected for distorting the deep-linear region.
pub struct Precomputed {
    /// ln(Id) nodes, row-major [vgs][vds].
    ln_id: Vec<f64>,
    /// Fritsch–Carlson node tangents along Vgs (per volt) for every Vds column: [vds][vgs].
    d_vgs: Vec<Vec<f64>>,
    /// Fritsch–Carlson node tangents along Vds (per volt, LINEAR domain)
    /// for every Vgs row over columns 1..n: [vgs][vds-1].
    d_vds: Vec<Vec<f64>>,
}

impl Precomputed {
    fn build(table: &BsimTable) -> Precomputed {
        let nv = table.vgs.len();
        let nd = table.vds.len();
        const LN_FLOOR: f64 = 1e-18;
        let mut ln_id = vec![0.0; nv * nd];
        for i in 0..nv {
            for j in 0..nd {
                ln_id[i * nd + j] = table.id[i * nd + j].max(LN_FLOOR).ln();
            }
        }
        // Vgs-direction slopes per column (column 0 exists but is only used by
        // the deep-linear origin segment through column 1).
        let mut d_vgs = Vec::with_capacity(nd);
        for j in 0..nd {
            let col: Vec<f64> = (0..nv).map(|i| ln_id[i * nd + j]).collect();
            let dx: Vec<f64> = (0..nv - 1)
                .map(|i| table.vgs[i + 1] - table.vgs[i])
                .collect();
            d_vgs.push(fc_slopes(&col, &dx));
        }
        // Vds-direction slopes per row (columns 1..n-1, linear domain).
        let mut d_vds = Vec::with_capacity(nv);
        for i in 0..nv {
            let row: Vec<f64> = (1..nd).map(|j| table.id[i * nd + j]).collect();
            let dx: Vec<f64> = (1..nd - 1)
                .map(|j| table.vds[j + 1] - table.vds[j])
                .collect();
            d_vds.push(fc_slopes(&row, &dx));
        }
        Precomputed {
            ln_id,
            d_vgs,
            d_vds,
        }
    }
}

/// Fritsch–Carlson monotone-preserving cubic node tangents, spacing-aware
/// (tangents are per unit of the actual x axis).
fn fc_slopes(y: &[f64], dx: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 1 {
        return vec![0.0];
    }
    let mut m = vec![0.0; n];
    if n == 2 {
        let d = (y[1] - y[0]) / dx[0];
        m[0] = d;
        m[1] = d;
        return m;
    }
    let mut delta = vec![0.0; n - 1]; // per-x secants
    for (i, d) in delta.iter_mut().enumerate() {
        *d = (y[i + 1] - y[i]) / dx[i];
    }
    // Weighted harmonic secants (Steffen-style interior tangents).
    for i in 1..n - 1 {
        if delta[i - 1] * delta[i] <= 0.0 {
            m[i] = 0.0;
        } else {
            let w1 = 2.0 * delta[i] + delta[i - 1];
            let w2 = delta[i] + 2.0 * delta[i - 1];
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    // Fritsch–Carlson limiting: keep the cubic monotone on every cell. The
    // limiter acts on tangent/secant ratios, which are spacing-independent.
    for i in 0..n - 1 {
        if delta[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let a = m[i] / delta[i];
        let b = m[i + 1] / delta[i];
        let s = a * a + b * b;
        if s > 9.0 {
            let tau = 3.0 / s.sqrt();
            m[i] = tau * a * delta[i];
            m[i + 1] = tau * b * delta[i];
        }
    }
    m
}

/// Cubic Hermite value and per-x derivative over a cell of span `h`.
/// Tangents are per unit x (the same axis the cell lives in).
fn hermite(y0: f64, y1: f64, m0: f64, m1: f64, t: f64, h: f64) -> (f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let v = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + h * (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + h * (t3 - t2) * m1;
    let d = ((6.0 * t2 - 6.0 * t) * y0 + (-6.0 * t2 + 6.0 * t) * y1) / h
        + (3.0 * t2 - 4.0 * t + 1.0) * m0
        + (3.0 * t2 - 2.0 * t) * m1;
    (v, d)
}

/// Device profile: one characterized PDK device (or the ASAP7 envelope-only
/// profile whose evaluation fails closed).
pub struct BsimProfile {
    pub id: BsimModelId,
    pub pdk: &'static str,
    pub device: &'static str,
    pub mosfet_type: MosfetType,
    /// Nominal VDD (V) — the `02` reference-data envelope.
    pub vdd_nominal_v: f64,
    /// Vgs SOA ceiling (V); `Some` only where the verified reference data
    /// declares one (SKY130 core: 1.95 V).
    pub soa_vgs_ceiling_v: Option<f64>,
    /// Characterization geometry (µm).
    pub w_cal_um: f64,
    pub l_cal_um: f64,
    /// Cox = epsrox·ε0/tox from the pinned model cards (F/m²).
    pub cox_f_per_m2: f64,
    /// Gate overlap capacitances per unit width from the pinned cards (F/m).
    pub cgso_f_per_m: f64,
    pub cgdo_f_per_m: f64,
    /// Measured channel-charge scale: the ratio of the ngspice-reported
    /// saturation Cgs (minus overlap) to the textbook `(2/3)·Cox·W·L` channel
    /// term, averaged over the characterized saturation bias grid. Absorbs
    /// BSIM4 `lint/dlc` geometry reductions and charge-model internals; the
    /// charge functions below multiply the channel charge by it, which keeps
    /// `Qg + Qd + Qs ≡ 0` exact.
    pub channel_charge_scale: f64,
    /// Measured subthreshold ideality factor n (swing / 59.5 mV·dec⁻¹).
    pub n_sub: f64,
    /// Measured constant-current threshold (0.1 µA/µm·W at the first
    /// nonzero-Vds column), magnitude convention (positive for both types).
    pub vth_extracted_v: f64,
    /// Square-law fallback (extracted at the top table edge) used below the
    /// characterized Vgs bound, per the phase playbook.
    pub sh_fallback: MosfetParams,
    /// Characterization temperature (K) — the ngspice run temp.
    pub temp_k: f64,
    pub table: Option<&'static BsimTable>,
    pub probes: &'static [(f64, f64, f64)],
    precomp: OnceLock<Precomputed>,
}

impl BsimProfile {
    fn precomp(&self) -> &Precomputed {
        self.precomp
            .get_or_init(|| Precomputed::build(self.table.expect("precomp requires a table")))
    }
}

/// CODATA 2022 exact constants (anchored to `data/reference-library/01` via
/// `ep-core::reference` in the tests).
const BOLTZMANN_K: f64 = 1.380649e-23;
const ELEMENTARY_CHARGE_Q: f64 = 1.602176634e-19;

static SKY130A_NFET: BsimProfile = BsimProfile {
    id: BsimModelId::Sky130aNfet01v8,
    pdk: "SKY130A",
    device: "sky130_fd_pr__nfet_01v8 (tt, bin 0)",
    mosfet_type: MosfetType::N,
    vdd_nominal_v: 1.8,
    soa_vgs_ceiling_v: Some(1.95),
    w_cal_um: 1.26,
    l_cal_um: 0.15,
    cox_f_per_m2: 3.9 * VACUUM_PERMITTIVITY_F_PER_M / 4.148e-9,
    cgso_f_per_m: 2.449068e-10,
    cgdo_f_per_m: 2.449068e-10,
    channel_charge_scale: 0.476,
    n_sub: 1.429,
    vth_extracted_v: 0.5967,
    sh_fallback: MosfetParams {
        mosfet_type: MosfetType::N,
        vth: 0.5967,
        kp: 8.5439e-4,
        lambda: 0.1104,
        model: MosfetModel::ShichmanHodges,
        w_um: 1.26,
        l_um: 0.15,
    },
    temp_k: 303.15,
    table: Some(&SKY130A_NFET_TABLE),
    probes: &SKY130A_NFET_PROBES,
    precomp: OnceLock::new(),
};

static SKY130A_PFET: BsimProfile = BsimProfile {
    id: BsimModelId::Sky130aPfet01v8,
    pdk: "SKY130A",
    device: "sky130_fd_pr__pfet_01v8 (tt, bin 0)",
    mosfet_type: MosfetType::P,
    vdd_nominal_v: 1.8,
    soa_vgs_ceiling_v: Some(1.95),
    w_cal_um: 1.26,
    l_cal_um: 0.15,
    cox_f_per_m2: 3.9 * VACUUM_PERMITTIVITY_F_PER_M / 4.23e-9,
    cgso_f_per_m: 5.248925e-11,
    cgdo_f_per_m: 5.248925e-11,
    channel_charge_scale: 0.869,
    n_sub: 2.454,
    vth_extracted_v: 0.6213,
    sh_fallback: MosfetParams {
        mosfet_type: MosfetType::P,
        vth: -0.6213,
        kp: 3.60276e-4,
        lambda: 0.1345,
        model: MosfetModel::ShichmanHodges,
        w_um: 1.26,
        l_um: 0.15,
    },
    temp_k: 303.15,
    table: Some(&SKY130A_PFET_TABLE),
    probes: &SKY130A_PFET_PROBES,
    precomp: OnceLock::new(),
};

static FP45_NFET: BsimProfile = BsimProfile {
    id: BsimModelId::FreePdk45Nfet,
    pdk: "FreePDK45",
    device: "NMOS_VTH (PTM 45 nm nominal BSIM4 v4.0)",
    mosfet_type: MosfetType::N,
    vdd_nominal_v: 1.0,
    soa_vgs_ceiling_v: None,
    w_cal_um: 0.1,
    l_cal_um: 0.05,
    cox_f_per_m2: 3.9 * VACUUM_PERMITTIVITY_F_PER_M / 1.63e-9,
    cgso_f_per_m: 1.1e-10,
    cgdo_f_per_m: 1.1e-10,
    channel_charge_scale: 1.106,
    n_sub: 1.447,
    vth_extracted_v: 0.3847,
    sh_fallback: MosfetParams {
        mosfet_type: MosfetType::N,
        vth: 0.3847,
        kp: 2.82944e-4,
        lambda: 0.3366,
        model: MosfetModel::ShichmanHodges,
        w_um: 0.1,
        l_um: 0.05,
    },
    temp_k: 300.15,
    table: Some(&FP45_NFET_TABLE),
    probes: &FP45_NFET_PROBES,
    precomp: OnceLock::new(),
};

static FP45_PFET: BsimProfile = BsimProfile {
    id: BsimModelId::FreePdk45Pfet,
    pdk: "FreePDK45",
    device: "PMOS_VTH (PTM 45 nm nominal BSIM4 v4.0)",
    mosfet_type: MosfetType::P,
    vdd_nominal_v: 1.0,
    soa_vgs_ceiling_v: None,
    w_cal_um: 0.1,
    l_cal_um: 0.05,
    cox_f_per_m2: 3.9 * VACUUM_PERMITTIVITY_F_PER_M / 1.63e-9,
    cgso_f_per_m: 1.1e-10,
    cgdo_f_per_m: 1.1e-10,
    // FreePDK45 PTM cards set capmod=2 with no characterized ngspice caps
    // capture yet; the textbook scale (1.0) is used and the caps sanity
    // envelope test skips this profile (conservation still holds exactly).
    channel_charge_scale: 1.0,
    n_sub: 1.454,
    vth_extracted_v: 0.3434,
    sh_fallback: MosfetParams {
        mosfet_type: MosfetType::P,
        vth: -0.3434,
        kp: 1.63127e-4,
        lambda: 0.2448,
        model: MosfetModel::ShichmanHodges,
        w_um: 0.1,
        l_um: 0.05,
    },
    temp_k: 300.15,
    table: Some(&FP45_PFET_TABLE),
    probes: &FP45_PFET_PROBES,
    precomp: OnceLock::new(),
};

static ASAP7_NFET: BsimProfile = BsimProfile {
    id: BsimModelId::Asap7NfetRvt,
    pdk: "ASAP7",
    device: "nmos_rvt (BSIM-CMG level=72 — no ngspice-47 support)",
    mosfet_type: MosfetType::N,
    vdd_nominal_v: 0.7,
    soa_vgs_ceiling_v: None,
    w_cal_um: 0.0,
    l_cal_um: 0.021,
    cox_f_per_m2: 0.0,
    cgso_f_per_m: 0.0,
    cgdo_f_per_m: 0.0,
    channel_charge_scale: 0.0,
    n_sub: 0.0,
    vth_extracted_v: 0.0,
    sh_fallback: MosfetParams {
        mosfet_type: MosfetType::N,
        vth: 0.0,
        kp: 0.0,
        lambda: 0.0,
        model: MosfetModel::ShichmanHodges,
        w_um: 0.0,
        l_um: 0.021,
    },
    temp_k: 300.15,
    table: None,
    probes: &[],
    precomp: OnceLock::new(),
};

static ASAP7_PFET: BsimProfile = BsimProfile {
    id: BsimModelId::Asap7PfetRvt,
    pdk: "ASAP7",
    device: "pmos_rvt (BSIM-CMG level=72 — no ngspice-47 support)",
    mosfet_type: MosfetType::P,
    vdd_nominal_v: 0.7,
    soa_vgs_ceiling_v: None,
    w_cal_um: 0.0,
    l_cal_um: 0.021,
    cox_f_per_m2: 0.0,
    cgso_f_per_m: 0.0,
    cgdo_f_per_m: 0.0,
    channel_charge_scale: 0.0,
    n_sub: 0.0,
    vth_extracted_v: 0.0,
    sh_fallback: MosfetParams {
        mosfet_type: MosfetType::P,
        vth: 0.0,
        kp: 0.0,
        lambda: 0.0,
        model: MosfetModel::ShichmanHodges,
        w_um: 0.0,
        l_um: 0.021,
    },
    temp_k: 300.15,
    table: None,
    probes: &[],
    precomp: OnceLock::new(),
};

/// ASAP7 fail-closed reason (referenced by the admission path).
pub const ASAP7_NOT_CHARACTERIZED: &str =
    "ASAP7 nmos_rvt/pmos_rvt are BSIM-CMG (level=72) devices; \
the pinned ngspice 47 build does not support level 72, so no ngspice-characterized table exists. \
Refusing fail-closed instead of inventing FinFET parameters (Level-1 INSUFFICIENT_MODEL_DATA).";

/// Look up a device profile by model id.
pub fn profile(id: BsimModelId) -> &'static BsimProfile {
    match id {
        BsimModelId::Sky130aNfet01v8 => &SKY130A_NFET,
        BsimModelId::Sky130aPfet01v8 => &SKY130A_PFET,
        BsimModelId::FreePdk45Nfet => &FP45_NFET,
        BsimModelId::FreePdk45Pfet => &FP45_PFET,
        BsimModelId::Asap7NfetRvt => &ASAP7_NFET,
        BsimModelId::Asap7PfetRvt => &ASAP7_PFET,
    }
}

/// True when the profile can evaluate Id (has a characterized table).
pub fn is_characterized(id: BsimModelId) -> bool {
    profile(id).table.is_some()
}

// ---------------------------------------------------------------------------
// Table interpolation (log-domain monotone cubic, C1 surface)
// ---------------------------------------------------------------------------

/// Evaluate the characterized table on magnitude coordinates with first-order
/// W/L scaling. `vgs_eff` is the effective drive magnitude (vgs for N, vsg for
/// P; both handled by the caller), `vds_eff` the drain magnitude. Returns
/// `(id, gm, gds)` in the effective frame.
fn eval_table(
    p: &BsimProfile,
    pre: &Precomputed,
    vgs_eff: f64,
    vds_eff: f64,
    wl_scale: f64,
) -> (f64, f64, f64) {
    let table = p.table.expect("eval_table requires a table");
    let nv = table.vgs.len();
    let nd = table.vds.len();
    let vgs_lo = table.vgs[0];
    let vds1 = table.vds[1];
    let vds_hi = table.vds[nd - 1];

    // --- Vgs below the characterized bound: extracted square-law fallback ---
    if vgs_eff < vgs_lo {
        let q = &p.sh_fallback;
        let drive = (vgs_eff - p.vth_extracted_v).max(0.0);
        let id = 0.5 * q.kp * drive * drive * (1.0 + q.lambda * vds_eff.max(0.0)) * wl_scale;
        let gm = q.kp * drive * (1.0 + q.lambda * vds_eff.max(0.0)) * wl_scale;
        let gds = 0.5 * q.kp * drive * drive * q.lambda * wl_scale;
        return (id, gm, gds);
    }

    // Row-bracketing helpers (tangents blend linearly across the two bracketing
    // Vgs rows; the blend weights are independent of Vds so gm stays C1).
    let row_bracket = |x: f64| -> (usize, usize, f64) {
        let mut hi = nv - 1;
        for (i, &g) in table.vgs.iter().enumerate() {
            if g >= x {
                hi = i;
                break;
            }
        }
        let lo = if x >= table.vgs[nv - 1] {
            nv - 2
        } else {
            hi.saturating_sub(1)
        };
        let (i0, i1) = (lo.min(nv - 2), lo + 1);
        let tv = ((x - table.vgs[i0]) / (table.vgs[i1] - table.vgs[i0])).clamp(0.0, 1.0);
        (i0, i1, tv)
    };

    // --- Column evaluation: monotone cubic of ln(Id) along Vgs.
    // Returns (ln Id, d ln Id / dVgs). Tangents are per volt; hermite()
    // handles the cell span.
    let column_at = |j: usize, x: f64| -> (f64, f64) {
        let col = &pre.d_vgs[j];
        let (i0, i1, t) = row_bracket(x);
        let h = table.vgs[i1] - table.vgs[i0];
        let y0 = pre.ln_id[i0 * nd + j];
        let y1 = pre.ln_id[i1 * nd + j];
        hermite(y0, y1, col[i0], col[i1], t, h)
    };

    // --- Deep-linear origin segment [0, vds1]: linear-from-origin with C1 join.
    // Id is interpolated directly (linear domain): exact reproduction of the
    // deep-linear channel behaviour, tangent at the origin = Id1/vds1 (the
    // channel conductance), tangent at vds1 joined to the interior FC tangent.
    // The tangents' Vgs dependence enters only through Id1(vgs) and its
    // conductance shape s1 = m1/Id1 (slowly varying with Vgs; drift neglected,
    // bounded by the derivative-consistency test).
    if vds_eff <= vds1 {
        let (i0, i1, tv) = row_bracket(vgs_eff);
        let m1_volt = pre.d_vds[i0][0] * (1.0 - tv) + pre.d_vds[i1][0] * tv;
        let (ln1v, dln1) = column_at(1, vgs_eff);
        let id1 = ln1v.exp();
        let s1 = m1_volt / id1; // conductance shape (1/V), near id1/vds1 / id1
        let t = (vds_eff / vds1).clamp(0.0, 1.0);
        let (id_u, gds_u) = hermite(0.0, id1, id1 / vds1, m1_volt, t, vds1);
        // gm = Id1'(vgs)·[h01 + h10 + h11·s1·vds1] (the hermite basis weights
        // the two Vgs-carrying terms Id1 and Id1/vds1, plus the shape term).
        let (h01, h10, h11) = (
            -2.0 * t * t * t + 3.0 * t * t,
            t * t * t - 2.0 * t * t + t,
            t * t * t - t * t,
        );
        let gm_u = id1 * dln1 * (h01 + h10 + h11 * s1 * vds1);
        return (id_u * wl_scale, gm_u * wl_scale, gds_u * wl_scale);
    }

    // --- Linear-domain extrapolation beyond the characterized Vds bound ---
    if vds_eff > vds_hi {
        let (lnh, dlnh) = column_at(nd - 1, vgs_eff);
        let (i0, i1, tv) = row_bracket(vgs_eff);
        let m_edge = pre.d_vds[i0][nd - 2] * (1.0 - tv) + pre.d_vds[i1][nd - 2] * tv;
        let id_hi = lnh.exp();
        let dv_over = vds_eff - vds_hi;
        let id_u = (id_hi + m_edge * dv_over).max(0.0);
        // d(m_edge)/dVgs neglected: m_edge tracks Id's shape (bounded by the
        // derivative-consistency test).
        let gm_u = id_hi * dlnh * (1.0 + (m_edge / id_hi) * dv_over);
        let gds_u = m_edge;
        return (id_u * wl_scale, gm_u * wl_scale, gds_u * wl_scale);
    }

    // --- Interior: bracketing columns, C1 blend in the linear domain ---
    let mut j1 = 1;
    for (j, &d) in table.vds.iter().enumerate().skip(1) {
        j1 = j;
        if d >= vds_eff {
            break;
        }
    }
    let j0 = (j1 - 1).max(1);
    let (ln_a, dln_a) = column_at(j0, vgs_eff);
    let (ln_b, dln_b) = column_at(j1, vgs_eff);
    let (id_a, id_b) = (ln_a.exp(), ln_b.exp());
    let dv = table.vds[j1] - table.vds[j0];
    let t = ((vds_eff - table.vds[j0]) / dv).clamp(0.0, 1.0);
    // Row-blended linear-domain tangents (per volt).
    let (i0, i1, tv) = row_bracket(vgs_eff);
    let m0 = pre.d_vds[i0][j0 - 1] * (1.0 - tv) + pre.d_vds[i1][j0 - 1] * tv;
    let m1 = pre.d_vds[i0][j1 - 1] * (1.0 - tv) + pre.d_vds[i1][j1 - 1] * tv;
    let (id_u, gds_u) = hermite(id_a, id_b, m0, m1, t, dv);
    // gm: differentiate the hermite value in Vgs. The two bracketing column
    // values carry Id·dlnId; the row-blended tangents are treated as
    // shape-preserving (m' ≈ m·dlnId at their own node), which is exact for
    // geometrically scaled rows and C1 everywhere.
    let (h00, h01, h10, h11) = hermite_basis(t);
    let gm_u = dln_a * (h00 * id_a + dv * h10 * m0) + dln_b * (h01 * id_b + dv * h11 * m1);
    (id_u * wl_scale, gm_u * wl_scale, gds_u * wl_scale)
}

/// Cubic Hermite basis at parameter t: value weights (h00, h01) and tangent
/// weights (h10, h11) of the standard form
/// `h00·y0 + h·h10·m0 + h01·y1 + h·h11·m1`.
fn hermite_basis(t: f64) -> (f64, f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        2.0 * t3 - 3.0 * t2 + 1.0,
        -2.0 * t3 + 3.0 * t2,
        t3 - 2.0 * t2 + t,
        t3 - t2,
    )
}

// ---------------------------------------------------------------------------
// Charge model (charge-conserving gate capacitances)
// ---------------------------------------------------------------------------

/// Smooth weak/strong-inversion unifier: `2·n·vt·ln(1 + exp(x / (2·n·vt)))`.
/// → x in strong inversion, decays exponentially below threshold.
fn vgeff(x: f64, n: f64, vt: f64) -> f64 {
    let s = x / (2.0 * n * vt);
    if s > 40.0 {
        x // exp(40) already dominates ln(1+e^s) to full f64 precision
    } else if s < -40.0 {
        2.0 * n * vt * s.exp()
    } else {
        2.0 * n * vt * (1.0 + s.exp()).ln()
    }
}

/// Saturation roll-off interpolation F(θ): 1 in the linear region, 2/3 in
/// saturation (Ward–Dutton saturation limit of the channel charge), θ a
/// smooth saturation measure.
fn charge_shape(theta: f64) -> f64 {
    (1.0 + theta / 3.0) / (1.0 + theta)
}

/// Explicit, difference-only terminal charges in magnitude coordinates
/// (N-type sign convention; the caller maps P devices). Returns `(Qg, Qd, Qs)`
/// with `Qg + Qd + Qs ≡ 0` identically.
///
/// Channel charge: `Qchan = cscale·Cox·W·L·G(vsg, vsd)` with
/// `G = ψs·F(θ) + ψd·(1 − F(θ))` (ψ = smooth inversion densities at the
/// channel ends, θ = saturation measure). Ward–Dutton partition:
/// `Qd/Qchan = (ψs + 2ψd) / (3(ψs + ψd))` — 1/2 at Vds = 0, 1/3 in
/// saturation (Tsividis; the classic linear-profile partition).
fn charges_mag(p: &BsimProfile, vsg: f64, vsd: f64, w_um: f64, l_um: f64) -> (f64, f64, f64) {
    let vt = BOLTZMANN_K * p.temp_k / ELEMENTARY_CHARGE_Q;
    let n = p.n_sub;
    let area_cox = p.channel_charge_scale * p.cox_f_per_m2 * w_um * 1e-6 * l_um * 1e-6;
    // The unifier runs on the gate OVERDRIVE (vgs − vth): the channel charge
    // tends to Cox·W·L·(vgs − vth) in strong inversion and decays
    // exponentially below threshold (ψ → 0), matching the physics at both
    // ends; the calibrated channel-charge scale absorbs the residual shape.
    let od = vsg - p.vth_extracted_v;
    let psi_s = vgeff(od, n, vt);
    let psi_d = vgeff(od - vsd, n, vt);
    // Smooth saturation measure: 0 at vsd = 0, → 1 as vsd passes the pinch-off.
    let theta = {
        let x = vsd / od.max(0.05);
        let s = (x * x).min(20.0);
        1.0 - (-s).exp()
    };
    let f = charge_shape(theta);
    let qchan = area_cox * (psi_s * f + psi_d * (1.0 - f)).max(0.0);
    let sum_pd = psi_s + psi_d;
    let qd_frac = if sum_pd > 0.0 {
        ((psi_s + 2.0 * psi_d) / (3.0 * sum_pd)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let qov_s = p.cgso_f_per_m * w_um * 1e-6 * vsg;
    let qov_d = p.cgdo_f_per_m * w_um * 1e-6 * (vsg - vsd);
    let qg = qchan + qov_s + qov_d;
    let qd = -qd_frac * qchan - p.cgdo_f_per_m * w_um * 1e-6 * (vsg - vsd);
    let qs = -(1.0 - qd_frac) * qchan - p.cgso_f_per_m * w_um * 1e-6 * vsg;
    (qg, qd, qs)
}

/// Terminal charges (Coulombs) at the given magnitude bias. Charge-conserving:
/// the sum is identically zero.
pub fn terminal_charges(
    p: &BsimProfile,
    vgs_eff: f64,
    vds_eff: f64,
    w_um: f64,
    l_um: f64,
) -> Result<(f64, f64, f64), BsimError> {
    check_geometry(p, w_um, l_um)?;
    Ok(charges_mag(
        p,
        vgs_eff.max(0.0),
        vds_eff.max(0.0),
        w_um,
        l_um,
    ))
}

/// Full charge-conserving transcapacitance matrix `C[i][j] = −∂Qi/∂Vj` over
/// terminal voltages `(Vg, Vd, Vs)` in magnitude coordinates (rows: terminal
/// charge index `g, d, s`; columns: voltage). The charges depend on voltage
/// differences only, so row sums vanish exactly — the gauge-conservation
/// property Meyer-type capacitances lack. With `Vs = 0` held as the magnitude
/// reference, `∂Qi/∂Vg = f₁`, `∂Qi/∂Vd = f₂`, `∂Qi/∂Vs = −f₁ − f₂`.
pub fn capacitance_matrix(
    p: &BsimProfile,
    vgs_eff: f64,
    vds_eff: f64,
    w_um: f64,
    l_um: f64,
) -> Result<[[f64; 3]; 3], BsimError> {
    check_geometry(p, w_um, l_um)?;
    const H: f64 = 1e-6;
    let q = |vsg: f64, vsd: f64| charges_mag(p, vsg.max(0.0), vsd.max(0.0), w_um, l_um);
    let (g_p, d_p, s_p) = q(vgs_eff + H, vds_eff);
    let (g_m, d_m, s_m) = q(vgs_eff - H, vds_eff);
    let (g_p2, d_p2, s_p2) = q(vgs_eff, vds_eff + H);
    let (g_m2, d_m2, s_m2) = q(vgs_eff, vds_eff - H);
    // f1 = ∂Qi/∂vsg, f2 = ∂Qi/∂vsd (central differences).
    let f1 = [g_p - g_m, d_p - d_m, s_p - s_m];
    let f2 = [g_p2 - g_m2, d_p2 - d_m2, s_p2 - s_m2];
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        c[i][0] = -f1[i] / (2.0 * H);
        c[i][1] = -f2[i] / (2.0 * H);
        c[i][2] = (f1[i] + f2[i]) / (2.0 * H);
    }
    Ok(c)
}

/// Charge-conserving gate capacitances in the SPICE terminal-charge
/// convention: `Cgs = −∂Qg/∂Vs` (Vg, Vd held) and `Cgd = −∂Qg/∂Vd` (Vg, Vs
/// held) — exactly the gate row of [`capacitance_matrix`].
pub fn gate_capacitances(
    p: &BsimProfile,
    vgs_eff: f64,
    vds_eff: f64,
    w_um: f64,
    l_um: f64,
) -> Result<(f64, f64), BsimError> {
    let c = capacitance_matrix(p, vgs_eff, vds_eff, w_um, l_um)?;
    Ok((c[0][2], c[0][1]))
}

fn check_geometry(p: &BsimProfile, w_um: f64, l_um: f64) -> Result<(), BsimError> {
    if !w_um.is_finite() || !l_um.is_finite() || w_um <= 0.0 || l_um <= 0.0 {
        return Err(BsimError::BadGeometry(w_um, l_um));
    }
    if p.table.is_none() {
        return Err(BsimError::NotCharacterized(ASAP7_NOT_CHARACTERIZED));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public evaluation entry points
// ---------------------------------------------------------------------------

/// Effective-frame evaluation (both magnitudes ≥ 0).
fn eval_eff(
    p: &BsimProfile,
    vgs_eff: f64,
    vds_eff: f64,
    w_um: f64,
    l_um: f64,
) -> Result<BsimOp, BsimError> {
    check_geometry(p, w_um, l_um)?;
    let wl_scale = (w_um / p.w_cal_um) * (p.l_cal_um / l_um);
    let pre = p.precomp();
    let (id, gm, gds) = eval_table(p, pre, vgs_eff.max(0.0), vds_eff.max(0.0), wl_scale);
    let (cgs, cgd) = gate_capacitances(p, vgs_eff, vds_eff, w_um, l_um)?;
    let region = if id < 1e-12 * wl_scale {
        Region::Cutoff
    } else if vgs_eff < p.vth_extracted_v {
        Region::Subthreshold
    } else {
        let vdsat_est = (vgs_eff - p.vth_extracted_v).clamp(0.05, p.vdd_nominal_v);
        if vds_eff < vdsat_est {
            Region::Linear
        } else {
            Region::Saturation
        }
    };
    Ok(BsimOp {
        i: id,
        gm,
        gds,
        cgs,
        cgd,
        in_is_drain: true,
        region,
    })
}

/// SOA telemetry against the profile ceiling (None when the reference data
/// declares no ceiling for this PDK).
pub fn soa_check(p: &BsimProfile, vgs_magnitude: f64) -> Option<SoaTelemetry> {
    let ceiling = p.soa_vgs_ceiling_v?;
    let margin = ceiling - vgs_magnitude;
    Some(SoaTelemetry {
        vgs_v: vgs_magnitude,
        ceiling_v: ceiling,
        margin_v: margin,
        violated: margin < 0.0,
        warn: margin < 0.05 * ceiling,
    })
}

/// Full operating point with the same symmetric drain/source contract as the
/// Shichman–Hodges `operating_point`: `vgs = vg − vs`, `vds = vd − vs` for N;
/// for P the magnitudes are `vsg = vs − vg`, `vsd = vs − vd` and conduction
/// enters at the source. Negative effective Vds swaps the roles continuously.
pub fn operating_point(
    p: &BsimProfile,
    vg: f64,
    vd: f64,
    vs: f64,
    w_um: f64,
    l_um: f64,
) -> Result<BsimOp, BsimError> {
    check_geometry(p, w_um, l_um)?;
    match p.mosfet_type {
        MosfetType::N => {
            let vgs = vg - vs;
            let vds = vd - vs;
            if vds >= 0.0 {
                Ok(eval_eff(p, vgs, vds, w_um, l_um)?)
            } else {
                // Roles swap: the lower terminal becomes the effective source.
                let mut op = eval_eff(p, vgs - vds, -vds, w_um, l_um)?;
                op.in_is_drain = false;
                Ok(op)
            }
        }
        MosfetType::P => {
            let vsg = vs - vg;
            let vsd = vs - vd;
            if vsd >= 0.0 {
                let mut op = eval_eff(p, vsg, vsd, w_um, l_um)?;
                op.in_is_drain = false;
                Ok(op)
            } else {
                let mut op = eval_eff(p, vsg - vsd, -vsd, w_um, l_um)?;
                op.in_is_drain = true;
                Ok(op)
            }
        }
    }
}

/// Legacy single-number API: conduction current entering the in-terminal.
pub fn mosfet_current(
    p: &BsimProfile,
    vg: f64,
    vd: f64,
    vs: f64,
    w_um: f64,
    l_um: f64,
) -> Result<f64, BsimError> {
    Ok(operating_point(p, vg, vd, vs, w_um, l_um)?.i)
}

// ---------------------------------------------------------------------------
// ep-core reference anchoring
// ---------------------------------------------------------------------------

/// Values the crate's profile constants must match against the SHA-256-verified
/// reference library (fail-closed anchoring, enforced by tests against the real
/// `data/reference-library` files through `ep-core::reference`).
pub struct ReferenceAnchor {
    /// `02` JSON: `parameters.sky130a.core_soa_max_vgs_v`.
    pub sky130_core_soa_max_vgs_v: f64,
    /// `02` JSON: `parameters.sky130a.nominal_vdd_v`.
    pub sky130_nominal_vdd_v: f64,
    /// `02` JSON: `parameters.sky130a.vth_nmos_typ_v`.
    pub sky130_vth_nmos_typ_v: f64,
    /// `02` JSON: `parameters.sky130a.vth_pmos_typ_v`.
    pub sky130_vth_pmos_typ_v: f64,
    /// `02` JSON: `parameters.sky130a.oxide_thickness_nm.gate_oxide` (nm).
    pub sky130_gate_oxide_nm: f64,
    /// `02` JSON: `parameters.freepdk45.nominal_vdd_v`.
    pub freepdk45_nominal_vdd_v: f64,
    /// `02` JSON: `parameters.asap7.nominal_vdd_v`.
    pub asap7_nominal_vdd_v: f64,
    /// `08` JSON: `parameters.sky130a_binary_cells.voltage_domain_v.soa_vgs_ceiling`.
    pub sky130_cell_soa_vgs_ceiling_v: f64,
    /// Verified file count from the manifest.
    pub verified_files: usize,
}

impl ReferenceAnchor {
    /// Parse and cross-check the profile constants against the verified `02`/`08`
    /// payloads (bytes already byte-verified against `MANIFEST.json` by `ep-core`).
    pub fn from_verified_bytes(
        json02: &str,
        json08: &str,
        verified_files: usize,
    ) -> Result<ReferenceAnchor, String> {
        #[derive(serde::Deserialize)]
        struct File02 {
            parameters: std::collections::BTreeMap<String, serde_json::Value>,
        }
        #[derive(serde::Deserialize)]
        struct File08 {
            parameters: std::collections::BTreeMap<String, serde_json::Value>,
        }
        let f02: File02 = serde_json::from_str(json02).map_err(|e| format!("02 parse: {e}"))?;
        let f08: File08 = serde_json::from_str(json08).map_err(|e| format!("08 parse: {e}"))?;
        let num = |m: &std::collections::BTreeMap<String, serde_json::Value>,
                   path: &[&str]|
         -> Result<f64, String> {
            let mut cur = m
                .get(path[0])
                .ok_or_else(|| format!("missing {}", path[0]))?;
            for k in &path[1..] {
                cur = cur
                    .get(k)
                    .ok_or_else(|| format!("missing {}.{}", path[..1].join("/"), k))?;
            }
            cur.as_f64()
                .ok_or_else(|| format!("not a number: {}", path.join(".")))
        };
        let anchor = ReferenceAnchor {
            sky130_core_soa_max_vgs_v: num(&f02.parameters, &["sky130a", "core_soa_max_vgs_v"])?,
            sky130_nominal_vdd_v: num(&f02.parameters, &["sky130a", "nominal_vdd_v"])?,
            sky130_vth_nmos_typ_v: num(&f02.parameters, &["sky130a", "vth_nmos_typ_v"])?,
            sky130_vth_pmos_typ_v: num(&f02.parameters, &["sky130a", "vth_pmos_typ_v"])?,
            sky130_gate_oxide_nm: num(
                &f02.parameters,
                &["sky130a", "oxide_thickness_nm", "gate_oxide"],
            )?,
            freepdk45_nominal_vdd_v: num(&f02.parameters, &["freepdk45", "nominal_vdd_v"])?,
            asap7_nominal_vdd_v: num(&f02.parameters, &["asap7", "nominal_vdd_v"])?,
            sky130_cell_soa_vgs_ceiling_v: num(
                &f08.parameters,
                &[
                    "sky130a_binary_cells",
                    "voltage_domain_v",
                    "soa_vgs_ceiling",
                ],
            )?,
            verified_files,
        };
        // Fail-closed cross-checks: the crate constants must match the verified data.
        let checks: [(&str, f64, f64); 7] = [
            (
                "SKY130A core SOA ceiling",
                anchor.sky130_core_soa_max_vgs_v,
                1.95,
            ),
            ("SKY130A nominal VDD", anchor.sky130_nominal_vdd_v, 1.8),
            (
                "SKY130A NMOS typical vth",
                anchor.sky130_vth_nmos_typ_v,
                0.55,
            ),
            (
                "SKY130A PMOS typical vth",
                anchor.sky130_vth_pmos_typ_v,
                -0.55,
            ),
            ("SKY130A gate oxide", anchor.sky130_gate_oxide_nm, 4.1),
            ("FreePDK45 nominal VDD", anchor.freepdk45_nominal_vdd_v, 1.0),
            ("ASAP7 nominal VDD", anchor.asap7_nominal_vdd_v, 0.7),
        ];
        for (name, data, expect) in checks {
            if (data - expect).abs() > 1e-9 {
                return Err(format!(
                    "reference data drifted from the anchored constant: {name} = {data} != {expect}"
                ));
            }
        }
        if (anchor.sky130_cell_soa_vgs_ceiling_v - 1.95).abs() > 1e-9 {
            return Err(format!(
                "08 cell SOA ceiling drifted: {} != 1.95",
                anchor.sky130_cell_soa_vgs_ceiling_v
            ));
        }
        Ok(anchor)
    }
}
