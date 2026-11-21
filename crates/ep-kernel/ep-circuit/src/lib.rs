//! `ep-circuit` — kernel MNA DC solver (WP-2.1) with BSIM-level calibrated
//! MOSFET evaluation (WP-2.2).
//!
//! Stamps: R, V, I sources, symmetric drain/source Shichman–Hodges MOSFET
//! (default, parity oracle `packages/physics-engine/src/mna.mjs`) or the
//! ngspice-calibrated BSIM level (`bsim`/`bsim_tables` — SKY130A
//! `nfet_01v8`/`pfet_01v8` and FreePDK45 PTM profiles, charge-conserving
//! Cgs/Cgd, 1.95 V SKY130 core SOA telemetry). Newton–Raphson with numeric
//! Jacobian, damped stepping, source stepping (λ ∈ {0.05..1.0}), consistent
//! gmin escalation. Admission is enforced through `ep-core` (≤64 nodes /
//! ≤32 voltage sources, fail-closed) before any allocation; uncharacterized
//! BSIM profiles are refused fail-closed. Parity oracle:
//! `packages/physics-engine/src/mna.mjs`.

pub mod bsim;
pub mod bsim_tables;
#[cfg(test)]
mod bsim_tests;
pub mod mosfet;
pub mod netlist;
pub mod solver;
#[cfg(test)]
mod tests;

pub use bsim::{
    capacitance_matrix, gate_capacitances, mosfet_current as bsim_mosfet_current,
    operating_point as bsim_operating_point, profile as bsim_profile, soa_check as bsim_soa_check,
    terminal_charges, BsimError, BsimModelId, BsimOp, MosfetModel, Region, SoaTelemetry,
};
pub use mosfet::{mosfet_current, operating_point, MosfetParams, MosfetType, OperatingPoint};
pub use netlist::{Element, ElementKind, Netlist};
pub use solver::{
    kcl_max_residual, solve_dc, solve_dc_default, BsimDeviceTelemetry, SolveError, SolveOptions,
    SolveResult,
};

use ep_core::admission::AdmissionController;
use std::sync::OnceLock;

static ADMISSION: OnceLock<AdmissionController> = OnceLock::new();

/// Process-wide admission controller with the default WP-1.2 budgets.
pub fn admission() -> &'static AdmissionController {
    ADMISSION.get_or_init(AdmissionController::new)
}

/// Convenience: solve against the process-wide default admission controller.
pub fn solve(netlist: &Netlist) -> Result<SolveResult, SolveError> {
    solve_dc(netlist, &SolveOptions::default(), admission())
}
