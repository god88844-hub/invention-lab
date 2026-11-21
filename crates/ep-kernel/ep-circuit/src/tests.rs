//! WP-2.1 parity tests — oracles captured verbatim from
//! `packages/physics-engine/src/mna.mjs` (the ±1e-9 parity contract).

use crate::mosfet::{MosfetParams, MosfetType};
use crate::{solve_dc, SolveError, SolveOptions};
use ep_core::admission::AdmissionController;

fn admission() -> AdmissionController {
    AdmissionController::new()
}

fn divider() -> crate::Netlist {
    let mut nl = crate::Netlist::new(&["gnd", "vout", "vdd"]);
    nl.add_v_source("vdd", "gnd", 1.8, "VDD");
    nl.add_resistor("vdd", "vout", 1000.0, "R1");
    nl.add_resistor("vout", "gnd", 2000.0, "R2");
    nl
}

fn inverter() -> crate::Netlist {
    let mut nl = crate::Netlist::new(&["gnd", "vdd", "vin", "vout"]);
    nl.add_v_source("vdd", "gnd", 1.8, "VDD");
    nl.add_resistor("vout", "gnd", 100000.0, "RLOAD");
    let mp = MosfetParams::square_law(MosfetType::P, -0.55, 0.004, 0.06);
    let mn = MosfetParams::square_law(MosfetType::N, 0.55, 0.004, 0.06);
    nl.add_mosfet("MP", "vin", "vout", "vdd", mp);
    nl.add_mosfet("MN", "vin", "vout", "gnd", mn);
    nl
}

#[test]
fn divider_exact_math_kcl_and_power_conservation() {
    let nl = divider();
    let s = solve_dc(&nl, &SolveOptions::default(), &admission()).unwrap();
    assert!(s.converged);
    // Exact analytic: vout = 1.8·R2/(R1+R2) = 1.2 V (±1e-9 parity contract).
    assert!(
        (s.node_voltages[1] - 1.2).abs() < 1e-9,
        "vout={}",
        s.node_voltages[1]
    );
    // ESM oracle (mna.mjs): 1.1999999992721153 — same Newton path.
    assert!((s.node_voltages[1] - 1.1999999992721153).abs() < 1e-9);
    // KCL at every node < 1e-9.
    assert!(crate::kcl_max_residual(&nl, &s) < 1e-9, "KCL residual");
    // Power conservation: ΣP_devices = ΣP_delivered < 1e-9.
    let p_devices = s.device_power["R1"] + s.device_power["R2"];
    let p_delivered = s.device_power["VDD"];
    assert!(
        (p_devices - p_delivered).abs() < 1e-9,
        "power balance {}",
        p_devices - p_delivered
    );
    // ESM power oracles.
    assert!((s.device_power["R1"] - 0.0003600000010656).abs() < 1e-12);
    assert!((s.device_power["VDD"] - 0.0010800000046804).abs() < 1e-12);
}

#[test]
fn inverter_sweep_matches_esm_oracles() {
    // ESM oracle values (packages/physics-engine/src/mna.mjs, converged ±1e-6 path).
    const ORACLE: [(f64, f64); 5] = [
        (0.0, 1.796402804428224),
        (0.6, 1.790923648463386),
        (0.9, 0.6715735864269204),
        (1.2, 0.002125553493024613),
        (1.8, 0.0),
    ];
    for (vin, expected) in ORACLE {
        let mut nl = inverter();
        nl.add_v_source("vin", "gnd", vin, "VIN");
        let s = solve_dc(&nl, &SolveOptions::default(), &admission()).unwrap();
        assert!(s.converged, "vin={vin} must converge");
        let vout = s.node_voltages[3];
        if vin < 0.9 {
            assert!(vout > 1.5, "vin={vin} → vout high, got {vout}");
        } else {
            assert!(vout < 0.8, "vin={vin} → vout low, got {vout}");
        }
        assert!(
            (vout - expected).abs() < 1e-6,
            "vin={vin}: rust vout={vout} vs ESM {expected}"
        );
        // KCL on the nonlinear circuit < 1e-9.
        assert!(crate::kcl_max_residual(&nl, &s) < 1e-9, "vin={vin} KCL");
    }
}

#[test]
fn symmetric_mosfet_operating_point_matches_esm_exactly() {
    let qn = MosfetParams::square_law(MosfetType::N, 0.55, 0.004, 0.06);
    let qp = MosfetParams::square_law(MosfetType::P, -0.55, 0.004, 0.06);
    // Linear region (drive 1.25 > vds 0.9): 0.004·(1.25·0.9 − 0.5·0.81)·1.054.
    let esm = 0.00303552_f64;
    let a = crate::mosfet::operating_point(&qn, 1.8, 0.9, 0.0);
    assert!((a.i - esm).abs() < 1e-15 && a.in_is_drain, "{}", a.i);
    // Reversed roles: same |current|, opposite in-terminal.
    let b = crate::mosfet::operating_point(&qn, 1.8, 0.0, 0.9);
    assert!((b.i - esm).abs() < 1e-15 && !b.in_is_drain);
    // PMOS: source-high normal conduction enters at the source.
    let c = crate::mosfet::operating_point(&qp, 0.0, 0.9, 1.8);
    assert!((c.i - esm).abs() < 1e-15 && !c.in_is_drain);
}

#[test]
fn shorted_gate_fault_drags_output_low() {
    let mp = MosfetParams::square_law(MosfetType::P, -0.55, 0.004, 0.06);
    let mn = MosfetParams::square_law(MosfetType::N, 0.55, 0.004, 0.06);
    let mut nl = crate::Netlist::new(&["gnd", "vdd", "vin", "vout"]);
    nl.add_v_source("vdd", "gnd", 1.8, "VDD");
    nl.add_resistor("vin", "gnd", 10000.0, "RDRIVE");
    nl.add_resistor("vout", "gnd", 100000.0, "RLOAD");
    nl.add_mosfet("MP", "vin", "vout", "vdd", mp);
    nl.add_mosfet("MN", "vin", "vout", "gnd", mn);
    let healthy = solve_dc(&nl, &SolveOptions::default(), &admission()).unwrap();
    assert!(
        healthy.node_voltages[3] > 1.5,
        "healthy vout={}",
        healthy.node_voltages[3]
    );
    // Trip the permanent fault: gate oxide shorted (1 Ω gate→source).
    nl.mosfet_mut("MP").unwrap().gate_shorted = true;
    let faulted = solve_dc(&nl, &SolveOptions::default(), &admission()).unwrap();
    assert!(faulted.converged);
    assert!(
        faulted.node_voltages[2] > 1.7,
        "gate dragged high: {}",
        faulted.node_voltages[2]
    );
    assert!(
        faulted.node_voltages[3] < 0.5,
        "output stuck low: {}",
        faulted.node_voltages[3]
    );
}

#[test]
fn admission_refuses_65_nodes_and_33_v_sources_fail_closed() {
    // 65 interned nodes (gnd + 64) exceeds the MNA ≤64 budget.
    let names: Vec<&str> = std::iter::once("gnd")
        .chain((0..64).map(|i| match i {
            0 => "vdd",
            _ => Box::leak(format!("n{i}").into_boxed_str()) as &str,
        }))
        .collect();
    let mut nl = crate::Netlist::new(&names);
    nl.add_v_source("vdd", "gnd", 1.8, "VDD");
    let err = solve_dc(&nl, &SolveOptions::default(), &admission()).unwrap_err();
    let SolveError::Refused(d) = &err;
    assert!(!d.admitted);
    assert!(d.reason.contains("65 nodes > 64"), "{}", d.reason);
    // 33 voltage sources exceeds the ≤32 budget.
    let mut nl2 = crate::Netlist::new(&["gnd", "a", "b"]);
    for i in 0..33 {
        nl2.add_v_source("a", "b", 1.0, &format!("V{i}"));
    }
    let err = solve_dc(&nl2, &SolveOptions::default(), &admission()).unwrap_err();
    let SolveError::Refused(d) = &err;
    assert!(d.reason.contains("33 sources > 32"), "{}", d.reason);
    // A 64-node netlist passes admission (result carries, no Refused error).
    let names64: Vec<&str> = std::iter::once("gnd")
        .chain((0..63).map(|i| match i {
            0 => "vdd",
            _ => Box::leak(format!("m{i}").into_boxed_str()) as &str,
        }))
        .collect();
    let mut nl3 = crate::Netlist::new(&names64);
    nl3.add_v_source("vdd", "gnd", 1.8, "VDD");
    if let Err(SolveError::Refused(d)) = solve_dc(&nl3, &SolveOptions::default(), &admission()) {
        panic!("64 nodes must be admitted: {}", d.reason);
    }
}

#[test]
fn source_stepping_and_gmin_escalation_paths() {
    // Default path: source stepping converges without escalation.
    let s = solve_dc(&divider(), &SolveOptions::default(), &admission()).unwrap();
    assert!(s.converged && !s.escalated);
    // Forced escalation: vin=0.9 inverter with 2-step Newton (no stepping) —
    // the plain loop cannot converge in budget, so the gmin×1e3 fallback runs.
    let mut nl = inverter();
    nl.add_v_source("vin", "gnd", 0.9, "VIN");
    let opts = SolveOptions {
        max_newton_steps: 2,
        source_stepping: true,
        ..SolveOptions::default()
    };
    let s2 = solve_dc(&nl, &opts, &admission()).unwrap();
    assert!(
        s2.escalated,
        "fallback must engage when plain Newton cannot converge"
    );
    // The escalated rescue path solves the gmin-augmented system to the full
    // contract tolerance: the PHYSICAL KCL residual (element currents only)
    // satisfies < 1e-9.
    assert!(s2.converged, "escalated rescue must converge");
    assert!(s2.residual_norm < 1e-9, "residual {}", s2.residual_norm);
    assert!(
        crate::kcl_max_residual(&nl, &s2) < 1e-9,
        "escalated KCL residual"
    );
}

#[test]
fn no_source_stepping_still_converges_on_linear_circuits() {
    let opts = SolveOptions {
        source_stepping: false,
        ..SolveOptions::default()
    };
    let s = solve_dc(&divider(), &opts, &admission()).unwrap();
    assert!(s.converged);
    assert!((s.node_voltages[1] - 1.2).abs() < 1e-9);
    assert!(!s.escalated);
}
