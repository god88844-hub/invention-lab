//! MNA DC solver — numeric-Jacobian Newton–Raphson with damped stepping,
//! source stepping, and gmin escalation. Faithful port of
//! `packages/physics-engine/src/mna.mjs` → `solveDC` (the ±1e-9 parity oracle).
//!
//! WP-2.2: MOSFET conduction dispatches on the per-device compact model —
//! `ShichmanHodges` (the unchanged WP-2.1 parity path) or the ngspice-
//! calibrated BSIM level of `crate::bsim`. Uncharacterized BSIM profiles are
//! refused fail-closed at admission time before any Newton iteration.

use crate::bsim::{self, BsimOp, MosfetModel, SoaTelemetry};
use crate::mosfet::operating_point;
use crate::netlist::{ElementKind, Netlist};
use ep_core::admission::{AdmissionController, AdmissionDecision};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub max_newton_steps: usize,
    pub max_step: f64,
    pub tolerance: f64,
    pub source_stepping: bool,
    pub gmin: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_newton_steps: 60,
            max_step: 1.0,
            tolerance: 1e-9,
            source_stepping: true,
            gmin: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub converged: bool,
    pub iterations: usize,
    pub escalated: bool,
    /// Node voltages including ground (index 0 = 0.0).
    pub node_voltages: Vec<f64>,
    pub branch_currents: Vec<f64>,
    pub device_currents: BTreeMap<String, f64>,
    pub device_power: BTreeMap<String, f64>,
    /// ‖F‖∞ of the converged residual (gmin terms included).
    pub residual_norm: f64,
    /// Per-device BSIM telemetry at the converged operating point (gm/gds,
    /// charge-conserving Cgs/Cgd, region classification, SOA status). Empty
    /// for pure square-law netlists.
    pub device_bsim: BTreeMap<String, BsimDeviceTelemetry>,
}

/// BSIM-level telemetry for one device (Phase 4 DVFS consumes Cgs/Cgd).
#[derive(Debug, Clone)]
pub struct BsimDeviceTelemetry {
    pub op: BsimOp,
    pub soa: Option<SoaTelemetry>,
}

#[derive(Debug, Clone)]
pub enum SolveError {
    /// Admission refused the workload (fail-closed, honest reason inside).
    Refused(AdmissionDecision),
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::Refused(d) => write!(f, "admission refused: {}", d.reason),
        }
    }
}

/// Gaussian elimination with partial row pivoting. Row swaps do not reorder
/// unknowns, so back-substitution addresses them directly (ESM parity).
fn solve_linear_system(a: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut u: Vec<Vec<f64>> = a.iter().map(Clone::clone).collect();
    let mut y = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if u[r][col].abs() > u[piv][col].abs() {
                piv = r;
            }
        }
        if u[piv][col].abs() < 1e-300 {
            continue; // singular row — leave (gmin prevents this)
        }
        if piv != col {
            u.swap(piv, col);
            y.swap(piv, col);
        }
        let pivot_row = u[col].clone();
        let d = pivot_row[col];
        for r in col + 1..n {
            let fct = u[r][col] / d;
            if fct == 0.0 {
                continue;
            }
            for (cc, u_rc) in u[r].iter_mut().enumerate().skip(col) {
                *u_rc -= fct * pivot_row[cc];
            }
            y[r] -= fct * y[col];
        }
    }
    let mut sol = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = y[r];
        for cc in r + 1..n {
            s -= u[r][cc] * sol[cc];
        }
        let d = u[r][r];
        sol[r] = if d.abs() < 1e-300 { 0.0 } else { s / d };
    }
    sol
}

#[derive(Clone, Copy)]
struct ResidualContext<'a> {
    netlist: &'a Netlist,
    src_scale: f64,
    gmin: f64,
}

/// Unified MOSFET conduction op `(i, in_is_drain)` for the residual stamps:
/// the Shichman–Hodges parity path, or the BSIM level on magnitude
/// coordinates. Uncharacterized BSIM profiles never reach here — `solve_dc`
/// refuses them at admission.
fn mosfet_conduction(q: &crate::mosfet::MosfetParams, vg: f64, vd: f64, vs: f64) -> (f64, bool) {
    match q.model {
        MosfetModel::ShichmanHodges => {
            let op = operating_point(q, vg, vd, vs);
            (op.i, op.in_is_drain)
        }
        MosfetModel::Bsim(id) => {
            let p = bsim::profile(id);
            let op = bsim::operating_point(p, vg, vd, vs, q.w_um, q.l_um)
                .expect("bsim profile gated by the solve admission pre-check");
            (op.i, op.in_is_drain)
        }
    }
}

fn residual(ctx: &ResidualContext, x: &[f64], out: &mut [f64]) {
    let netlist = ctx.netlist;
    out.fill(0.0);
    let n_nodes = netlist.n_nodes();
    let v = |idx: usize| -> f64 {
        if idx == 0 {
            0.0
        } else {
            x[idx - 1]
        }
    };
    // Branch-current lookup: V sources in stamp order → x[n_nodes-1+k].
    let mut branch_index = 0usize;
    for e in &netlist.elements {
        match e.kind {
            ElementKind::Resistor => {
                let i = (v(e.nodes[0]) - v(e.nodes[1])) / e.r;
                if e.nodes[0] != 0 {
                    out[e.nodes[0] - 1] += i;
                }
                if e.nodes[1] != 0 {
                    out[e.nodes[1] - 1] -= i;
                }
            }
            ElementKind::VSource => {
                let i = x[n_nodes - 1 + branch_index];
                if e.nodes[0] != 0 {
                    out[e.nodes[0] - 1] += i;
                }
                if e.nodes[1] != 0 {
                    out[e.nodes[1] - 1] -= i;
                }
                out[n_nodes - 1 + branch_index] =
                    v(e.nodes[0]) - v(e.nodes[1]) - e.v * ctx.src_scale;
                branch_index += 1;
            }
            ElementKind::ISource => {
                let i = e.i * ctx.src_scale;
                if e.nodes[0] != 0 {
                    out[e.nodes[0] - 1] -= i;
                }
                if e.nodes[1] != 0 {
                    out[e.nodes[1] - 1] += i;
                }
            }
            ElementKind::Mosfet => {
                let q = e.mosfet.as_ref().expect("mosfet params");
                let vg = v(e.nodes[0]);
                let vd = v(e.nodes[1]);
                let vs = v(e.nodes[2]);
                if e.gate_shorted {
                    // Permanent oxide-breakdown fault: gate tied to source through R.
                    let i = (vg - vs) / e.gate_short_r;
                    if e.nodes[0] != 0 {
                        out[e.nodes[0] - 1] += i;
                    }
                    if e.nodes[2] != 0 {
                        out[e.nodes[2] - 1] -= i;
                    }
                }
                let (i, in_is_drain) = mosfet_conduction(q, vg, vd, vs);
                let in_idx = if in_is_drain { e.nodes[1] } else { e.nodes[2] };
                let out_idx = if in_is_drain { e.nodes[2] } else { e.nodes[1] };
                if in_idx != 0 {
                    out[in_idx - 1] += i;
                }
                if out_idx != 0 {
                    out[out_idx - 1] -= i;
                }
            }
        }
    }
    // gmin leakage to ground folded into the residual (Jacobian stays consistent).
    for (x_i, out_i) in x
        .iter()
        .take(n_nodes - 1)
        .zip(out.iter_mut().take(n_nodes - 1))
    {
        *out_i += ctx.gmin * x_i;
    }
}

/// One Newton loop at a given source scale and gmin. Returns converged?
fn newton_loop(
    netlist: &Netlist,
    src_scale: f64,
    gmin: f64,
    x: &mut [f64],
    opts: &SolveOptions,
    iterations: &mut usize,
) -> bool {
    let n = netlist.n_nodes() - 1 + netlist.v_source_count();
    let ctx = ResidualContext {
        netlist,
        src_scale,
        gmin,
    };
    let mut f = vec![0.0; n];
    let mut f2 = vec![0.0; n];
    let mut j_mat = vec![vec![0.0; n]; n];

    let norm_of =
        |buf: &[f64]| -> f64 { buf.iter().fold(0.0_f64, |m: f64, v: &f64| m.max(v.abs())) };

    for _ in 0..opts.max_newton_steps {
        *iterations += 1;
        ctx_res(&ctx, x, &mut f);
        let norm = norm_of(&f);
        if norm < opts.tolerance {
            return true;
        }
        // Numeric Jacobian (forward differences), consistent with the residual.
        for jj in 0..n {
            let h = (1e-7_f64).max(x[jj].abs() * 1e-7);
            let orig = x[jj];
            x[jj] = orig + h;
            ctx_res(&ctx, x, &mut f2);
            x[jj] = orig;
            for ii in 0..n {
                j_mat[ii][jj] = (f2[ii] - f[ii]) / h;
            }
        }
        // Damped Newton step with adaptive limiter.
        let dx = solve_linear_system(&j_mat, &f, n);
        let mut worst: f64 = 0.0;
        for d in dx.iter().take(netlist.n_nodes() - 1) {
            worst = worst.max(d.abs());
        }
        let t = if worst > opts.max_step {
            opts.max_step / worst
        } else {
            1.0
        };
        for ii in 0..n {
            x[ii] -= t * dx[ii];
        }
        if t < 0.02 && worst > 1e3 {
            return false;
        }
    }
    // Final residual check after exhausting steps.
    ctx_res(&ctx, x, &mut f);
    norm_of(&f) < opts.tolerance
}

fn ctx_res(ctx: &ResidualContext, x: &[f64], out: &mut [f64]) {
    residual(ctx, x, out);
}

/// Solve the DC operating point. Admission is enforced before any allocation.
pub fn solve_dc(
    netlist: &Netlist,
    opts: &SolveOptions,
    admission: &AdmissionController,
) -> Result<SolveResult, SolveError> {
    let decision = admission.admit(&ep_core::admission::AdmissionRequest::Mna {
        nodes: netlist.n_nodes(),
        voltage_sources: netlist.v_source_count(),
    });
    if !decision.admitted {
        return Err(SolveError::Refused(decision));
    }

    // WP-2.2 fail-closed model gate: every BSIM device must have a
    // ngspice-characterized table. Refusing is honest — never silently
    // degrade to a fabricated substitute (Level-1 INSUFFICIENT_MODEL_DATA).
    for e in &netlist.elements {
        if e.kind == ElementKind::Mosfet {
            let q = e.mosfet.as_ref().expect("mosfet params");
            if let MosfetModel::Bsim(id) = q.model {
                if !bsim::is_characterized(id) {
                    let p = bsim::profile(id);
                    return Err(SolveError::Refused(AdmissionDecision {
                        admitted: false,
                        reason: format!(
                            "BSIM device '{}' ({}): no ngspice-characterized table — refusing fail-closed. {}",
                            p.device, p.pdk, bsim::ASAP7_NOT_CHARACTERIZED
                        ),
                    }));
                }
            }
        }
    }

    let n_nodes = netlist.n_nodes();
    let m = netlist.v_source_count();
    let n = n_nodes - 1 + m;

    let mut iterations = 0usize;
    let mut escalated = false;

    // Initial guess: all node voltages at the first source value (MOSFET-friendly start).
    let mut x = vec![0.0; n];
    let v0 = netlist
        .elements
        .iter()
        .find(|e| e.is_v_source())
        .map(|e| e.v)
        .unwrap_or(0.0);
    for x_i in x.iter_mut().take(n_nodes - 1) {
        *x_i = v0;
    }

    let scales: Vec<f64> = if opts.source_stepping {
        vec![0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    } else {
        vec![1.0]
    };

    // ESM parity: iterate ALL source-stepping scales (the ramp conditions the
    // operating point); only the last scale's convergence counts.
    let mut converged = false;
    let last_scale_index = scales.len() - 1;
    for (si, &scale) in scales.iter().enumerate() {
        converged = newton_loop(netlist, scale, opts.gmin, &mut x, opts, &mut iterations);
        if !converged && si == last_scale_index {
            // Final fallback: gmin escalation (consistent residual, ×1e3).
            // Rescue path: give it a full Newton budget so the escalated
            // system is polished to the contract tolerance (deliberate
            // improvement over the ESM parity, which reused the cut budget).
            escalated = true;
            let rescue_opts = SolveOptions {
                max_newton_steps: opts.max_newton_steps.max(60),
                ..opts.clone()
            };
            converged = newton_loop(
                netlist,
                scale,
                opts.gmin * 1e3,
                &mut x,
                &rescue_opts,
                &mut iterations,
            );
            if converged {
                // Continuation step (SPICE gmin-stepping practice): the
                // escalated gmin shifts the solution slightly — warm-start a
                // re-solve at the TRUE gmin to recover the physical solution.
                converged = newton_loop(
                    netlist,
                    scale,
                    opts.gmin,
                    &mut x,
                    &rescue_opts,
                    &mut iterations,
                );
            }
        }
    }

    // Extract solution + device operating point.
    let mut node_voltages = vec![0.0];
    node_voltages.extend_from_slice(&x[..n_nodes - 1]);
    let branch_currents = x[n_nodes - 1..n_nodes - 1 + m].to_vec();

    let v = |idx: usize| -> f64 { node_voltages[idx] };
    let mut device_currents = BTreeMap::new();
    let mut device_power = BTreeMap::new();
    let mut device_bsim = BTreeMap::new();
    for e in &netlist.elements {
        match e.kind {
            ElementKind::Resistor => {
                let i = (v(e.nodes[0]) - v(e.nodes[1])) / e.r;
                device_currents.insert(e.label.clone(), i);
                device_power.insert(e.label.clone(), i * (v(e.nodes[0]) - v(e.nodes[1])));
            }
            ElementKind::VSource => {
                let k = netlist
                    .elements
                    .iter()
                    .filter(|el| el.is_v_source())
                    .take_while(|el| !std::ptr::eq(*el, e))
                    .count();
                let i_branch = branch_currents[k];
                device_currents.insert(e.label.clone(), i_branch);
                device_power.insert(e.label.clone(), -i_branch * (v(e.nodes[0]) - v(e.nodes[1])));
            }
            ElementKind::ISource => {
                device_currents.insert(e.label.clone(), e.i);
            }
            ElementKind::Mosfet => {
                let q = e.mosfet.as_ref().unwrap();
                let vg = v(e.nodes[0]);
                let vd = v(e.nodes[1]);
                let vs = v(e.nodes[2]);
                let (i, in_is_drain) = mosfet_conduction(q, vg, vd, vs);
                let v_in = if in_is_drain { vd } else { vs };
                let v_out = if in_is_drain { vs } else { vd };
                device_currents.insert(e.label.clone(), i);
                device_power.insert(e.label.clone(), i * (v_in - v_out));
                device_power.insert(
                    format!("{}.vgs", e.label),
                    if q.mosfet_type == crate::mosfet::MosfetType::N {
                        vg - v_out
                    } else {
                        v_in - vg
                    },
                );
                device_power.insert(format!("{}.vds", e.label), v_in - v_out);
                if let MosfetModel::Bsim(id) = q.model {
                    let p = bsim::profile(id);
                    if let Ok(op) = bsim::operating_point(p, vg, vd, vs, q.w_um, q.l_um) {
                        let vgs_mag = if p.mosfet_type == crate::mosfet::MosfetType::N {
                            (vg - v_out).abs()
                        } else {
                            (v_in - vg).abs()
                        };
                        let soa = bsim::soa_check(p, vgs_mag);
                        device_bsim.insert(e.label.clone(), BsimDeviceTelemetry { op, soa });
                    }
                }
            }
        }
    }

    // Residual norm at the solution (with gmin folded in, matching the loop).
    let ctx = ResidualContext {
        netlist,
        src_scale: 1.0,
        gmin: opts.gmin,
    };
    let mut f = vec![0.0; n];
    ctx_res(&ctx, &x, &mut f);
    let residual_norm = f.iter().fold(0.0_f64, |m: f64, v: &f64| m.max(v.abs()));

    Ok(SolveResult {
        converged,
        iterations,
        escalated,
        node_voltages,
        branch_currents,
        device_currents,
        device_power,
        residual_norm,
        device_bsim,
    })
}

/// Convenience: default options + default budgets.
pub fn solve_dc_default(netlist: &Netlist) -> Result<SolveResult, SolveError> {
    solve_dc(
        netlist,
        &SolveOptions::default(),
        &ep_core::admission::AdmissionController::new(),
    )
}

/// Max |KCL residual| over all non-ground nodes: the sum of element currents
/// leaving each node must be ~0 at the solution. Power-of-honesty oracle.
pub fn kcl_max_residual(netlist: &Netlist, result: &SolveResult) -> f64 {
    let v = |idx: usize| -> f64 { result.node_voltages[idx] };
    let node_count = netlist.n_nodes() - 1;
    let mut residuals = vec![0.0; node_count];
    for e in &netlist.elements {
        match e.kind {
            ElementKind::Resistor => {
                let i = (v(e.nodes[0]) - v(e.nodes[1])) / e.r;
                if e.nodes[0] != 0 {
                    residuals[e.nodes[0] - 1] += i;
                }
                if e.nodes[1] != 0 {
                    residuals[e.nodes[1] - 1] -= i;
                }
            }
            ElementKind::VSource => {
                let i = result
                    .device_currents
                    .get(&e.label)
                    .copied()
                    .unwrap_or_else(|| result.branch_currents.first().copied().unwrap_or(0.0));
                if e.nodes[0] != 0 {
                    residuals[e.nodes[0] - 1] += i;
                }
                if e.nodes[1] != 0 {
                    residuals[e.nodes[1] - 1] -= i;
                }
            }
            ElementKind::ISource => {
                if e.nodes[0] != 0 {
                    residuals[e.nodes[0] - 1] -= e.i;
                }
                if e.nodes[1] != 0 {
                    residuals[e.nodes[1] - 1] += e.i;
                }
            }
            ElementKind::Mosfet => {
                let q = e.mosfet.as_ref().unwrap();
                let (i, in_is_drain) =
                    mosfet_conduction(q, v(e.nodes[0]), v(e.nodes[1]), v(e.nodes[2]));
                let in_idx = if in_is_drain { e.nodes[1] } else { e.nodes[2] };
                let out_idx = if in_is_drain { e.nodes[2] } else { e.nodes[1] };
                if in_idx != 0 {
                    residuals[in_idx - 1] += i;
                }
                if out_idx != 0 {
                    residuals[out_idx - 1] -= i;
                }
            }
        }
    }
    residuals.iter().fold(0.0_f64, |m, r| m.max(r.abs()))
}
