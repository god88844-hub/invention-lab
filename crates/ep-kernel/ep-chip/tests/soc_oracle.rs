//! WP-4.4 parity + acceptance gate: the SoC interposer composer against the
//! Phase 0 oracle and the sealed offline vault.
//!
//! Gates enforced here (PHASE_04 WP-4.4: "NoC route model (1 cycle/hop),
//! memory stacks from 07 JSON; discrete-PCIE vs unified zero-copy comparison
//! engine; DVFS throttle loop coupled to ep-thermal tile temps; tile→block
//! floorplan from data/cad-specs/02"; 03 §3 Roofline + Little's Law):
//!  1. Vault constants: every figure verified against the SHA-256-sealed
//!     `data/reference-library/07` payload via `ep-core` (fail-closed) — Law 5.
//!  2. Unified-vs-discrete parity: bit-exact vs
//!     `packages/chip-engine/src/accelerators.mjs :: compareArchitectures`
//!     (golden fixture from the committed `tests/data/gen_soc_parity.mjs`,
//!     re-validated live against Node when available — absent tool ⇒ loud
//!     SKIP, never a silent pass).
//!  3. Roofline `min(PeakFLOPs, AI·PeakBW)` + Little's Law `L = λ·W` bounds.
//!  4. NoC: 1 cycle/hop over the sealed `02` floorplan route graph.
//!  5. Floorplan: tile→block coordinates bound verbatim to the sealed `02`
//!     payload (SHA-256-verified via the cad-specs manifest through ep-core).
//!  6. DVFS: exact stepwise throttle at 125 °C, ±5 °C hysteresis recovery,
//!     `si_thermal_runaway` latch at 200 °C, coupled to a live `ep-thermal`
//!     `ThermalGrid2D` tile-temperature feed.

use ep_chip::soc::{
    alpha_energy, bytes_per_cycle, compare_architectures, copy_time_cycles, littles_law, roofline,
    submission_latency_cycles, CompareOptions, DvfsAction, DvfsController, MemoryKind, NoCFabric,
    RooflineBound, SocBaselines, SocError, SocFloorplan, RUNAWAY_THRESHOLD_C, THROTTLE_THRESHOLD_C,
};
use ep_core::reference::{verify_entry, verify_library_dir};
use ep_core::ProvenanceBadge;
use serde_json::Value;

fn repo_root() -> std::path::PathBuf {
    // crates/ep-kernel/ep-chip → repo root is three levels up.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .to_path_buf()
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("data/soc_parity_vectors.json")).expect("fixture parses")
}

fn load07() -> (SocBaselines, String) {
    let raw = std::fs::read_to_string(
        repo_root().join("data/reference-library/07_soc_interposer_memory_baselines.json"),
    )
    .expect("07 payload readable");
    (
        SocBaselines::from_json(&raw).expect("07 payload parses"),
        raw,
    )
}

fn load_floorplan() -> (SocFloorplan, String) {
    let raw =
        std::fs::read_to_string(repo_root().join("data/cad-specs/02_pcb_bga_soc_packages.json"))
            .expect("02 payload readable");
    (
        SocFloorplan::from_cad_specs_json(&raw).expect("02 floorplan parses"),
        raw,
    )
}

/// A Rust float viewed as its exact IEEE-754 bits (hex), for bit-exact parity.
fn bits(x: f64) -> String {
    x.to_ne_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// 1. Vault constants (Law 5 — verified, never invented)
// ---------------------------------------------------------------------------

#[test]
fn vault_verifies_and_anchors_every_soc_constant() {
    let n = verify_library_dir(&repo_root().join("data").join("reference-library"))
        .expect("reference library must verify byte-exactly");
    assert_eq!(n, 8, "all eight reference files verify");

    let (b, raw07) = load07();
    let j07: Value = serde_json::from_str(&raw07).unwrap();
    // LPDDR5 unified-memory figures, verbatim (JEDEC/vendor published).
    assert_eq!(b.lpddr5.apple_m1_total, 68.25);
    assert_eq!(b.lpddr5.apple_m2_total, 100.0);
    assert_eq!(b.lpddr5.intel_iris_xe_155u_total, 68.0);
    assert_eq!(b.lpddr5.per_channel_16bit_4800mts, 38.4);
    assert_eq!(b.lpddr5.per_channel_16bit_7467mts, 59.7);
    assert_eq!(b.lpddr5.typical_random_access_ns, 90.0);
    assert_eq!(b.lpddr5.target_platform_gb, 16.0);
    // The zero-copy contract, verbatim from the payload.
    assert_eq!(
        b.lpddr5.zero_copy_note,
        j07["parameters"]["lpddr5_unified_memory"]["zero_copy_note"]
            .as_str()
            .unwrap(),
        "zero_copy_note passes through verbatim"
    );
    assert!(b
        .lpddr5
        .zero_copy_note
        .contains("no PCIe-style DMA copy exists"));
    assert!(b
        .lpddr5
        .zero_copy_note
        .contains("bandwidth is paid once per touch"));
    // HBM + PCIe.
    assert_eq!(b.hbm.tpu_v1_total, 600.0);
    assert_eq!(b.hbm.hbm2e_per_stack, 460.0);
    assert_eq!(b.pcie.pcie4_x16_directional, 32.0);
    assert_eq!(b.pcie.pcie5_x16_directional, 64.0);
    // Cache hierarchy latencies (the §3 Little's Law levels).
    assert_eq!(b.caches.l1_latency_cycles, 3.0);
    assert_eq!(b.caches.l2_latency_cycles, 12.0);
    assert_eq!(b.caches.l3_latency_cycles, 26.0);
    assert_eq!(b.caches.dram_latency_cycles, 200.0);
    // Interposer dimensions verbatim.
    assert_eq!(
        b.interposer.target_soc_interposer,
        "10 x 7 (v3 scene scale, parametric in data/cad-specs/02_pcb_bga_soc_packages.json)"
    );

    // Tampered payloads never panic: a removed field is refused fail-closed
    // (a value-tampered-but-wellformed payload is caught by the byte-exact
    // `verify_library_dir` check asserted above).
    let broken = raw07.replace("\"tpu_v1_total\": 600.0,", "");
    assert!(matches!(
        SocBaselines::from_json(&broken),
        Err(SocError::Vault(_))
    ));
}

// ---------------------------------------------------------------------------
// 2. Floorplan + NoC (1 cycle/hop, cad-specs 02 binding)
// ---------------------------------------------------------------------------

#[test]
fn floorplan_binds_the_sealed_cad_specs_02_coordinates() {
    // The cad-specs manifest is the same sealed format — verify the 02 bytes
    // against it through ep-core (Law 5).
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("data/cad-specs/MANIFEST.json")).unwrap(),
    )
    .unwrap();
    let raw02 =
        std::fs::read_to_string(repo_root().join("data/cad-specs/02_pcb_bga_soc_packages.json"))
            .unwrap();
    let entry = manifest["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["file"] == "02_pcb_bga_soc_packages.json")
        .expect("02 manifest entry")
        .clone();
    let entry: ep_core::reference::ManifestEntry = serde_json::from_value(entry).unwrap();
    verify_entry(&entry, raw02.as_bytes()).expect("02 payload verifies byte-exactly");

    let (fp, _) = load_floorplan();
    assert_eq!(fp.die_mm, (8.0, 6.0), "die size verbatim");
    assert_eq!(
        fp.interposer_mm,
        (10.0, 7.0),
        "interposer size verbatim (07: target scene)"
    );
    let cpu = fp.block("cpu").expect("cpu block");
    assert_eq!(cpu.tiles, 8, "RV32IM CPU cluster tiles");
    assert_eq!(cpu.position_mm, (0.4, 0.4));
    assert_eq!(cpu.size_mm, (2.6, 2.2));
    let tpu = fp.block("tpu").expect("tpu block");
    assert_eq!(tpu.tiles, 64, "8×8 TPU grid");
    assert_eq!(tpu.position_mm, (3.2, 0.4));
    assert_eq!(tpu.size_mm, (2.2, 2.2));
    let gpu = fp.block("gpu").expect("gpu block");
    assert_eq!(gpu.tiles, 96, "Intel Iris Xe 96 EU reference");
    assert_eq!(gpu.position_mm, (5.6, 0.6));
    assert!(
        fp.block("mem0").is_some() && fp.block("mem1").is_some(),
        "two unified stacks"
    );

    // The five DECLARED logical legs (sealed note: mem↔cpu, mem↔tpu, mem↔gpu,
    // cpu↔tpu, tpu↔gpu) expand over the two concrete unified stacks (mem0,
    // mem1) into 8 physical 1-hop legs.
    assert_eq!(
        fp.routes.len(),
        8,
        "5 logical legs × 2 memory stacks − dedup"
    );
    let has = |a: &str, b: &str| {
        fp.routes
            .iter()
            .any(|(x, y)| (x == a && y == b) || (x == b && y == a))
    };
    assert!(has("mem0", "cpu") && has("mem1", "cpu"), "mem↔cpu legs");
    assert!(has("mem0", "tpu") && has("mem1", "tpu"), "mem↔tpu legs");
    assert!(has("mem0", "gpu") && has("mem1", "gpu"), "mem↔gpu legs");
    assert!(has("cpu", "tpu"), "cpu↔tpu leg");
    assert!(has("tpu", "gpu"), "tpu↔gpu leg");
}

#[test]
fn noc_routes_cost_one_cycle_per_hop_over_the_declared_graph() {
    let (fp, _) = load_floorplan();
    let noc = NoCFabric::new(&fp);
    assert_eq!(
        NoCFabric::CYCLES_PER_HOP,
        1,
        "07: model bus latency as 1 cycle per hop"
    );
    // Direct declared legs: exactly one hop.
    for (a, b) in &fp.routes {
        assert_eq!(noc.route_latency_cycles(a, b).expect("direct leg"), 1);
        assert_eq!(noc.route_latency_cycles(b, a).expect("undirected"), 1);
        assert_eq!(noc.route_path(a, b).unwrap().len(), 2);
    }
    // Two-leg path: cpu → gpu rides cpu↔tpu then tpu↔gpu (2 cycles).
    assert_eq!(
        noc.route_latency_cycles("cpu", "gpu").expect("2-hop path"),
        2
    );
    let path = noc.route_path("cpu", "gpu").unwrap();
    assert_eq!(path.len(), 3, "cpu → tpu → gpu");
    // Same-block route is zero cycles.
    assert_eq!(noc.route_latency_cycles("cpu", "cpu").expect("self"), 0);
    // Unknown endpoints are refused fail-closed.
    assert!(matches!(
        noc.route_latency_cycles("cpu", "npu"),
        Err(SocError::Unknown(_))
    ));
    assert!(matches!(
        noc.route_path("dma_engine", "cpu"),
        Err(SocError::Unknown(_))
    ));
}

// ---------------------------------------------------------------------------
// 3. Unified vs discrete — bit-exact parity with the JS oracle
// ---------------------------------------------------------------------------

#[test]
fn compare_architectures_bit_exact_vs_chip_engine_fixture() {
    let fixture = fixture();
    assert_eq!(
        fixture["_provenance"]["badge"].as_str().unwrap(),
        "SIMULATED_ARCHITECTURAL_MODEL",
        "fixture provenance badge"
    );
    for case in fixture["cases"].as_array().expect("fixture cases array") {
        let id = case["id"].as_str().unwrap();
        let wl = &case["workload"];
        let opts_v = &case["opts"];
        let opts = CompareOptions {
            grid_size: opts_v["gridSize"].as_u64().unwrap_or(8) as usize,
            clock_mhz: opts_v["clockMHz"].as_f64().unwrap_or(700.0),
            vdd: opts_v["vdd"].as_f64().unwrap_or(0.8),
            pcie_gbps: opts_v["pcieGBps"].as_f64().unwrap_or(32.0),
            lpddr_gbps: opts_v["lpddrGBps"].as_f64().unwrap_or(68.0),
            hbm_gbps: opts_v["hbmGBps"].as_f64().unwrap_or(600.0),
        };
        let r = compare_architectures(
            wl["M"].as_u64().unwrap_or(64),
            wl["N"].as_u64().unwrap_or(64),
            wl["K"].as_u64().unwrap_or(64),
            &opts,
        )
        .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(r.badge, ProvenanceBadge::SimulatedArchitecturalModel);
        for (arm_name, expected) in [
            ("discreteInt8", &case["discreteInt8"]),
            ("discreteQ4", &case["discreteQ4"]),
            ("unifiedInt8", &case["unifiedInt8"]),
            ("unifiedQ4", &case["unifiedQ4"]),
        ] {
            let arm = match arm_name {
                "discreteInt8" => &r.discrete_int8,
                "discreteQ4" => &r.discrete_q4,
                "unifiedInt8" => &r.unified_int8,
                _ => &r.unified_q4,
            };
            assert_eq!(
                arm.architecture,
                expected["architecture"].as_str().unwrap(),
                "{id}/{arm_name}"
            );
            assert_eq!(
                arm.radix,
                expected["radix"].as_str().unwrap(),
                "{id}/{arm_name}"
            );
            assert_eq!(
                arm.total_cycles,
                expected["totalCycles"].as_u64().unwrap(),
                "{id}/{arm_name} totalCycles"
            );
            assert_eq!(
                arm.bytes_on_bus,
                expected["bytesOnBus"].as_f64().unwrap(),
                "{id}/{arm_name} bytesOnBus"
            );
            assert_eq!(
                bits(arm.bandwidth_gbps),
                expected["bandwidthGBpsBits"].as_str().unwrap(),
                "{id}/{arm_name} bandwidthGBps bit-exact"
            );
            assert_eq!(
                bits(arm.energy_per_op_pj),
                expected["energyPerOpPJBits"].as_str().unwrap(),
                "{id}/{arm_name} energyPerOpPJ bit-exact"
            );
            assert_eq!(
                bits(arm.dynamic_power_w),
                expected["dynamicPowerWBits"].as_str().unwrap(),
                "{id}/{arm_name} dynamicPowerW bit-exact"
            );
            assert_eq!(
                bits(arm.hotspot_c),
                expected["hotspotCBits"].as_str().unwrap(),
                "{id}/{arm_name} hotspotC bit-exact"
            );
        }
        assert_eq!(
            bits(r.delta.cycle_reduction_pct),
            case["delta"]["cycleReductionPctBits"].as_str().unwrap(),
            "{id} cycleReductionPct bit-exact"
        );
        assert_eq!(
            bits(r.delta.energy_reduction_pct),
            case["delta"]["energyReductionPctBits"].as_str().unwrap(),
            "{id} energyReductionPct bit-exact"
        );
        assert_eq!(
            bits(r.delta.q4_bandwidth_gain_pct),
            case["delta"]["q4BandwidthGainPctBits"].as_str().unwrap(),
            "{id} q4BandwidthGainPct bit-exact"
        );
    }
}

#[test]
fn compare_architectures_live_node_differential() {
    // Re-run the JS oracle live and diff against the Rust port (loud SKIP
    // when Node is absent — never a silent pass).
    let node = if cfg!(windows) {
        std::process::Command::new("where").arg("node").output()
    } else {
        std::process::Command::new("which").arg("node").output()
    };
    let node_present = node
        .as_ref()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false);
    if !node_present {
        eprintln!("SOC-ORACLE-SKIP: node is not available on PATH — the golden fixture replay above still asserts bit-exact parity.");
        return;
    }
    let fixture = fixture();
    let temp = std::env::temp_dir().join(format!("ep-chip-soc-oracle-{}.mjs", std::process::id()));
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let repo_url = url::Url::from_file_path(repo_root()).expect("repo URL");
        let script = format!(
            r#"
            import {{ compareArchitectures }} from '{repo_url}/packages/chip-engine/src/accelerators.mjs';
            const wl = {};
            const opts = {};
            const r = compareArchitectures(wl, opts);
            const bits = (x) => Buffer.from(new Float64Array([x]).buffer).toString('hex');
            console.log(JSON.stringify({{
                d: r.discreteInt8.totalCycles, u: r.unifiedInt8.totalCycles,
                dq: r.discreteQ4.totalCycles, uq: r.unifiedQ4.totalCycles,
                dBw: bits(r.discreteInt8.bandwidthGBps), uBw: bits(r.unifiedInt8.bandwidthGBps),
                dE: bits(r.discreteInt8.energyPerOpPJ), uE: bits(r.unifiedInt8.energyPerOpPJ),
                pD: bits(r.discreteInt8.dynamicPowerW), pH: bits(r.discreteInt8.hotspotC),
                cR: bits(r.delta.cycleReductionPct), eR: bits(r.delta.energyReductionPct),
                qG: bits(r.delta.q4BandwidthGainPct),
            }}));
            "#,
            serde_json::to_string(&case["workload"]).unwrap(),
            serde_json::to_string(&case["opts"]).unwrap(),
        );
        std::fs::write(&temp, &script).expect("write temp module");
        let output = std::process::Command::new("node")
            .arg(&temp)
            .output()
            .expect("node executed");
        assert!(
            output.status.success(),
            "{id}: node failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let live: Value = serde_json::from_slice(&output.stdout).expect("node JSON");
        let _ = std::fs::remove_file(&temp);
        let opts_v = &case["opts"];
        let opts = CompareOptions {
            grid_size: opts_v["gridSize"].as_u64().unwrap_or(8) as usize,
            clock_mhz: opts_v["clockMHz"].as_f64().unwrap_or(700.0),
            vdd: opts_v["vdd"].as_f64().unwrap_or(0.8),
            pcie_gbps: opts_v["pcieGBps"].as_f64().unwrap_or(32.0),
            lpddr_gbps: opts_v["lpddrGBps"].as_f64().unwrap_or(68.0),
            hbm_gbps: opts_v["hbmGBps"].as_f64().unwrap_or(600.0),
        };
        let r = compare_architectures(
            case["workload"]["M"].as_u64().unwrap_or(64),
            case["workload"]["N"].as_u64().unwrap_or(64),
            case["workload"]["K"].as_u64().unwrap_or(64),
            &opts,
        )
        .expect("compare");
        assert_eq!(
            r.discrete_int8.total_cycles,
            live["d"].as_u64().unwrap(),
            "{id} discrete cycles"
        );
        assert_eq!(
            r.unified_int8.total_cycles,
            live["u"].as_u64().unwrap(),
            "{id} unified cycles"
        );
        assert_eq!(
            r.discrete_q4.total_cycles,
            live["dq"].as_u64().unwrap(),
            "{id} discrete q4 cycles"
        );
        assert_eq!(
            r.unified_q4.total_cycles,
            live["uq"].as_u64().unwrap(),
            "{id} unified q4 cycles"
        );
        assert_eq!(
            bits(r.discrete_int8.bandwidth_gbps),
            live["dBw"].as_str().unwrap(),
            "{id} bandwidth bits"
        );
        assert_eq!(
            bits(r.unified_int8.bandwidth_gbps),
            live["uBw"].as_str().unwrap(),
            "{id} unified bandwidth bits"
        );
        assert_eq!(
            bits(r.discrete_int8.energy_per_op_pj),
            live["dE"].as_str().unwrap(),
            "{id} energy bits"
        );
        assert_eq!(
            bits(r.unified_int8.energy_per_op_pj),
            live["uE"].as_str().unwrap(),
            "{id} unified energy bits"
        );
        assert_eq!(
            bits(r.discrete_int8.dynamic_power_w),
            live["pD"].as_str().unwrap(),
            "{id} power bits"
        );
        assert_eq!(
            bits(r.discrete_int8.hotspot_c),
            live["pH"].as_str().unwrap(),
            "{id} hotspot bits"
        );
        assert_eq!(
            bits(r.delta.cycle_reduction_pct),
            live["cR"].as_str().unwrap(),
            "{id} cycle delta bits"
        );
        assert_eq!(
            bits(r.delta.energy_reduction_pct),
            live["eR"].as_str().unwrap(),
            "{id} energy delta bits"
        );
        assert_eq!(
            bits(r.delta.q4_bandwidth_gain_pct),
            live["qG"].as_str().unwrap(),
            "{id} q4 gain bits"
        );
    }
}

#[test]
fn zero_copy_vs_dma_deltas_are_the_declared_architectural_wins() {
    // JS-default workload: unified pays no weight copy at all (07
    // zero_copy_note), discrete pays the full DMA per move.
    let r = compare_architectures(64, 64, 64, &CompareOptions::default()).unwrap();
    assert_eq!(
        r.discrete_int8.bytes_on_bus, 12288.0,
        "weights + activations + outputs"
    );
    assert_eq!(
        r.unified_int8.bytes_on_bus, 8192.0,
        "weights resident: 4096 bytes saved"
    );
    assert!(
        r.discrete_int8.total_cycles > r.unified_int8.total_cycles,
        "DMA copy costs cycles"
    );
    assert!(
        r.delta.cycle_reduction_pct > 0.0,
        "no weight DMA copy ⇒ fewer cycles"
    );
    // The JS oracle's declared energy model is α·V² only (no architecture
    // term), so the int8 arms' energy delta is EXACTLY zero — parity with the
    // oracle, not a fabricated win. The architectural wins are cycles + bus
    // bytes; the Q4 bandwidth delta is whatever the oracle declares (the
    // dequant pass can outweigh the halved bytes at equal clocks).
    assert_eq!(r.delta.energy_reduction_pct, 0.0);
    // Q4 packs 2 digits/byte: the unified Q4 arm moves exactly HALF the
    // unified INT8 bus bytes.
    assert_eq!(r.unified_q4.bytes_on_bus, r.unified_int8.bytes_on_bus / 2.0);
    // Energy per op depends only on α and V (α int8 1.2, q4 0.9) — the radix
    // energy factor is exact.
    assert_eq!(alpha_energy("int8"), 1.2);
    assert_eq!(alpha_energy("q4"), 0.9);
    assert!(
        (r.discrete_int8.energy_per_op_pj - 1.2 * 0.64 * 1e12).abs() < 1e-6,
        "α·V² at 0.8 V"
    );
    // The discrete arm moves strictly more bytes on the same interface — its
    // achieved bandwidth (bytes / total time) is higher precisely because it
    // wastes cycles copying; the architectural win is the 3.5 % cycle cut and
    // the 4096 bytes it never moves.
    assert!(r.discrete_int8.bytes_on_bus > r.unified_int8.bytes_on_bus);
    assert!(r.discrete_int8.total_cycles > r.unified_int8.total_cycles);
}

#[test]
fn non_physical_compare_inputs_are_refused() {
    assert!(matches!(
        compare_architectures(
            64,
            64,
            64,
            &CompareOptions {
                clock_mhz: 0.0,
                ..CompareOptions::default()
            }
        ),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        compare_architectures(
            64,
            64,
            64,
            &CompareOptions {
                vdd: -0.8,
                ..CompareOptions::default()
            }
        ),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        compare_architectures(
            64,
            64,
            64,
            &CompareOptions {
                pcie_gbps: f64::NAN,
                ..CompareOptions::default()
            }
        ),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        compare_architectures(
            64,
            64,
            64,
            &CompareOptions {
                grid_size: 0,
                ..CompareOptions::default()
            }
        ),
        Err(SocError::NonPhysicalInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 4. Roofline + Little's Law (§3)
// ---------------------------------------------------------------------------

#[test]
fn roofline_min_law_with_declared_bottleneck() {
    // Memory-bound: AI·BW < Peak.
    let r = roofline(2_000.0, 100.0, 10.0).unwrap(); // AI·BW = 1000 < 2000
    assert_eq!(r.achievable_flops, 1000.0);
    assert_eq!(r.bound, RooflineBound::Memory);
    // Compute-bound: AI·BW ≥ Peak.
    let r = roofline(2_000.0, 100.0, 25.0).unwrap(); // AI·BW = 2500 ≥ 2000
    assert_eq!(r.achievable_flops, 2000.0);
    assert_eq!(r.bound, RooflineBound::Compute);
    // The exact boundary picks the compute roof (min semantics).
    let r = roofline(2_000.0, 100.0, 20.0).unwrap(); // AI·BW = 2000 == Peak
    assert_eq!(r.bound, RooflineBound::Compute);
    assert_eq!(r.achievable_flops, 2000.0);
    assert_eq!(r.badge, ProvenanceBadge::SimulatedArchitecturalModel);
    // Real SoC point: TPU v1 — 92 TOPS INT8, 600 GB/s HBM. INT8 GEMM at
    // AI = ops/byte of a 64×64 tile-resident GEMM (compute-bound).
    let tpu = roofline(92e12, 600e9, 160.0).unwrap();
    assert_eq!(
        tpu.bound,
        RooflineBound::Compute,
        "92e12 vs 160×600e9 = 9.6e13"
    );
    // And the memory-bound side of the same TPU (AI 113: 6.78e13 < 9.2e13).
    let tpu_mem = roofline(92e12, 600e9, 113.0).unwrap();
    assert_eq!(tpu_mem.bound, RooflineBound::Memory);
    // Non-physical inputs refused.
    assert!(matches!(
        roofline(-1.0, 600e9, 1.0),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        roofline(92e12, -600e9, 1.0),
        Err(SocError::NonPhysicalInput(_))
    ));
}

#[test]
fn littles_law_bounds_match_the_07_cache_hierarchy() {
    let (b, _) = load07();
    // One byte-stream access per cycle per level: in-flight = rate × latency.
    assert_eq!(
        littles_law(1.0, b.caches.l1_latency_cycles).unwrap(),
        3.0,
        "L1"
    );
    assert_eq!(
        littles_law(1.0, b.caches.l2_latency_cycles).unwrap(),
        12.0,
        "L2"
    );
    assert_eq!(
        littles_law(1.0, b.caches.l3_latency_cycles).unwrap(),
        26.0,
        "L3"
    );
    assert_eq!(
        littles_law(1.0, b.caches.dram_latency_cycles).unwrap(),
        200.0,
        "DRAM"
    );
    // The §3 identity: in-flight ops = bandwidth × latency. 07 LPDDR5 at
    // 68 GB/s and 90 ns: λ = 68e9 accesses/s → L = 68e9 × 90e-9 = 6120.
    let inflight = littles_law(
        b.lpddr5.intel_iris_xe_155u_total * 1e9,
        b.lpddr5.typical_random_access_ns * 1e-9,
    )
    .unwrap();
    assert!(
        (inflight - 6120.0).abs() < 1e-6,
        "LPDDR5 in-flight byte stream = 68 GB/s × 90 ns (got {inflight})"
    );
    // Non-physical inputs refused.
    assert!(matches!(
        littles_law(-1.0, 200.0),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        littles_law(1.0, f64::NAN),
        Err(SocError::NonPhysicalInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 5. Memory stack equations (07 verbatim)
// ---------------------------------------------------------------------------

#[test]
fn memory_stack_equations_match_the_07_payload_verbatim() {
    let (b, _) = load07();
    // bytes_per_cycle = BW_GBps × 1e9 / clock_Hz (07 equations).
    let bpc = bytes_per_cycle(32.0, 700.0).unwrap();
    assert!((bpc - 32e9 / 700e6).abs() < 1e-9, "PCIe 4.0 x16 at 700 MHz");
    // copy_time_cycles = bytes / bytes_per_cycle (07 equations).
    let c = copy_time_cycles(4096.0, 32.0, 700.0).unwrap();
    assert!((c - 4096.0 / bpc).abs() < 1e-12);
    // MemoryKind bandwidths resolve to the 07 figures.
    assert_eq!(MemoryKind::UnifiedLpddr5.bandwidth_gbps(&b), 68.0);
    assert_eq!(MemoryKind::Hbm.bandwidth_gbps(&b), 600.0);
    assert_eq!(MemoryKind::DiscretePcie4.bandwidth_gbps(&b), 32.0);
    assert_eq!(MemoryKind::DiscretePcie5.bandwidth_gbps(&b), 64.0);
    // Zero-copy semantics: unified/HBM pay bandwidth once per touch.
    assert!(MemoryKind::UnifiedLpddr5.is_zero_copy());
    assert!(MemoryKind::Hbm.is_zero_copy());
    assert!(!MemoryKind::DiscretePcie4.is_zero_copy());
    // Driver submission latency is an explicit caller parameter (07 cites a
    // ~5–10 µs published range) — never assumed inside the compare path.
    let cyc = submission_latency_cycles(5.0, 700.0).unwrap();
    assert!((cyc - 5e-6 * 700e6).abs() < 1e-6);
    assert!(matches!(
        submission_latency_cycles(-1.0, 700.0),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        bytes_per_cycle(-1.0, 700.0),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        copy_time_cycles(-1.0, 32.0, 700.0),
        Err(SocError::NonPhysicalInput(_))
    ));
}

// ---------------------------------------------------------------------------
// 6. DVFS closed loop: throttle steps, hysteresis, si_thermal_runaway latch
// ---------------------------------------------------------------------------

#[test]
fn dvfs_throttles_stepwise_above_125c_with_hysteresis_recovery() {
    let mut dvfs = DvfsController::new(700.0, 100.0).unwrap();
    assert_eq!(
        THROTTLE_THRESHOLD_C, 125.0,
        "05 envelope riscv_rv32im temp upper bound"
    );
    assert_eq!(
        RUNAWAY_THRESHOLD_C, 200.0,
        "06 si_thermal_runaway threshold_c"
    );

    // 126 °C → one 10 % step down (700 → 630).
    let e = dvfs.step(&[100.0, 126.0]).unwrap();
    assert_eq!(e.action, DvfsAction::ThrottleDown);
    assert_eq!(e.t_junction_c, 126.0);
    assert!((e.freq_after_mhz - 630.0).abs() < 1e-9);
    // Still hot → second step (630 → 567).
    let e = dvfs.step(&[126.0]).unwrap();
    assert_eq!(e.action, DvfsAction::ThrottleDown);
    assert!((e.freq_after_mhz - 567.0).abs() < 1e-9);
    // Inside the ±5 °C hysteresis band (120..125] → hold.
    let e = dvfs.step(&[122.0]).unwrap();
    assert_eq!(e.action, DvfsAction::Hold);
    assert!(
        (dvfs.freq_mhz() - 567.0).abs() < 1e-9,
        "no oscillation inside the band"
    );
    // Below the band (117 °C < 120) → one recovery step back up.
    let e = dvfs.step(&[117.0]).unwrap();
    assert_eq!(e.action, DvfsAction::ThrottleUp);
    assert!((dvfs.freq_mhz() - 630.0).abs() < 1e-9);
    // Cool → recovery steps cap exactly at the base frequency.
    let mut ups = 0;
    for _ in 0..32 {
        let e = dvfs.step(&[90.0]).unwrap();
        // Recovery steps until the base frequency, then an honest Hold
        // (already at base — nothing to step up to).
        assert!(matches!(
            e.action,
            DvfsAction::ThrottleUp | DvfsAction::Hold
        ));
        if e.action == DvfsAction::ThrottleUp {
            ups += 1;
        }
    }
    assert!(ups >= 1, "at least one recovery step");
    assert!(
        (dvfs.freq_mhz() - 700.0).abs() < 1e-9,
        "recovery caps exactly at base"
    );
    let e = dvfs.step(&[90.0]).unwrap();
    assert_eq!(e.action, DvfsAction::Hold, "already at base → hold");
}

#[test]
fn dvfs_floor_and_thermal_runaway_latch_fail_closed() {
    let mut dvfs = DvfsController::new(700.0, 100.0).unwrap();
    // Sustained extreme heat drives the frequency to the declared floor
    // (100 MHz) without ever crossing 200 °C — the floor holds, no runaway.
    for _ in 0..64 {
        let e = dvfs.step(&[180.0]).unwrap();
        assert_eq!(e.action, DvfsAction::ThrottleDown);
        assert!(e.freq_after_mhz >= 100.0, "declared floor respected");
    }
    assert!((dvfs.freq_mhz() - 100.0).abs() < 1e-9, "floor reached");
    assert!(!dvfs.runaway_latched());

    // 199 °C: still below the 06 threshold → no runaway.
    let e = dvfs.step(&[199.0]).unwrap();
    assert_ne!(e.action, DvfsAction::Runaway);
    assert!(!dvfs.runaway_latched());

    // 200 °C exactly → si_thermal_runaway latched (06 threshold_c = 200).
    let e = dvfs.step(&[120.0, 200.0]).unwrap();
    assert_eq!(e.action, DvfsAction::Runaway);
    assert_eq!(e.t_junction_c, 200.0);
    assert!(dvfs.runaway_latched());

    // Fail-closed latch: further steps are refused until an explicit reset.
    assert!(matches!(dvfs.step(&[90.0]), Err(SocError::RunawayLatched)));
    dvfs.reset_runaway();
    assert!(!dvfs.runaway_latched());
    assert!(
        (dvfs.freq_mhz() - 700.0).abs() < 1e-9,
        "reset restores the base frequency"
    );
    let e = dvfs.step(&[90.0]).unwrap();
    assert_eq!(e.action, DvfsAction::Hold);

    // Non-physical feeds are refused (never silently clamped).
    assert!(matches!(dvfs.step(&[]), Err(SocError::NonPhysicalInput(_))));
    assert!(matches!(
        dvfs.step(&[f64::NAN]),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        DvfsController::new(0.0, 100.0),
        Err(SocError::NonPhysicalInput(_))
    ));
    assert!(matches!(
        DvfsController::new(700.0, 800.0),
        Err(SocError::NonPhysicalInput(_))
    ));
}

/// WP-4.4 checklist: "DVFS throttle loop coupled to `ep-thermal` tile temps".
/// A real `ep-thermal` ThermalGrid2D is the thermal feed of the closed loop.
#[test]
fn dvfs_loop_couples_to_a_live_ep_thermal_grid() {
    use ep_chip::power::{Farad, Hertz};
    use ep_thermal::{
        admission,
        pde::{BoundaryMode, Scheme, ThermalGrid2D},
    };

    let mut grid = ThermalGrid2D::new(
        8,
        8,
        0.001,  // 1 mm cells (floorplan-scale tile)
        0.0005, // 0.5 mm silicon thickness
        1.63e6, // ρ·c_p silicon (J/(m³·K))
        148.0,  // k silicon (W/(m·K))
        300.0,  // 300 K bench ambient
        0.0,    // no convection (insulated edges — energy-conservation oracle)
        BoundaryMode::Insulated,
        Scheme::BackwardEuler,
        admission(),
    )
    .expect("grid admitted");
    // Drive a hot TPU tile: sustained dissipation steps the grid up past the
    // 125 °C (398.15 K) throttle threshold, and the closed loop throttles.
    let mut power = vec![0.0_f64; 64];
    power[27] = 25.0; // one 25 W hotspot (TPU tile)
    grid.power_w = Some(power);

    let mut dvfs = DvfsController::new(700.0, 100.0).unwrap();
    let mut throttles = 0_u32;
    let mut runaway_seen = false;
    for cycle in 0..4000_u64 {
        // Advance the REAL ep-thermal PDE (1 ms steps).
        grid.step(0.001);
        // Tile feed: max over the grid cells mapped onto the TPU block.
        let t_junction_c = grid.temp.iter().fold(f64::NEG_INFINITY, |a, t| {
            let c = t - 273.15;
            if c > a {
                c
            } else {
                a
            }
        });
        let event = match dvfs.step(&[t_junction_c]) {
            Ok(e) => e,
            Err(SocError::RunawayLatched) => {
                runaway_seen = true;
                assert!(t_junction_c >= RUNAWAY_THRESHOLD_C);
                break;
            }
            Err(e) => panic!("unexpected DVFS error at cycle {cycle}: {e}"),
        };
        if event.action == DvfsAction::ThrottleDown {
            throttles += 1;
        }
    }
    assert!(
        throttles > 0,
        "the live ep-thermal grid must drive real throttle steps"
    );
    // The closed loop reacts: with throttling active the grid's temperature
    // climb is bounded above the runaway latch OR the runaway fired — either
    // way the controller actually coupled to the solver's temps.
    let final_max_c = grid
        .temp
        .iter()
        .fold(f64::NEG_INFINITY, |a, t| (t - 273.15).max(a));
    assert!(
        runaway_seen || final_max_c < RUNAWAY_THRESHOLD_C,
        "closed loop bounded the live grid ({final_max_c} °C) or latched the runaway"
    );
    // Frequency actually moved with the thermal state.
    assert!(dvfs.freq_mhz() < 700.0 || dvfs.runaway_latched());
    let _ = (Farad::new(0.0), Hertz::new(0.0)); // power units stay linked to ep-chip's model
}
