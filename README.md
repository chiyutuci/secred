# SecRed

SecRed reduces Feynman integrals to a basis of **master integrals** by solving
integration-by-parts (IBP) identities. Given a loop topology and a list of target
integrals, it returns every target as a rational-coefficient linear combination of
the masters:

```
I(a₁, …, aₙ)  =  Σ  cᵢ · Mᵢ ,      cᵢ ∈ ℚ(d, s, t, …)
```

The IBP and Lorentz-invariance relations are generated and reduced **sector by
sector**, and the relations of all sectors are then collected into a single global
system whose solution gives the final reduction. Two backends are available for that
global step:

| `global_tool` | Backend | Notes |
|---------------|---------|-------|
| `kira` (default) | [Kira](https://gitlab.com/kira-pyred/kira) as an external subprocess | The system is written to disk and Kira is invoked on it. |
| `finiteflow` | [FiniteFlow](https://github.com/peraro/finiteflow), in-process | The elimination runs inside SecRed and FiniteFlow reconstructs the rational coefficients. No external tool is required. |

**Preprint:** _link to be added_ (placeholder).

---

## Requirements

| Dependency | Purpose | Notes |
|------------|---------|-------|
| C++20 compiler | build | GCC ≥ 10 or Clang ≥ 12 |
| CMake ≥ 3.16, Ninja (or Make) | build | |
| [GMP](https://gmplib.org/), [MPFR](https://www.mpfr.org/) | build | used by FLINT and by the GiNaC/CLN stack |
| [FLINT](https://flintlib.org/) | build | arbitrary-precision rationals (`lib/arith`) |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp) | build | input files |
| [GiNaC](https://www.ginac.de/) (with CLN) | build | symbolic manipulation |
| [FiniteFlow](https://github.com/peraro/finiteflow) | build **and** run | finite-field linear algebra and rational reconstruction; required by both backends |
| [Kira](https://gitlab.com/kira-pyred/kira) | run | only when `global_tool: kira` (the default) |
| [Fermat](http://home.bway.net/lewis/) | run | only when Kira reduces a *symbolic* system (see below) |

On Debian/Ubuntu most of them come from the package manager:

```bash
sudo apt install build-essential cmake ninja-build libgmp-dev libmpfr-dev \
     libflint-dev libyaml-cpp-dev libginac-dev libcln-dev
```

FiniteFlow is built from source and installs `libfflow`:

```bash
git clone https://github.com/peraro/finiteflow
cd finiteflow && ./configure --prefix=/usr/local && make -j && sudo make install
```

Kira is installed separately, e.g.

```bash
git clone https://gitlab.com/kira-pyred/kira
cd kira && ./configure --prefix=/usr/local && make -j && sudo make install
```

and must then be reachable through `PATH` (or through the `kira_cmd` setting).

### Kira version

The Kira backend is validated against **Kira 2.3**; use that version to reproduce
the numbers in the paper. Kira 3.0 selects the **same master basis** — verified on
`box.yaml` (9 masters), `dbox.yaml` (12) and `sunrise.yaml` (7) — but it does not
always carry a target all the way down to that basis: with `box.yaml` it returns the
first target as a combination that still contains the non-master integral
`[1, 2, 1, 2]`, and with `dbox.yaml` the target `[2, 1, 1, 1, 1, 1, 1, -1, 0]` keeps
`[1, 1, 2, 1, 1, 1, 1, -1, 0]`. Such results are algebraically equal to the 2.3 ones
(verified exactly, coefficient by coefficient, on both examples; correct, just not
fully reduced); adding the leftover integral to `targets` and re-running gives the
fully reduced form with either version.

### Fermat

When the global system carries symbolic coefficients (the family's `dimension` is a
symbol such as `d`, or the kinematics are kept symbolic), Kira solves it with
Fermat. Install Fermat and point `FERMATPATH` at the binary:

```bash
export FERMATPATH=/path/to/fer64
```

Purely numeric systems do not need Fermat: SecRed then injects a dummy variable
(`dum`) into one equation so that Kira accepts a symbol-free rational system.

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The dependencies are located with CMake's `find_library`, so a library installed
under a non-standard prefix must be passed in:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/finiteflow
```

This produces the program `build/secred` and the unit-test binary `build/secred_tests`.

## Run

```bash
./build/secred example/box.yaml
```

The input file is described in the next section; `settings.cache_dir` collects
everything the run writes (see [Output](#output)). All settings are optional.

## Input file

```yaml
family:
  name:        box                # topology name (free text)
  dimension:   399/100            # space-time dimension: a number, or a symbol such as d
                                  # (default: the symbol d, i.e. symbolic)
  internals:   [ l1 ]             # loop momenta
  externals:   [ p1, p2, p3 ]     # external momenta
  invariants:  [ [s, 2] ]         # kinematic invariants with their mass dimension
  invar_one:   s                  # [optional] invariant to be set to 1
  sps_rules:                      # scalar products: pi·pj = expr
    - [ p1, p1, 0 ]
  propagators:                    # propagator numerators and masses
    - [ l1, 2 ]
  cut:         [0, 0, 0, 0]       # [optional] cut flags for the propagators

settings:                         # all optional
  threads: 64
  cache_dir: cache_box

targets:                          # propagator index vectors to reduce
  - [ 2, 2, 1, 1 ]
```

Each propagator is given as `[momentum, mass]` and denotes
`momentum² − mass²`; the mass must be an invariant of mass dimension 1.
Powers of an invariant are written with `^` in the expressions
(e.g. `2^2` for `4`).

### Settings

| Key | Default | Meaning |
|-----|---------|---------|
| `cache_dir` | `cache` | Directory for all output and for Kira's scratch files. |
| `threads` | `64` | Worker threads for the sector-wise reduction. |
| `parallel_mode` | `fine` | `layer`: sectors of the same depth run as a batch; `fine`: sectors are scheduled individually. |
| `ordering_mode` | `balance` | Integral ordering: `normal` (generation order) or `balance` (default: seeds, subsector columns and the global integral numbering all follow the balance key — total power, depth, rank, max denominator power, max numerator power). |
| `subsector_filter_n` | `1` | Offset *n* of the subsector filter: a subsector integral is dropped when its boundary quantity `depth + rank` is more than *n* below the boundary. |
| `cut` | — | Cut propagator flags (family section). |
| `max_memory` | `512` | Allowed total memory in GB. Set it smaller to run on a machine with less RAM; sectors are admitted only while the process stays below this ceiling. |
| `memory_reserve` | `60` | Headroom kept free below `max_memory`, in GB. |
| `job_dir` | current dir | Base directory used to resolve a relative `cache_dir` before Kira is launched there. |
| `kira_cmd` | `kira` | Command used to invoke Kira (path or name on `PATH`). |
| `kira_jobs_file` | `jobs.yaml` | Name of the Kira job file. |
| `kira_bunch_size` | `64` | Kira's `--bunch_size` (integrals reduced per bunch). |
| `kira_parallel` | `64` | Kira's `-p` (parallel workers; Kira also starts this many Fermat instances in its final result-writing step — see Examples). |
| `global_tool` | `kira` | `kira` or `finiteflow` — see the table above. |
| `global_finiteflow_threads` | `64` | Threads used by the in-process FiniteFlow reconstruction. |

The equation set of the global reduction (forward closure plus numeric sufficiency repair),
the elimination engine of both stages (ffred: forward elimination with back substitution) and
the seed-boundary rule of the sector stage (independent depth and rank hard caps plus a cap on
the total power `depth + rank` starting at the top of the seed box) are fixed and have no keys;
an input file that still carries the former keys `equation_selection`, `sector_elimination`,
`boundary_mode`, `global_otf` or `subsector_depth_offset` parses, but those lines are ignored.

## Output

Everything is written below `cache_dir`:

| Path | Content |
|------|---------|
| `reductions` | The result: for each target, one line per term, `[a₁, …, aₙ]*<coefficient>`, with the target's own row first (coefficient `-1`) followed by the masters. A blank line separates consecutive targets. |
| `masters_sector` | The master candidates left by the sector-wise stage, one `[a₁, …, aₙ]` per line (this is not yet the final basis — the global step may keep fewer). |
| `global/results/Tuserweight/masters` | The master integrals of the global reduction: the final basis. |
| `masters_final` | The same final basis, written by the FiniteFlow backend as soon as its learn pass has fixed it — i.e. before the reconstruction, which can be the longest phase of a run. Absent on the Kira path, which has no separate learn phase. |
| `sectors/` | Per-sector logs and target lists of the sector-wise stage. |
| `global/` | The global system, Kira's job file and results, and the FiniteFlow reconstruction data. |

The run also prints a summary with the sector-level timings (the adaptive
two-pass seeding of the sector stage) and a `Reduction Statistics` block. It
closes with a `Timing Statistics` block — the sector and global phases, the
FiniteFlow backend's learn and reconstruction phases, the total wall clock and
the peak memory. Lines from the in-process FiniteFlow backend carry the
`[fflow]` tag, so `grep '\[fflow\]'` collects that whole phase.

## Examples

| File | Topology | Dimension | `global_tool` | Typical runtime |
|------|----------|-----------|---------------|-----------------|
| `example/box.yaml` | one-loop box, 4 propagators | symbolic `d`, `s`, `t`, `m` | `kira` | seconds |
| `example/dbox.yaml` | massless double box, 9 propagators | symbolic `d`, `s`, `t` | `kira` | seconds |
| `example/sunrise.yaml` | two-loop massive sunrise, 5 propagators | symbolic `d` | `kira` | seconds |
| `example/topoA.yaml` | two-loop three-point with masses, 9 propagators | symbolic `d`, `s`, `t`, `u`, `m` | `kira` | minutes |
| `example/F234_N.yaml` | two-loop five-point non-planar, 9 propagators | numeric | `finiteflow` | minutes |
| `example/F234_S.yaml` | the same family | numeric `d`, symbolic `s`, `t` | `kira` | hours |
| `example/F1.yaml` | four-loop, 14 propagators | numeric | `finiteflow` | days |

`box.yaml` is the reference input: it is the file used to validate changes to the
input format.

The three small examples set `kira_parallel: 8`. At the end of every Kira run its
`kira2file: Reconstruct results` step starts one Fermat/Blackbox instance per
`-p` slot — serially, ~0.15 s each — even when, as here, every target is already
reduced and the instances are never used. With the default 64 that fixed start-up
costs ~9 s (Kira 2.3 and 3.0 alike); with 8 it is ~2 s, and the reductions are
byte-identical (verified on all three examples).

## Tests

```bash
cmake --build build --target secred_tests && ./build/secred_tests
```

## Citation

If you use SecRed in your work, please cite the preprint (reference to be added):

```bibtex
@article{...}
```

## Licence

MIT — see [LICENSE](LICENSE).
