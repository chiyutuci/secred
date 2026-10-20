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
system whose solution gives the final reduction. That global step runs entirely
**inside SecRed** — no external solver is invoked — with one of two engines, and
which one is not a setting: it follows from the system itself.

| System | Engine | Why |
|--------|--------|-----|
| numeric (a numeric `dimension`, e.g. `399/100`) | [FiniteFlow](https://github.com/peraro/finiteflow) | The modular elimination runs inside SecRed and FiniteFlow reconstructs the rational coefficients. |
| symbolic (`dimension: d`, or any invariant in use) | [FireFly](https://github.com/peraro/firefly)'s `Reconstructor` | The same elimination feeds a black box that FireFly reconstructs from, which copes with the symbols in the coefficients. |

A system counts as symbolic when `dimension` is not a number, or when it has an
invariant besides the one named by `invar_one`; `invar_one` alone does not make a
system symbolic, since the invariant is set to 1 before the reduction.

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
| [jemalloc](https://jemalloc.net/) | build and run | the allocator — **required**, not optional; see below |
| [FiniteFlow](https://github.com/peraro/finiteflow) | build and run | finite-field linear algebra and rational reconstruction |
| [FireFly](https://github.com/peraro/firefly) | build and run | FireFly's `Reconstructor`, the global engine of a symbolic system |

On Debian/Ubuntu most of them come from the package manager:

```bash
sudo apt install build-essential cmake ninja-build libgmp-dev libmpfr-dev \
     libflint-dev libyaml-cpp-dev libginac-dev libcln-dev libjemalloc-dev
```

FiniteFlow and FireFly are built from source and install `libfflow` / `libfirefly`:

```bash
git clone https://github.com/peraro/finiteflow
cd finiteflow && ./configure --prefix=/usr/local && make -j && sudo make install

git clone https://github.com/peraro/firefly
cd firefly && ./configure --prefix=/usr/local && make -j && sudo make install
```

Both are linked in at build time and run with the program, so nothing has to be on
`PATH` when SecRed runs.

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
  sthreads: 64
  cache_dir: cache_box

targets:                          # propagator index vectors to reduce
  - [ 2, 2, 1, 1 ]

preferred:                        # [optional] integrals asked for as masters
  - [ 0, 0, 1, 0 ]
```

Each propagator is given as `[momentum, mass]` and denotes
`momentum² − mass²`; the mass must be an invariant of mass dimension 1.
Powers of an invariant are written with `^` in the expressions
(e.g. `2^2` for `4`).

`preferred` names integrals the user wants in the master basis, with the same
syntax as `targets`. Each entry that takes part in the reduction is ordered as the
*simplest* integral of its sector, so the elimination does not consume it as a pivot.
An entry that never appears — dropped by the subsector filter, or its sector is not
reached — is ignored. The list is a set: the order of its entries does not matter.
It is a **bias, not a guarantee**: the number of masters is fixed by the mathematics,
so an entry can still lose to another preferred integral that is linearly dependent
on it modulo the simpler integrals.

### Settings

| Key | Default | Meaning |
|-----|---------|---------|
| `cache_dir` | `cache` | Directory for all output. |
| `sthreads` | `64` | Worker threads for the sector-wise reduction. |
| `gthreads` | `sthreads` | Threads of the global phase's reconstruction: how many FireFly probes, or FiniteFlow rational reconstructions, run at once. Unset, it takes `sthreads`, so one number configures a run of either size. |
| `parallel_mode` | `fine` | `layer`: sectors of the same depth run as a batch; `fine`: sectors are scheduled individually. |
| `ordering_mode` | `balance` | Integral ordering: `normal` (generation order) or `balance` (default: seeds, subsector columns and the global integral numbering all follow the balance key — total power, depth, rank, max denominator power, max numerator power). |
| `subsector_filter_n` | `1` | Offset *n* of the subsector filter: a subsector integral is dropped when its boundary quantity `depth + rank` is more than *n* below the boundary. |
| `seed_level_offset` | automatic: `1`, or `0` when the targets are pure denominators | Relaxation of the derived seed-truncation level: `l = max(0, popcount(top) − s_max − offset)`. Larger relaxes more: more equations, fewer seeds cut. Ignored when `probe_mode: all`. |
| `probe_mode` | `targets` | Probe scope: `targets` — only a master the sector's **input** targets need may widen its seed box, a master only a spill request needs is accepted where it is; `all` — every out-of-box master reports and may widen the box, and `seed_level_offset` is void (the truncation is derived at offset 0). An unknown value is an error. |
| `cut` | — | Cut propagator flags (family section). |
| `max_memory` | `512` | Allowed total memory in GB. Set it smaller to run on a machine with less RAM; sectors are admitted only while the process stays below this ceiling. |
| `memory_reserve` | `30` | Headroom kept free below `max_memory`, in GB. |
| `max_primes` | `500` | Cap on the numeric reconstruction's primes. The CRT stops at convergence, so this bounds a run rather than driving one; if the cap is exhausted without convergence the reconstruction fails. Raise it for coefficients that need more primes (very large rational parameters, a wide reconstruction). Primes past FiniteFlow's 201-entry table are generated as needed. |

## Output

Everything is written below `cache_dir`:

| Path | Content |
|------|---------|
| `reductions` | The result: for each target, one line per term, `[a₁, …, aₙ]*<coefficient>`, with the target's own row first (coefficient `-1`) followed by the masters. A blank line separates consecutive targets. |
| `masters_sector` | The master candidates left by the sector-wise stage, one `[a₁, …, aₙ]` per line (this is not yet the final basis — the global step may keep fewer). |
| `masters_global` | The final basis: the master integrals the global reduction settled on. Written as soon as the learn pass has fixed them — i.e. before the reconstruction, which can be the longest phase of a run. |
| `sectors/` | Per-sector logs and target lists of the sector-wise stage. |
| `secred.log` | A copy of everything the run printed, so a run is readable after the fact. |
| `firefly.log` | FireFly's own log, on a run that used it. |

The global system and the reconstruction's intermediate data are kept in scratch
files that the run removes when it is done, so the cache directory holds the run's
results and its logs, not its leftovers.

The run also prints a summary with the sector-level timings (the adaptive
two-pass seeding of the sector stage) and a `Reduction Statistics` block. It
closes with a `Timing Statistics` block — the sector and global phases, the
global stage's learn and reconstruction phases, the total wall clock and the peak
memory. Lines from the FiniteFlow backend carry the `[fflow]` tag, so
`grep '\[fflow\]'` collects that whole phase; a symbolic system's run is instead
described by FireFly's own log, `firefly.log` in the cache directory.

## Examples

| File | Topology | Dimension | Global engine | Typical runtime |
|------|----------|-----------|---------------|-----------------|
| `example/box.yaml` | one-loop box, 4 propagators | symbolic `d`, `s`, `t`, `m` | FireFly | seconds |
| `example/dbox.yaml` | massless double box, 9 propagators | symbolic `d`, `s`, `t` | FireFly | seconds |
| `example/sunrise.yaml` | two-loop massive sunrise, 5 propagators | symbolic `d` | FireFly | seconds |
| `example/topoA.yaml` | two-loop three-point with masses, 9 propagators | symbolic `d`, `s`, `t`, `u`, `m` | FireFly | minutes |
| `example/F234_N.yaml` | two-loop five-point non-planar, 9 propagators | numeric | FiniteFlow | minutes |
| `example/F234_S.yaml` | the same family | numeric `d`, symbolic `s`, `t` | FireFly | hours |
| `example/F1.yaml` | four-loop, 14 propagators | numeric | FiniteFlow | days |

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

GPL-3.0 — see [LICENSE](LICENSE).
