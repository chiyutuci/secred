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
**inside SecRed** — no external solver is invoked — and two engines are available
for it:

| `global_tool` | Backend | Notes |
|---------------|---------|-------|
| `finiteflow` (default) | [FiniteFlow](https://github.com/peraro/finiteflow) | The modular elimination runs inside SecRed and FiniteFlow reconstructs the rational coefficients. Handles a numeric and a symbolic system alike. |
| `firefly` | [FireFly](https://github.com/peraro/firefly)'s `Reconstructor` | For a *symbolic* system: the same elimination feeds a black box that FireFly reconstructs from. A numeric system takes the `finiteflow` route either way. |

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
| [FiniteFlow](https://github.com/peraro/finiteflow) | build and run | finite-field linear algebra and rational reconstruction |
| [FireFly](https://github.com/peraro/firefly) | build and run | FireFly's `Reconstructor`, for the `global_tool: firefly` backend |

On Debian/Ubuntu most of them come from the package manager:

```bash
sudo apt install build-essential cmake ninja-build libgmp-dev libmpfr-dev \
     libflint-dev libyaml-cpp-dev libginac-dev libcln-dev
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
| `cache_dir` | `cache` | Directory for all output. |
| `threads` | `64` | Worker threads for the sector-wise reduction. |
| `parallel_mode` | `fine` | `layer`: sectors of the same depth run as a batch; `fine`: sectors are scheduled individually. |
| `ordering_mode` | `balance` | Integral ordering: `normal` (generation order) or `balance` (default: seeds, subsector columns and the global integral numbering all follow the balance key — total power, depth, rank, max denominator power, max numerator power). |
| `subsector_filter_n` | `1` | Offset *n* of the subsector filter: a subsector integral is dropped when its boundary quantity `depth + rank` is more than *n* below the boundary. |
| `cut` | — | Cut propagator flags (family section). |
| `max_memory` | `512` | Allowed total memory in GB. Set it smaller to run on a machine with less RAM; sectors are admitted only while the process stays below this ceiling. |
| `memory_reserve` | `60` | Headroom kept free below `max_memory`, in GB. |
| `global_tool` | `finiteflow` | `finiteflow` or `firefly` — see the table above. |
| `global_finiteflow_threads` | `64` | Threads used by the global reduction's reconstruction (for `firefly`, how many probes run at once). |

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
| `masters_final` | The same final basis, written as soon as the learn pass has fixed it — i.e. before the reconstruction, which can be the longest phase of a run. |
| `sectors/` | Per-sector logs and target lists of the sector-wise stage. |
| `global/` | The global system, the masters the learn pass fixed, and the reconstruction data. |

The run also prints a summary with the sector-level timings (the adaptive
two-pass seeding of the sector stage) and a `Reduction Statistics` block. It
closes with a `Timing Statistics` block — the sector and global phases, the
global stage's learn and reconstruction phases, the total wall clock and the peak
memory. Lines from the FiniteFlow backend carry the `[fflow]` tag, so
`grep '\[fflow\]'` collects that whole phase.

## Examples

| File | Topology | Dimension | `global_tool` | Typical runtime |
|------|----------|-----------|---------------|-----------------|
| `example/box.yaml` | one-loop box, 4 propagators | symbolic `d`, `s`, `t`, `m` | `finiteflow` | seconds |
| `example/dbox.yaml` | massless double box, 9 propagators | symbolic `d`, `s`, `t` | `finiteflow` | seconds |
| `example/sunrise.yaml` | two-loop massive sunrise, 5 propagators | symbolic `d` | `finiteflow` | seconds |
| `example/topoA.yaml` | two-loop three-point with masses, 9 propagators | symbolic `d`, `s`, `t`, `u`, `m` | `finiteflow` | minutes |
| `example/F234_N.yaml` | two-loop five-point non-planar, 9 propagators | numeric | `finiteflow` | minutes |
| `example/F234_S.yaml` | the same family | numeric `d`, symbolic `s`, `t` | `finiteflow` | hours |
| `example/F1.yaml` | four-loop, 14 propagators | numeric | `finiteflow` | days |

`box.yaml` is the reference input: it is the file used to validate changes to the
input format.

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
