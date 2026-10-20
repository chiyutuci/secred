
#pragma once

#include <vector>

#include <fflow/alg_reconstruction.hh>
#include <fflow/common.hh>
#include <fflow/matrix.hh>
#include <fflow/mp_common.hh>
#include <fflow/mp_gcd.hh>
#include <fflow/rational_function.hh>

namespace secred {

// Per-output state of the univariate linear reconstruction with partially
// recovered coefficients.  The CRT accumulators hold every numerator and
// denominator coefficient modulo the product of the primes seen so far; a
// coefficient moves to the "known" set as soon as rational reconstruction
// agrees at two consecutive primes, which removes it from the per-prime
// linear solve.
struct UniLinearState {
  std::vector<fflow::MPInt> crt_num, crt_den;
  std::vector<fflow::MPRational> rat_num, rat_den;
  std::vector<fflow::MPRational> prev_num, prev_den;
  std::vector<unsigned char> known_num, known_den;
  std::vector<unsigned char> prev_ok_num, prev_ok_den;
  fflow::MPInt mod;
};

// Number of coefficients still unknown: every numerator coefficient plus the
// denominator coefficients other than the normalization pivot.
unsigned uni_unknown_count(const UniLinearState &st, unsigned pivot);

// Seed the state from the first prime's trial function: it fixes the pivot
// normalization, which coefficients are structurally zero, and the first CRT
// residue of every coefficient.
void uni_seed(UniLinearState &st, const fflow::SparseRationalFunction &f,
              unsigned nd, unsigned dd, unsigned pivot, fflow::UInt p);

// One prime of the partial linear solve: solves only for the coefficients
// still unknown, with the recovered ones and the pivot held fixed, extends the
// CRT modulus, and promotes coefficients whose rational reconstruction agrees
// at two consecutive primes.  On success num/den hold the full coefficient
// vector modulo this prime and the state has advanced by one prime.
fflow::Ret uni_linear_step(UniLinearState &st, unsigned nd, unsigned dd,
                           unsigned pivot, fflow::Mod mod, unsigned n_checks,
                           unsigned n_singular,
                           fflow::SampledRatFun &sampledfun,
                           std::vector<fflow::UInt> &num,
                           std::vector<fflow::UInt> &den);

}
