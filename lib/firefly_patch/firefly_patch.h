#pragma once

#include <cstddef>

namespace firefly {

// Grows the 63-bit prime table (firefly::primes()) to at least n entries by generating
// the primes below its current last entry, keeping the table in descending order.  The
// first 300 entries are FireFly's own and never change.  Call it before a
// reconstruction that may ask for more prime fields than the table currently holds.
void ensure_primes(std::size_t n);

}
