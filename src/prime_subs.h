#pragma once

// The numeric generators substitute a prime for each symbolic parameter, by
// position.  A family with more parameters than this has no numeric
// representation: the extra symbol would survive into the coefficient text and
// be parsed as a number, so the family is refused when it is read.
inline constexpr const char *kPrimeSubs[] = {
    "2147483647", "2147483629", "2147483587", "2147483579",
    "2147483563", "2147483549", "2147483543", "2147483497",
    "2147483489", "2147483477", "2147483423", "2147483399",
    "2147483353", "2147483323", "2147483269", "2147483249",
    "2147483237", "2147483179", "2147483171", "2147483137",
    "2147483123", "2147483077", "2147483069", "2147483059",
};
inline constexpr unsigned kNumPrimeSubs =
    sizeof(kPrimeSubs) / sizeof(kPrimeSubs[0]);
