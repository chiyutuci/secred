#include "family_generator.h"

#include <map>
#include <ranges>
#include <sstream>
#include <utility>

#include "prime_subs.h"

void FamilyGenerator::clear_ibp() {
  _ibp.clear();
  _ibp.shrink_to_fit();
}

void FamilyGenerator::release_sector_rows() {
  _ibpFF.clear();
  _ibpFF.shrink_to_fit();
  _ibpFFmod.clear();
  _ibpFFmod.shrink_to_fit();
  _ibpM.clear();
  _ibpM.shrink_to_fit();
}

// Generate the IBP rows for the current run.
void FamilyGenerator::generate(const FamilyModel &model) {
  _generate_ibp(model);

  if (model._symbols.size() == 0) {
    _generate_ibp_ff_n(model);
  } else {
    _generate_ibp_ff_m(model);
    _generate_ibp_ff_n(model);
  }
}

// Generate the IBP rows for the current run.
void FamilyGenerator::_generate_ibp(const FamilyModel &model) {
  GiNaC::ex coeff, coeffD;
  std::map<RawIntegral, GiNaC::ex> equation;

  for (size_t i = 0; i < model._nints; ++i) {
    for (size_t j = 0; j < model._nints + model._nexts; ++j) {
      equation.clear();

      if (i == j)
        equation[RawIntegral(model._nprops, 0)] += model._dimension;
      for (size_t s = 0; s < model._nprops; ++s) {
        if (j < model._nints)
          coeff = (-model._symIndices[s]) * model._internals[j] *
                  GiNaC::diff(model._propagators[s], model._internals[i]);
        else
          coeff = (-model._symIndices[s]) * model._externals[j - model._nints] *
                  GiNaC::diff(model._propagators[s], model._internals[i]);
        coeff = coeff.expand()
                    .subs(model._spsRules, GiNaC::subs_options::algebraic)
                    .expand();
        if (coeff != 0) {
          RawIntegral integral(model._nprops, 0);
          integral[s] = 1;

          coeff = coeff.subs(model._spsFromProps,
                             GiNaC::subs_options::algebraic);

          for (size_t t = 0; t < model._nprops; ++t) {
            coeffD = coeff.diff(model._symProps[t]);
            if (coeffD != 0) {
              integral[t] -= 1;
              equation[integral] += coeffD;
              integral[t] += 1;
            }
            coeff = coeff.subs(model._symProps[t] == 0);
            if (coeff == 0)
              break;
          }
          coeff = coeff.expand();
          if (coeff != 0)
            equation[integral] += coeff;
        }
      }

      IBPProto ibp;
      for (const auto &item : equation | std::ranges::views::reverse) {
        coeff = item.second.expand();
        if (coeff != 0)
          ibp.emplace_back(item.first, coeff);
      }
      if (!ibp.empty())
        _ibp.push_back(std::move(ibp));
    }
  }
  _generate_li(model);
}

// Generate the Lorentz-invariance rows.
void FamilyGenerator::_generate_li(const FamilyModel &model) {
  GiNaC::ex coeff1, coeff2, coeff, coeffD;
  std::map<RawIntegral, GiNaC::ex> equation;

  for (unsigned r = 0; r < model._nexts; ++r) {
    for (unsigned s = r + 1; s < model._nexts; ++s) {
      equation.clear();

      for (unsigned i = 0; i < model._nexts; ++i) {
        for (unsigned p = 0; p < model._nprops; ++p) {
          coeff1 = (2 * model._propagators[p].diff(model._externals[i]) *
                    (-model._symIndices[p]) * model._externals[r])
                       .expand()
                       .subs(model._spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          coeff2 = (model._externals[i] * model._externals[s])
                       .subs(model._spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          if (coeff1 != 0 && coeff2 != 0) {
            RawIntegral integral(model._nprops, 0);
            integral[p] = 1;

            coeff = (coeff1 * coeff2)
                        .expand()
                        .subs(model._spsFromProps,
                              GiNaC::subs_options::algebraic);

            for (size_t t = 0; t < model._nprops; ++t) {
              coeffD = coeff.diff(model._symProps[t]);
              if (coeffD != 0) {
                integral[t] -= 1;
                equation[integral] += coeffD;
                integral[t] += 1;
              }
              coeff = coeff.subs(model._symProps[t] == 0);
              if (coeff == 0)
                break;
            }
            coeff = coeff.expand();
            if (coeff != 0)
              equation[integral] += coeff;
          }

          coeff1 = (-2 * model._propagators[p].diff(model._externals[i]) *
                    (-model._symIndices[p]) * model._externals[s])
                       .expand()
                       .subs(model._spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          coeff2 = (model._externals[i] * model._externals[r])
                       .subs(model._spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          if (coeff1 != 0 && coeff2 != 0) {
            RawIntegral integral(model._nprops, 0);
            integral[p] = 1;

            coeff = (coeff1 * coeff2)
                        .expand()
                        .subs(model._spsFromProps,
                              GiNaC::subs_options::algebraic);

            for (size_t t = 0; t < model._nprops; ++t) {
              coeffD = coeff.diff(model._symProps[t]);
              if (coeffD != 0) {
                integral[t] -= 1;
                equation[integral] += coeffD;
                integral[t] += 1;
              }
              coeff = coeff.subs(model._symProps[t] == 0);
              if (coeff == 0)
                break;
            }
            coeff = coeff.expand();
            if (coeff != 0)
              equation[integral] += coeff;
          }
        }
      }

      IBPProto ibp;
      for (const auto &item : equation | std::ranges::views::reverse) {
        coeff1 = item.second.expand();
        if (coeff1 != 0)
          ibp.emplace_back(item.first, coeff1);
      }
      if (!ibp.empty())
        _ibp.push_back(std::move(ibp));
    }
  }
}

// Generate the numeric-path IBP rows.
void FamilyGenerator::_generate_ibp_ff_n(const FamilyModel &model) {

  GiNaC::lst primeSubs;
  if (!model._symbols.empty()) {
    for (unsigned i = 0;
         i < model._symbols.size() && i < kNumPrimeSubs; ++i)
      primeSubs.append(model._symbols[i] == GiNaC::numeric(kPrimeSubs[i]));
  }

  const bool keep_rational = model._symbols.empty();
  for (const auto &ibp : _ibp) {
    IBPProtoFF ibpFF;
    IBPProtoFFmod ibpFFmod;
    for (const auto &term : ibp) {
      GiNaC::ex item = primeSubs.nops() ? term.second.subs(primeSubs).expand()
                                        : term.second;
      std::vector<rational> coeffs;
      std::vector<ffint> coeffs_mod;
      for (unsigned i = 0; i < model._nprops; ++i) {
        std::stringstream ss;
        ss << item.coeff(model._symIndices[i]);
        if (keep_rational)
          coeffs.emplace_back(ss.str());
        coeffs_mod.emplace_back(ss.str());
        item -= item.coeff(model._symIndices[i]) * model._symIndices[i];
        item = item.expand();
      }

      std::stringstream ss;
      ss << item;
      if (keep_rational)
        coeffs.emplace_back(ss.str());
      coeffs_mod.emplace_back(ss.str());
      if (keep_rational)
        ibpFF.emplace_back(term.first, std::move(coeffs));
      ibpFFmod.emplace_back(term.first, std::move(coeffs_mod));
    }
    if (keep_rational)
      _ibpFF.emplace_back(std::move(ibpFF));
    _ibpFFmod.emplace_back(std::move(ibpFFmod));
  }
}

// Generate the symbolic-path IBP rows.
void FamilyGenerator::_generate_ibp_ff_m(const FamilyModel &model) {

  for (const auto &ibp : _ibp) {
    IBPProtoM ibpM;
    for (const auto &term : ibp) {
      GiNaC::ex item = term.second;
      std::vector<GiNaC::ex> coeffs;
      for (unsigned i = 0; i < model._nprops; ++i) {
        GiNaC::ex coe = item.coeff(model._symIndices[i]);
        coeffs.emplace_back(coe);
        item -= item.coeff(model._symIndices[i]) * model._symIndices[i];
        item = item.expand();
      }

      coeffs.emplace_back(item);
      ibpM.emplace_back(term.first, std::move(coeffs));
    }
    _ibpM.emplace_back(std::move(ibpM));
  }
}
