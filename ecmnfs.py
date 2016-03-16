#! /usr/bin/env python3

# This is written to Python 3.5 standards (though so far I think it would work fine on e.g. 3.3)
# Note: tab depth is 5, as a personal preference


#    Copyright (C) 2016 Bill Winslow
#
#    This module is a part of the ecmnfs package.
#
#    This program is libre software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#    See the LICENSE file for more details.

####################################################################################################

# Some functions to decide the crossover between NFS and ECM, given: total work required for NFS
# and work required per curve at the given ECM level

# The premise for this module is that while ECM-work-done/odds-of-success-for-that-work is less than
# the total NFS work required, keep going.
# Thus, given the relevant parameters, this module will calculate the effective ECM-work-done
# crossover point, where it becomes more effective to switch to NFS.

from functools import lru_cache
from math import isnan, floor, log, exp, expm1 # Higher precision than exp(x) - 1 for small |x|
lg10 = log(10)
def rnd(n):
     return int(round(n))


@lru_cache()
def odds_of_factor_between(a, b, n=105720747827131650775565137946594727648048676926428801477497713261333062158444658783837181718187127016255169032147325982158719006483898971407998273736975091062494070213867530300194317862973608499):
     '''This gives the odds that n has a factor between 10^a and 10^b.'''
     lgn = log(n)/lg10
     if not (lgn/2 >= b > a > 0):
          raise ValueError("sqrt(n) must be larger than 10^b must be larger than 10^a must be larger than 1")
     # This function follows the calculation given in:
     # Robert D. Silverman and Samuel S. Wagstaff,
     # A practical analysis of the elliptic curve factoring algorithm,
     # Math. Comp. 61 (1993), no. 203, 445-462. MR 1122078, http://dx.doi.org/10.1090/S0025-5718-1993-1122078-7
     # Full text freely available at
     # http://www.ams.org/journals/mcom/1993-61-203/S0025-5718-1993-1122078-7/S0025-5718-1993-1122078-7.pdf
     # The argument is given for factors between y and y^(1+eps), so y=10^a and eps = b/a-1
     # Next let y = n^delta, then d = floor(1/delta) is the number of terms in the sum given.
     # 10^a = n^delta => a log(10) = delta log(n) => delta = a log(10)/log(n)
     d = floor(lgn/a)
     # The sum is the first d terms of the Taylor series for 1-exp(-x), where x = the sum of the inverse primes
     # between 10^a and 10^b. Merterns' second theorem gives this is as log log 10^b - log log 10^a
     # = log (b log(10)) - log(a log(10)) = log(b) + log(10) - log(a) - log(10) = log(b/a)
     x = log(b/a)
     odds = 0
     factorial = 1
     power = 1
     sign = -1
     for i in range(1, d+1):
          sign = -sign
          power *= x
          factorial /= i
          term = sign*power*factorial
          if isnan(term):
               break
          newodds = odds + term
          if newodds == odds:
               break
          odds = newodds

     return odds


def calc_odds_factor_exists(digit_level, twork_at_prior_level=1):
     # This accounts for the amount of work done at the digits-5 level (default assumed to be standard 1t-level)
     # On the whole, the next several lines are relatively shoddy, all things considered. There's a ton of
     # guesswork, approximation, and straight up fudge work that may even be patently false. But I guess
     # it's a decent first order approximation... hopefully
     #
     odds_of_factor = odds_of_factor_between(digit_level-5, digit_level)
     missed_factor_odds = exp(-twork_at_prior_level) * odds_of_factor_between(digit_level-10, digit_level-5)
     # Ignore the nearly trivial work at lower levels
     
     # While the lower level ECM may have missed a factor, there's also a chance for it to find a
     # factor larger than digits-5.
     # To quantify this, I follow the analysis here: http://mersenneforum.org/showthread.php?p=427989#post427989
     # The bottom line is that at pseudo-optimal B1 bounds (there is some slight variation between different
     # bounds), ECM effort doubles roughly every 2 digits, so that (e.g.) 1t55 ~= 1/(2^(5/2)) t60 ~= 0.1768 t60.
     # Thus the odds of a higher factor being missed are exp(-2^(-2.5))
     five_digit_ecm_effort_ratio = 2**(5/2) # This is particularly sensitive to the ECM information being derived
                                            # from pseudo-optimal B1 bound
     equiv_effort_at_higher_digit_level = twork_at_prior_level/five_digit_ecm_effort_ratio
     odds_higher_factor_missed = exp(-equiv_effort_at_higher_digit_level)

     return missed_factor_odds + odds_higher_factor_missed*odds_of_factor
     # This assumes all factors between digits-9 and digits are equally likely to be found by a given curve,
     # which is distinctly untrue


def generate_functions(median_curves, work_per_curve, odds_factor_exists, nfs_work):
     '''This takes the input data and generates the interesting functions to be optimized, without
     actually doing the optimization yet.'''
     # We want to find the crossover where ecm-work/odds_of_success = nfs_work.
     # That is, curves*work_per_curve/[odds*(1-exp(-curves/median))] = nfs_work, which is transcendental.
     # Fortunately it's an analytic function with one pathology at zero, yet even there f and all its
     # derivatives have a limit at 0, i.e., the Taylor series converges everywhere. So we can solve
     # the equation with blind and stupid Newton iteration.
     cdf = lambda curves: odds_factor_exists*-expm1(-curves/median_curves)
     # limit as x->0 of x/(1-exp(-x/a)) is a
     ecm_func = lambda curves: work_per_curve * (curves/cdf(curves) if curves != 0 else median_curves/odds_factor_exists)
     f = lambda curves: ecm_func(curves) - nfs_work

     # limit as x->0 of d/dcurves ecm_func = 1/2
     def fprime(curves):
          if curves == 0:
               return work_per_curve / (2*odds_factor_exists)
          arg = curves/median_curves
          l = -expm1(-arg) # 1-exp(-arg)
          # d/dx (x/l) = (l - x*l')/l^2
          # x*l' = arg*(1-l)
          return work_per_curve / odds_factor_exists * (l+arg*(l-1)) / (l*l)

     return cdf, ecm_func, f, fprime


def solve_xover(median_curves, f, fprime):
     '''This is the meat, the workhorse. Everything else is a helper to precompute the arguments for
     this function.
     
     The return value is either number of curves to do, or negative if you should just do the usual
     ECM before proceeding to the next level. 
     '''
     if f(1) > 0:
          return 0

     x0 = median_curves
     x1 = x0 - f(x0)/fprime(x0)
     while abs(x1 - x0) > 0.5:
          x0 = x1
          x1 = x0 - f(x0)/fprime(x0)

     x1 = rnd(x1)
     if x1 > 3*median_curves:
          return -x1
     return x1


def analyze_one_digit_level(digit_level, median_curves, work_per_curve, nfs_work, twork_at_prior_level=1):
     '''This puts a couple other functions together.
     Return value is (count, ecm_func, cdf, odds_factor_exists).'''

     odds_factor_exists = calc_odds_factor_exists(digit_level, twork_at_prior_level)

     cdf, ecm_func, f, fprime = generate_functions(median_curves, work_per_curve, odds_factor_exists, nfs_work)

     count = solve_xover(median_curves, f, fprime)

     return count, cdf, ecm_func, odds_factor_exists


def analyze_all_digit_levels(ecm_data, nfs_work):
     '''`ecm_data` is an ordered list of (digit_level, median_curves, work_per_curve) tuples
     describing all possible digit levels of work. This function will then analyze how much ECM
     should be done. The results will return a specific digit level and curve count, and it is
     assumed that all lower levels will also get one t-effort's worth of work in addition to the
     final level curve count.'''
     ecm_data.sort(key=lambda tup: tup[0])
     # This is sadly a bit of a mess...

     # First we analyze all curve levels, and then toss all the highest ones that aren't worthwhile.
     # Then we adjust the final level to account for the total work done at the prior levels.
     results = []
     for digit_level, median_curves, work_per_curve in ecm_data:
          odds_factor_exists = calc_odds_factor_exists(digit_level, 1) # 1 t-effort of work is assumed
          cdf, ecm_func, f, fprime = generate_functions(median_curves, work_per_curve, odds_factor_exists, nfs_work)
          if f(1) > 0: # This and all higher levels aren't worthwhile
               break
          else:
               results.append((digit_level, median_curves, work_per_curve, odds_factor_exists, cdf, ecm_func))
     else:
          print("The highest digit level {} has positive worthwhile work, be sure you shouldn't be going higher as well".format(digit_level))

     ###############################################################################################
     # Now we know which levels are worthwhile, we can adjust the final level work to account for
     # previous level work done
     work_budget = nfs_work
     out = []
     # For all but the last level, we do just a standard t-effort of work
     for digit_level, median_curves, _, _, cdf, ecm_func in results[:-1]:
          equiv_work_at_this_level = ecm_func(median_curves) # ecm_func = total_work/cdf
          work_budget -= equiv_work_at_this_level
          out.append((digit_level, equiv_work_at_this_level, cdf(median_curves)))
     # Now redo the final level cross over
     digit_level, median_curves, work_per_curve, odds_factor_exists, _, _ = results[-1]
     cdf, ecm_func, f, fprime = generate_functions(median_curves, work_per_curve, odds_factor_exists, work_budget)
     count = solve_xover(median_curves, f, fprime)
     if count == 0:
          raise ValueError("Damn, impressive. You ran into a strange edge case where accounting for prior work renders the highest level no longer worthwhile.")
     return count, digit_level, median_curves, cdf, ecm_func, odds_factor_exists, rnd(work_budget), out

# End calculation functions
####################################################################################################
# Begin interface functions

# First the old school, one-level-at-a-time interface

def main(argv):
     # Bit of a mess
     if len(argv) != 3 and len(argv) != 4:
          raise ValueError("Args must be 'digits nfs_work [prior level t-effort]'")

     digit_level, nfs_work = [int(arg) for arg in argv[1:3]]           
     if digit_level not in ecm_table:
          raise ValueError("Only have information to analyze the following digit levels:\n{}".format(ecm_table.keys()))

     median_curves, work_per_curve = ecm_table[digit_level]
     if len(argv) == 4:
          count, cdf, ecm_func, odds_factor_exists = \
               analyze_one_digit_level(digit_level, median_curves, work_per_curve, nfs_work, float(argv[3]))
     else:
          count, cdf, ecm_func, odds_factor_exists = \
               analyze_one_digit_level(digit_level, median_curves, work_per_curve, nfs_work)

     print_result(count, digit_level, cdf, odds_factor_exists, nfs_work)
     do_fancy_plots(count, digit_level, median_curves, cdf, ecm_func, nfs_work)

###############################################################
# Now the newer analyze-all-levels stuff

ecm_table= [(40, 2350, 23/3600),
            (45, 4480, 84/3600),
            (50, 7553, 316/3600),
            (55, 17769, 880/3600),
            (60, 42017, 2040/3600),
            (65, 69408, 1.77)
           ]
# Digit level, median curve count, and hours per curve on a 195 digit number
# Median = scale = 1/rate in CDF(x) = 1 - exp(-x/scale)

def alt_main(argv):
     count, digit_level, median_curves, cdf, ecm_func, odds_factor_exists, work_budget, out = analyze_all_digit_levels(ecm_table, int(argv[1]))

     for digits, equiv_work_at_this_level, odds in out:
          print("Doing the median curves at {} digits is {} equivalent work with {:.1f}% odds of success".format(digits, rnd(equiv_work_at_this_level), odds*100))
     
     print_result(count, digit_level, cdf, odds_factor_exists, work_budget)
     do_fancy_plots(count, digit_level, median_curves, cdf, ecm_func, work_budget)

####################################################################################################
# Dump this out of the way

def print_result(count, digit_level, cdf, odds_factor_exists, work_budget):
     if count == 0:
          print("Not even one curve should be done, forget about ECM at {} digits.".format(digit_level))
     elif count < 0:
          print("Doing triple the expected curves would still be worth it, so you should definitely consider doing work at {} digits".format(digit_level+5))
     else:
          print("With ~{:.1f}% odds of a factor existing and ~{:.1f}% net odds of success, you should do {} curves at {} digits ({} equiv work budget) before switching to NFS".format(odds_factor_exists*100, cdf(count)*100, count, digit_level, work_budget))

def do_fancy_plots(count, digit_level, median_curves, cdf, ecm_func, nfs_work):
     try:
          import matplotlib.pyplot as plt
          import numpy as np
          x=np.arange(0, 3*median_curves+1, 1)
          y=np.array([ecm_func(a) for a in x])
          z=np.array([nfs_work for a in x])
          plt.subplot(2, 1, 1)
          zprime=np.array([100*cdf(a) for a in x])
          plt.plot(x, zprime)
          plt.title('Net odds of factor vs curves ({} digit level)'.format(digit_level))
          plt.xlabel('Curves')
          plt.ylabel('Net odds (%) of factor')
          if count > 0:
               dotx, doty = count, cdf(count)*100
               plt.plot([dotx], [doty], 'ro')
               plt.annotate('~{:.1f}% odds of success at {} curves'.format(doty, count), xy=(dotx, doty), xytext=(5, -15), textcoords='offset points')
               

          plt.subplot(2, 1, 2)
          plt.plot(x, y, x, z)
          plt.axis(xmin=0, ymin=0)
          plt.title('ECM work/succes odds vs curves ({} digit level)'.format(digit_level))
          plt.xlabel('Curves')
          plt.ylabel('ECM work/success odds')
          if count > 0:
               dotx, doty = count, ecm_func(count)
               plt.plot([dotx], [doty], 'ro')
               plt.annotate('At {} total work budget, the crossover is {} curves'.format(nfs_work, count), xy=(dotx, doty), xytext=(5, -15), textcoords='offset points')

          plt.tight_layout()
          plt.show()
     except:
          pass

if __name__ == '__main__':
     from sys import argv
     if len(argv) > 2:
          main(argv)
     else:
          alt_main(argv)
