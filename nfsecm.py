#! /usr/bin/python3

# Some functions to decide the crossover between NFS and ECM, given: thread hours required for NFS
# and thread hours required per curve at the given ECM level

# The premise for this module is that while ECM-work-done/odds-of-success-for-that-work is less than
# the total NFS work required, keep going.
# Thus, given the relevant parameters, this module will calculate the effective ECM-work-done
# crossover point, where it becomes more effective to switch to NFS.

from math import exp, expm1 # Higher precision than exp(x) - 1 for small |x|

ecm_table= {50: (7553, 316/3600),
            55: (17769, 880/3600),
            60: (42017, 2040/3600),
            65: (69408, 1.77)
           }
# Digit level, median curve count, and hours per curve on a 195 digit number
# Median = scale = 1/rate in CDF(x) = 1 - exp(-x/scale)

def odds_of_factor_between(m, n):
     # Returns the odds that there's a factor with size between m and n digits inclusive
     odds = 0
     for d in range(m, n+1):
          odds += 1/d
     return odds

def compare_nfs_ecm(nfs_hours, median_curves, hours_per_curve, odds_factor_exists):
     '''This is the meat, the workhorse. Everything else is a helper to precompute the arguments for
     this function.
     
     The arguments are self explanatory. The return value is either number of curves to do, or
     negative if you should just do the usual ECM before proceeding to the next level.
     '''
     # Now we want to find the crossover where ecm-work/odds_of_success = nfs_work.
     # That is, curves*work_per_curve/[odds*(1-exp(-curves/median))] = nfs_work, which is transcendental.
     # Fortunately it's an analytic function with no pathologies, so a stupid Newton iteration to
     # find the solution will work fine.
     f = lambda curves: curves*hours_per_curve/(-expm1(-curves/median_curves)) - odds_factor_exists*nfs_hours

     if f(1) > 0:
          return 0
     if f(3*median_curves) < 0:
          return -median_curves

     def fprime(curves):
          arg = curves/median_curves
          l = -expm1(-arg) # 1-exp(-arg)
          # d/dx (x/l) = (l - x*l')/l^2
          # x*l' = arg*(1-l)
          return hours_per_curve * ( (l+arg*(l-1)) / (l*l) )

     x0 = median_curves
     x1 = x0 - f(x0)/fprime(x0)
     while x1 - x0 > 1:
          x0 = x1
          x1 = x0 - f(x0)/fprime(x0)

     return int(x1)

def analyze_ecm_nfs_xover(digit_level, nfs_hours, include_missed_factors=1/2):
     if digit_level not in ecm_table:
          raise ValueError("Only have information to analyze the following digit levels:\n{}".format(ecm_table.keys()))
     median_curves, hours_per_curve = ecm_table[digit_level]

     # This assumes that ECM has been run to one full t-level at the digits-5 level
     odds_of_factor = odds_of_factor_between(digit_level-4, digit_level)
     missed_factor_odds = exp(-1) * odds_of_factor_between(digit_level-9, digit_level-5)
     # While the lower level ECM may have missed a factor, there's also a chance for it to find a
     # factor larger than digits-5. In lieu of actually quantifying those odds, we merely fudge off
     # some of the missed factor odds.
     missed_factor_odds *= include_missed_factors
     odds_factor_exists = odds_of_factor + missed_factor_odds

     count = compare_nfs_ecm(nfs_hours, median_curves, hours_per_curve, odds_factor_exists)

     if count == 0:
          print("Not even one curve should be done, forget about ECM at {} digits.".format(digit_level))
          return count
     if count < 0:
          print("Doing triple the expected curves would still be worth it, so you'll need to go to {} digits".format(digit_level+5))
          return count
     print("With roughly {:3f}% odds of a factor existing, you should do {} curves at {} digits before switching to NFS".format(odds_factor_exists*100, count, digit_level))
     return count

def main():
     from sys import argv
     if len(argv) == 3:
          analyze_ecm_nfs_xover(int(argv[1]), int(argv[2]))
     elif len(argv) == 4:
          analyze_ecm_nfs_xover(int(argv[1]), int(argv[2]), float(argv[3]))
     else:
          raise ValueError("Args must be 'digits nfs_hours [missed factor fraction]'")

if __name__ == '__main__':
     main()