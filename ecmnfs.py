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
     
     The arguments are self explanatory. The return value (curves, cdf, ecm_func), where `curves`
     is either number of curves to do, or negative if you should just do the usual ECM before
     proceeding to the next level. `cdf` is the function describing net odds of success, and `ecm_func`
     is the function ecm_work/cdf.
     '''
     # Now we want to find the crossover where ecm-work/odds_of_success = nfs_work.
     # That is, curves*work_per_curve/[odds*(1-exp(-curves/median))] = nfs_work, which is transcendental.
     # Fortunately it's an analytic function with one pathology at zero, yet even there f and all its
     # derivatives have a limit at 0, i.e., the Taylor series converges everywhere. So we can solve
     # the equation with blind and stupid Newton iteration.
     cdf = lambda curves: odds_factor_exists*-expm1(-curves/median_curves)
     # limit as x->0 of x/(1-exp(-x/a)) is a
     ecm_func = lambda curves: hours_per_curve * (curves/cdf(curves) if curves != 0 else median_curves/odds_factor_exists)
     f = lambda curves: ecm_func(curves) - nfs_hours

     if f(1) > 0:
          return 0, cdf, ecm_func
     if f(3*median_curves) < 0:
          return -median_curves, cdf, ecm_func

     # limit as x->0 of d/dcurves ecm_func = 1/2
     def fprime(curves):
          if curves == 0:
               return hours_per_curve / (2*odds_factor_exists)
          arg = curves/median_curves
          l = -expm1(-arg) # 1-exp(-arg)
          # d/dx (x/l) = (l - x*l')/l^2
          # x*l' = arg*(1-l)
          return hours_per_curve / odds_factor_exists *  (l+arg*(l-1)) / (l*l)

     x0 = median_curves
     x1 = x0 - f(x0)/fprime(x0)
     #print("not looping:", x0, x0/median_curves, f(x0), x1)
     while abs(x1 - x0) > 0.5:
          x0 = x1
          x1 = x0 - f(x0)/fprime(x0)
          #print("looping:", x0, x0/median_curves, f(x0), x1)

     #print(x1, f(x1), f(x1-1), f(x1+1))
     return int(x1), cdf, ecm_func

def analyze_ecm_nfs_xover(digit_level, nfs_hours):
     if digit_level not in ecm_table:
          raise ValueError("Only have information to analyze the following digit levels:\n{}".format(ecm_table.keys()))
     median_curves, hours_per_curve = ecm_table[digit_level]

     # This assumes that ECM has been run to one full t-level at the digits-5 level
     odds_of_factor = odds_of_factor_between(digit_level-4, digit_level)
     missed_factor_odds = exp(-1) * odds_of_factor_between(digit_level-9, digit_level-5)
     # While the lower level ECM may have missed a factor, there's also a chance for it to find a
     # factor larger than digits-5.
     # To quantify this, I follow the analysis here: http://mersenneforum.org/showthread.php?p=427989#post427989
     # The bottom line is that at pseudo-optimal B1 bounds (there is some slight variation between different
     # bounds), ECM effort doubles roughly every 2 digits, so that (e.g.) 1t55 ~= 1/(2^(5/2)) t60 ~= 0.1768 t60.
     # Thus the odds of a higher factor being missed are exp(-2^(-2.5))
     digits_ecm_effort_doubles = 2
     equiv_effort_at_higher_digit_level = 1/(2**(5/digits_ecm_effort_doubles))
     odds_to_find_higher_factor = -expm1(-equiv_effort_at_higher_digit_level)

     odds_factor_exists = missed_factor_odds + (1-odds_to_find_higher_factor)*odds_of_factor

     count, cdf, ecm_func = compare_nfs_ecm(nfs_hours, median_curves, hours_per_curve, odds_factor_exists)

     if count == 0:
          print("Not even one curve should be done, forget about ECM at {} digits.".format(digit_level))
     if count < 0:
          print("Doing triple the expected curves would still be worth it, so you'll need to go to {} digits".format(digit_level+5))
     else:
          print("With ~{:.2f}% odds of a factor existing and ~{:.2f}% net odds of success, you should do {} curves at {} digits before switching to NFS".format(odds_factor_exists*100, cdf(count)*100, count, digit_level))

     try:
          import matplotlib.pyplot as plt
          import numpy as np
          x=np.arange(0, 3*median_curves+1, 1)
          y=np.array([ecm_func(a) for a in x])
          z=np.array([nfs_hours for a in x])
          plt.subplot(2, 1, 1)
          zprime=np.array([100*cdf(a) for a in x])
          plt.plot(x, zprime)
          plt.title('Net odds of factor vs curves ({} digit level)'.format(digit_level))
          plt.xlabel('Curves')
          plt.ylabel('Net odds (%) of factor')
          if count > 0:
               dotx, doty = count, cdf(count)*100
               plt.plot([dotx], [doty], 'ro')
               plt.annotate('~{:.2f}% odds of success at {} curves'.format(doty, count), xy=(dotx, doty), xytext=(5, -15), textcoords='offset points')
               

          plt.subplot(2, 1, 2)
          plt.plot(x, y, x, z)
          plt.axis(xmin=0, ymin=0)
          plt.title('ECM work/succes odds vs curves ({} digit level)'.format(digit_level))
          plt.xlabel('Curves')
          plt.ylabel('ECM work/success odds')
          if count > 0:
               dotx, doty = count, ecm_func(count)
               plt.plot([dotx], [doty], 'ro')
               plt.annotate('At {} NFS thread hours, the crossover is {} curves'.format(nfs_hours, count), xy=(dotx, doty), xytext=(5, -15), textcoords='offset points')

          plt.tight_layout()
          plt.show()
     except:
          pass

     return count

def main():
     from sys import argv
     if len(argv) == 3:
          analyze_ecm_nfs_xover(int(argv[1]), int(argv[2]))
     else:
          raise ValueError("Args must be 'digits nfs_hours [missed factor fraction]'")

if __name__ == '__main__':
     main()