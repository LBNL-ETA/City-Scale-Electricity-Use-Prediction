# copyright: https://github.com/LBNL-JCI-ICF/better

from scipy import optimize, stats
import matplotlib.pyplot as plt
import numpy as np
import os

class InverseModel:
    def __init__(self, temperature, load, var_name,
                 energy_type='Energy type unknown', significance_threshold=0.1):

        '''
        temperature: np.array
        load: np. array
        var_name: regressed variable name, shown in the ylabel of the figure
        '''

        if (np.size(load) != np.size(temperature)):
            print("Please make sure load and temperature arrays have the same length")
        else:
            self.temperature = temperature
            self.load = load
            self.y_name = var_name
            self.energy_type = energy_type
            self.hcp_bound_percentile = 45
            self.ccp_bound_percentile = 55
            percentiles = self.hcp_bound_percentile, self.ccp_bound_percentile
            self.hcp, self.ccp = np.percentile(self.temperature,
                                               percentiles)  # Set initial boundaries for change-point models
            self.hcp_min = self.hcp  # Heating change-point minimum
            self.hcp_max = self.ccp  # Heating change-point maximum
            self.ccp_min = self.hcp  # Cooling change-point minimum
            self.ccp_max = self.ccp  # Cooling change-point minimum
            self.base_min = 0  # Baseload minimum
            self.base_max = np.inf  # Baseload maximum
            self.hsl_min = -np.inf  # Heating slope minimum
            self.hsl_max = 0  # Heating slope maximum
            self.csl_min = 0  # Cooling slope minimum
            self.csl_max = np.inf  # Cooling slope maximum
            self.significance_threshold = significance_threshold
            self.hsl_insignificant = False  # assume significant heating slope
            self.csl_insignificant = False  # assume significant cooling slope
            self.best_model = None
            self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    @staticmethod
    def piecewise_linear(x, hcp, ccp, base, hsl, csl):
        #  k1  \              / k2
        #       \            /
        # y0     \__________/
        #        cpL      cpR

        # Handle 3P models when use this function to predict.
        if np.isnan(hcp) and np.isnan(hsl):
            hcp = ccp
            hsl = 0
        if np.isnan(csl) and np.isnan(csl):
            ccp = hcp
            csl = 0

        conds = [x < hcp, (x >= hcp) & (x <= ccp), x > ccp]

        funcs = [lambda x: hsl * x + base - hsl * hcp,
                 lambda x: base,
                 lambda x: csl * x + base - csl * ccp]

        return np.piecewise(x, conds, funcs)

    def rmse(self):
        yp = self.piecewise_linear(self.temperature, *self.p)
        y = self.load
        return np.sqrt(np.mean([(i - j) ** 2 for i, j in zip(y, yp)]))

    def R_Squared(self, adjusted_r2_calc=False):
        x = self.temperature
        y = self.load
        residuals = y - self.piecewise_linear(x, *self.p)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        rsquared = 1 - (ss_res / ss_tot)
        r2_result = rsquared
        n = len(self.load)
        numP = len(np.nonzero(self.p))
        # If we need to calculate adjusted r-squared
        if (adjusted_r2_calc and n - numP - 1 != 0):
            r2_result = 1 - (1 - rsquared) * (n - 1) / (n - numP - 1)

        self.r2 = r2_result
        return (r2_result)

    def fit(self):
        try:
            self.p, self.e = optimize.curve_fit(
                self.piecewise_linear,
                self.temperature,
                self.load,
                bounds=([self.hcp_min, self.ccp_min, self.base_min, self.hsl_min, self.csl_min],
                        [self.hcp_max, self.ccp_max, self.base_max, self.hsl_max, self.csl_max]
                        )
            )
            # Model coefficients
            self.hcp, self.ccp, self.base, self.hsl, self.csl = self.p

            # Get p-value from t-stes for the model coefficients
            n = len(self.temperature)
            self.p_base = stats.t.sf(self.base / np.sqrt(np.diag(self.e)[2] / n), df=n - 2)
            self.p_hsl = stats.t.cdf(self.hsl / np.sqrt(np.diag(self.e)[3] / n), df=n - 2)
            self.p_csl = stats.t.sf(self.csl / np.sqrt(np.diag(self.e)[4] / n), df=n - 2)
            # self.p_hcp = stats.t.cdf(abs(self.hcp - self.hcp_min) / np.sqrt(np.diag(self.e)[0]/n), df = n-2)
            # self.p_ccp = stats.t.cdf(abs(self.ccp - self.ccp_max) / np.sqrt(np.diag(self.e)[1]/n), df = n-2)
        except:
            self.has_fit = False

    def optimize_cp_limit(self, point):
        # Finds the optimum range for heating and cooling change-points bounds
        if point == "R":
            percentiles = [[i, i + 5] for i in np.arange(30, 90, 5)]
        else:
            percentiles = [[i, i + 5] for i in np.arange(10, 70, 5)]

        var = []
        # print('optimize cp limits')
        for per in percentiles:
            cp_limit_min, cp_limit_max = np.percentile(self.temperature, per)
            # print('Percentile = {}, search range: {:04.2f} - {:04.2f}, rmse = {:04.3f}'.format( per, cp_limit_min, cp_limit_max, self.rmse()))

            if point == "L":
                self.hcp_min = cp_limit_min  # Heating change-point minimum
                self.hcp_max = cp_limit_max  # Heating change-point maximum
            elif point == "R":
                self.ccp_min = cp_limit_min  # Cooling change-point minimum
                self.ccp_max = cp_limit_max  # Cooling change-point maximum

            self.fit()
            # print(self.p)
            r2 = self.R_Squared()
            # print('P value: base= {:04.3f}, left= {:04.3f}, right= {:04.3f}, R2 = {:04.2f} '.format(self.p_base, self.p_hsl, self.p_csl, r2 ))
            var.append(r2)

        optimum_limits = percentiles[var.index(max(var))]
        cp_limit_min, cp_limit_max = np.percentile(self.temperature, optimum_limits)
        if point == "L":
            self.hcp_min = cp_limit_min  # Heating change-point minimum
            self.hcp_max = cp_limit_max  # Heating change-point maximum
        elif point == "R":
            self.ccp_min = cp_limit_min  # Cooling change-point minimum
            self.ccp_max = cp_limit_max  # Cooling change-point maximum

        self.fit()
        return optimum_limits

    def fit_model(self, has_fit=False, threshold=0.1):
        ### Handle outliers (TBD)

        ### Fit change-point model
        self.fit()  # Initial guess
        self.p_init = self.p

        if (self.R_Squared() < threshold):
            print('No fit found')
            # Cannot accept model immediately. No meaningful correlation found.
            return (has_fit)
        else:
            self.optimize_slopes()
            self.optimize_cp_limit("L")
            self.optimize_cp_limit("R")
            self.optimize_slopes()
            self.inverse_cp()
            self.model_type()  # Get model type
            has_fit = True
            self.has_fit = has_fit
            # Save final model coefficients
            return (has_fit)

    def optimize_slopes(self):
        import math
        if (not (self.significant(self.p_hsl)) or math.isnan(self.p_hsl)):
            # print("--->Left slope is not significant! - P=",self.p_hsl)
            self.hsl_min = -10 ** -3
            self.hsl_insignificant = True

        if (not (self.significant(self.p_csl)) or math.isnan(self.p_csl)):
            # print("--->Right slope is not significant!- P=",self.p_csl)
            self.csl_max = 10 ** -3
            self.csl_insignificant = True

        self.fit()

        if self.hsl_insignificant:
            self.hcp = self.ccp
            self.hsl = 0

        if self.csl_insignificant:
            self.ccp = self.hcp
            self.csl = 0

        if self.hsl_insignificant and self.csl_insignificant:
            self.hcp = self.ccp = 0
            self.hsl = self.csl = 0

    def inverse_cp(self):
        if self.hcp > self.ccp and not self.csl_insignificant and not self.hsl_insignificant:
            cp = (self.hsl * self.hcp - self.csl * self.ccp) / (self.hsl - self.csl)
            self.hcp = self.ccp = cp

    def significant(self, x, threshold=0.05):
        sig = True if x < threshold else False
        return sig

    def model_type(self):
        # This function clean up the model parameters and assign model type string
        self.cp_txt = []
        self.R_Squared()
        if (self.hcp is None):
            self.model_type_str = 'No fit'
            self.has_fit = False
            self.coeff_validation = {'base': False, 'csl': False, 'ccp': False, 'hsl': False, 'hcp': False}
        elif (self.hcp == self.ccp and self.hsl == 0):
            self.model_type_str = "3P Cooling"
            self.cp_txt = "(" + str(round(self.ccp, 1)) + ", " + str(round(self.base, 1)) + ")"
            self.hcp = self.ccp
            self.hsl = 0
            # self.hsl, self.hcp = np.nan, np.nan

            self.coeff_validation = {'base': True, 'csl': True, 'ccp': True, 'hsl': False, 'hcp': False}
        elif (self.hcp == self.ccp and self.csl == 0):
            self.model_type_str = "3P Heating"
            self.cp_txt = "(" + str(round(self.hcp, 1)) + ", " + str(round(self.base, 1)) + ")"
            self.ccp = self.hcp
            self.csl = 0
            # self.csl, self.ccp = np.nan, np.nan

            self.coeff_validation = {'base': True, 'csl': False, 'ccp': False, 'hsl': True, 'hcp': True}
        elif (self.hcp == self.ccp and self.csl != 0 and self.hsl != 0):
            self.model_type_str = "4P"
            self.cp_txt = "(" + str(round(self.hcp, 1)) + ", " + str(round(self.base, 1)) + ")"
            self.coeff_validation = {'base': True, 'csl': True, 'ccp': True, 'hsl': True, 'hcp': True}
        elif (self.hcp != self.ccp and self.csl != 0 and self.hsl != 0):
            self.model_type_str = "5P"
            self.cp_txt.append("(" + str(round(self.hcp, 1)) + ", " + str(round(self.base, 1)) + ")")
            self.cp_txt.append("(" + str(round(self.ccp, 1)) + ", " + str(round(self.base, 1)) + ")")
            self.coeff_validation = {'base': True, 'csl': True, 'ccp': True, 'hsl': True, 'hcp': True}
        # Finally assign the model coefficients
        self.coeffs = {'base': self.base, 'csl': self.csl, 'ccp': self.ccp, 'hsl': abs(self.hsl), 'hcp': self.hcp}
        self.model_p = np.array([self.base, self.hcp, self.hsl, self.ccp, self.csl])

    def plot(self, ax):
        self.temp_min = self.temperature.min()
        self.temp_max = self.temperature.max()
        self.temp_plot = np.linspace(self.temp_min,self.temp_max,100)
        self.load_plot = self.piecewise_linear(self.temp_plot, *self.p)
        ax.scatter(self.temperature, self.load, color=self.colors[0], alpha=0.5, s=3)
        ax.plot(self.temp_plot, self.load_plot, color=self.colors[1], lw=3)
        ax.set_xlabel('Temperature, daily mean (degC)')
        ax.set_ylabel(self.y_name)
