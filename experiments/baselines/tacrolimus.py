from scipy.integrate import odeint
import numpy as np


# Adapted from https://github.com/krzysztof-kacprzyk/TIMEVIEW
class TacrolimusPK():

    def __init__(self):
        self.feature_names = ['DOSE', 'DV_0', 'SEX', 'WEIGHT', 'HT', 'HB', 'CREAT', 'CYP', 'FORM']

        self.params = {
            'TVCL': 21.2, # Typical value of clearance (L/h)
            'TVV1': 486, # Typical central volume of distribution (L)
            'TVQ': 79, # Typical intercomp clearance 1 (L/h)
            'TVV2': 271, # Typical peripheral volume of distribution 1 (L)
            'TVKTR': 3.34, # Typical transfert rate constant(1/h)
            'HTCL': -1.14, # effect of HT on cl
            'CYPCL': 2.00, # effect of CYP on cl
            'STKTR': 1.53, # effect of study on ktr
            'STV1': 0.29 # effect of study on v1
        }

    def get_feature_names(self):
        return self.feature_names
    
    def get_feature_ranges(self):
        return {
            'DOSE': (1, 10),
            'DV_0': (0, 20),
            'SEX': [0, 1],
            'WEIGHT': (45, 110),
            'HT': (20, 47),
            'HB': (6, 16),
            'CREAT': (60, 830),
            'CYP': [0, 1],
            'FORM': [0, 1]
        }


    def predict(self, covariates, time_points):

        assert covariates.shape[0] == 1, "TacrolimusPK only supports one patient at a time"
        assert covariates.shape[1] == 9, "TacrolimusPK requires 9 covariates"

        # Covariates
        cov = {name: covariates[0,i] for i, name in enumerate(self.feature_names)}

        # ODE Parameters
        CL = self.params['TVCL'] * ((cov['HT'] / 35) ** self.params['HTCL']) * (self.params['CYPCL']) ** cov['CYP']
        V1 = self.params['TVV1'] * ((self.params['STV1']) ** cov['FORM'])
        Q = self.params['TVQ']
        V2 = self.params['TVV2']
        KTR = self.params['TVKTR'] * ((self.params['STKTR']) ** cov['FORM'])

        # Initial conditions
        DEPOT_0 = cov['DOSE']
        TRANS1_0 = 0
        TRANS2_0 = 0
        TRANS3_0 = 0
        CENT_0 = cov['DV_0'] * (V1 / 1000)
        PERI_0 = V2/V1 * CENT_0

        # print(f"CL: {CL}, V1: {V1}, Q: {Q}, V2: {V2}, KTR: {KTR}, DEPOT_0: {DEPOT_0}, CENT_0: {CENT_0}, PERI_0: {PERI_0}")

        # ODE
        def tacrolimus_ode(y, t):
            dDEPOTdt = -KTR * y[0]
            dTRANS1dt = KTR * y[0] - KTR * y[1]
            dTRANS2dt = KTR * y[1] - KTR * y[2]
            dTRANS3dt = KTR * y[2] - KTR * y[3]
            dCENTdt = KTR*y[3] - ((CL + Q) * y[4]/V1) + (Q * y[5] / V2)
            dPERIdt = (Q * y[4]/V1) - (Q * y[5]/V2)
            return [dDEPOTdt, dTRANS1dt, dTRANS2dt, dTRANS3dt, dCENTdt, dPERIdt]
        
        # Solve ODE
        y = odeint(tacrolimus_ode, [DEPOT_0, TRANS1_0, TRANS2_0, TRANS3_0, CENT_0, PERI_0], time_points)

        y = y[:, 4] # Only return central compartment
        y = y * 1000 / V1 # Convert back to micro g/mL

        return y
    
    def predict_from_parameters(self, parameters, time_points):

        # Covariates

        # ODE Parameters
        CL = parameters[0]
        V1 = parameters[1]
        Q = parameters[2]
        V2 = parameters[3]
        KTR = parameters[4]

        # Initial conditions
        DEPOT_0 = parameters[5]
        TRANS1_0 = 0
        TRANS2_0 = 0
        TRANS3_0 = 0
        CENT_0 = parameters[6] * (V1 / 1000)
        PERI_0 = V2/V1 * CENT_0

        # ODE
        def tacrolimus_ode(y, t):
            dDEPOTdt = -KTR * y[0]
            dTRANS1dt = KTR * y[0] - KTR * y[1]
            dTRANS2dt = KTR * y[1] - KTR * y[2]
            dTRANS3dt = KTR * y[2] - KTR * y[3]
            dCENTdt = KTR*y[3] - ((CL + Q) * y[4]/V1) + (Q * y[5] / V2)
            dPERIdt = (Q * y[4]/V1) - (Q * y[5]/V2)
            return [dDEPOTdt, dTRANS1dt, dTRANS2dt, dTRANS3dt, dCENTdt, dPERIdt]
        
        # Solve ODE
        y = odeint(tacrolimus_ode, [DEPOT_0, TRANS1_0, TRANS2_0, TRANS3_0, CENT_0, PERI_0], time_points)

        y = y[:, 4] # Only return central compartment
        y = y * 1000 / V1 # Convert back to micro g/mL

        return y
    
    def get_random_dataset(self,n_patients, t_range, seed=0):

        covariates = self.get_random_covariates(n_patients, seed=seed)
        trajectories = self(covariates, t_range=t_range)
        
        return covariates, trajectories
    
    def get_random_covariates(self, n_patients, seed=0):

        # Random number generator
        rng = np.random.default_rng(seed)

        dataset = []

        for i in range(n_patients):

            covariates = []
            for feature in self.get_feature_names():
                f_range = self.get_feature_ranges()[feature]
                if isinstance(f_range, list):
                    # choose a random value from the list
                    value = rng.choice(f_range)
                else:
                    # choose a random value from the range
                    value = rng.uniform(f_range[0], f_range[1])
                covariates.append(value)
            covariates = np.array(covariates).reshape(1, -1)
            dataset.append(covariates)
        
        return np.concatenate(dataset, axis=0)
    
    def predict_from_X_T(self, X, T):
        model = TacrolimusPK()
        results = []
        X = X.copy()
        X = X.to_numpy()
        for i in range(len(X)):
            x = np.copy(X[[i],:])
            t = np.copy(T[i,:])
            x[:,1] *= 20
            t *= 12
            pred = model.predict(x, t)
            pred /= 20
            results.append(pred)
        return np.expand_dims(np.array(results),-1)


class TacrolimusBaseline():

    def __init__(self):
        pass

    def predict(self,X,T):
        model = TacrolimusPK()
        results = []
        X = X.copy()
        X = X.to_numpy()
        T = T.copy()
        T = np.stack(T,axis=0)
        for i in range(len(X)):
            x = np.copy(X[[i],:])
            t = np.copy(T[i,:])
            x[:,1] *= 20
            t *= 12
            pred = model.predict(x, t)
            pred /= 20
            results.append(pred)
        return np.expand_dims(np.array(results),-1)