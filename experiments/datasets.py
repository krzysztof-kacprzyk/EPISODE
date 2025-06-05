
from experiments.baselines.tacrolimus import TacrolimusBaseline, TacrolimusPK
from episode.data import Dataset
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.stats import beta


def get_toy_dataset(n_samples=100, n_measurements=20, noise_std=0.1):
    

    x_range = (0,1)
    x = np.linspace(x_range[0], x_range[1], n_samples)
    u_range = (0,1)
    u = np.linspace(u_range[0], u_range[1], n_samples)
   
    Y = []
    T = []
    for i in range(n_samples):
        t = np.linspace(0,1,n_measurements)
        y = x[i]*np.exp(u[i]*t)
        y += np.random.normal(0, noise_std, y.shape)
        T.append(t)
        Y.append(y.reshape(-1,1))
      
    V = pd.DataFrame(x, columns=['x'])
    V['u'] = u

    dataset_name = f"toy__n_samples={n_samples}__n_measurements={n_measurements}__noise_std={noise_std}"

    return Dataset(dataset_name, V, T, Y, t_range=(0,1), x0={0:0})

def get_SIR_dataset(n_samples=100, n_measurements=20, noise_std=0.1, normalize_time=True, seed=0):

    # SIR model differential equations.
    def sir_model(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def predict_sir_trajectories(S0, I0, R0, beta, gamma, timepoints):
        # Initial number of infected and recovered individuals, everyone else is susceptible to infection initially.
        y0 = S0, I0, R0
        # A grid of time points (in days)
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(sir_model, y0, timepoints, args=(beta, gamma))
        return ret
    
    beta = 0.3
    gamma = 0.1
    timepoints = np.linspace(0,100,n_measurements)

    S0_range = (0.6,1.0)
    I0_range = (0.01,0.1)
    R0_range = (0.0,1.0)

    # Create a dataset of initial conditions
    gen = np.random.default_rng(seed=seed)

    S0 = gen.uniform(S0_range[0], S0_range[1], n_samples)
    I0 = gen.uniform(I0_range[0], I0_range[1], n_samples)
    R0 = gen.uniform(R0_range[0], R0_range[1], n_samples)

    # Predict the trajectories
    ys = []
    for i in range(n_samples):
        y = predict_sir_trajectories(S0[i], I0[i], R0[i], beta, gamma, timepoints)
        # add noise
        y += gen.normal(0, noise_std, y.shape)
        ys.append(y)

    Y = np.stack(ys, axis=0)

    T = np.tile(timepoints, (n_samples,1))

    if normalize_time:
        T = T/100

    V = pd.DataFrame(np.stack([S0,I0,R0], axis=1))

    x0_dict = {0:0,1:1,2:2}

    t_range = (0,1)

    dataset_name = f"SIR__n_samples={n_samples}__n_measurements={n_measurements}__noise_std={noise_std}__normalize_time={normalize_time}"

    return Dataset(dataset_name, V, T, Y, t_range, x0=x0_dict)

def get_HIV_dataset(n_samples=100, n_measurements=20, noise_std=0.1, seed=0):

    # Define the HIV Viral Dynamics Model
    def hiv_model(y, t, lam, d, beta, delta, p, c):
        T, I, V = y
        dTdt = lam - d * T - beta * T * V
        dIdt = beta * T * V - delta * I
        dVdt = p * I - c * V
        return [dTdt, dIdt, dVdt]
    
    def simulate_hiv(lam, d, beta, delta, p, c, T0, I0, V0, timepoints):
        # Initial conditions
        y0 = [T0, I0, V0]
        # Integrate the ODE system
        sol = odeint(hiv_model, y0, timepoints, args=(lam, d, beta, delta, p, c))
        T = sol[:, 0]
        I = sol[:, 1]
        V = sol[:, 2]
        return T, I, V
    
    gen = np.random.default_rng(seed=seed)

    # Parameters
    lam = 10 * np.ones(n_samples)  # Constant rate of T cell production
    d = 0.1 * np.ones(n_samples)    # Death rate of T cells
    T0 = 100 * np.ones(n_samples)  # Initial number of T cells
    I0 = np.zeros(n_samples)  # Initial number of infected T cells

    V0 = gen.uniform(1,100, n_samples)  # Initial viral load
    delta = gen.uniform(0.2,0.5, n_samples)  # Death rate of infected T cells
    beta = gen.uniform(4e-4,8e-4,n_samples) # Rate of viral infection of T cells
    c = gen.uniform(2,3, n_samples)  # Rate of viral clearance
    p = gen.uniform(100,150, n_samples)  # Rate of viral production

    timepoints = np.linspace(0, 50, n_measurements)

    ys = []
    for i in range(n_samples):
        T, I, V = simulate_hiv(lam[i], d[i], beta[i], delta[i], p[i], c[i], T0[i], I0[i], V0[i], timepoints)
        
        # Normalize the time and the variables
        T = T / 50
        I = I / 50
        V = V / 2000

        # add noise
        T += gen.normal(0, noise_std, T.shape)
        I += gen.normal(0, noise_std, I.shape)
        V += gen.normal(0, noise_std, V.shape)
        ys.append(np.stack([T,I,V], axis=1))

    Y = np.stack(ys, axis=0)
    Time = np.tile(timepoints, (n_samples,1)) / 50

    new_V0 = Y[:,0,2]

    V_df = pd.DataFrame(np.stack([new_V0, beta, delta, p, c], axis=1), columns=['V0', 'beta', 'delta', 'p', 'c'])

    x0_dict = {0:2.0,1:0.0,2:0}

    t_range = (0,1)

    dataset_name = f"HIV__n_samples={n_samples}__n_measurements={n_measurements}__noise_std={noise_std}"

    return Dataset(dataset_name, V_df, Time, Y, t_range, x0=x0_dict)

def extract_data_from_one_dataframe(df):
    """
    This function extracts the data from one dataframe
    Args:
        df a pandas dataframe with columns ['id','x1','x2',...,'xM','t','y'] where the first M columns are the static features and the last two columns are the time and the observation
    Returns:
        X: a pandas dataframe of shape (D,M) where D is the number of samples and M is the number of static features
        ts: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
        ys: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
    """
    # TODO: Validate data

    ids = df['id'].unique()
    X = []
    ts = []
    ys = []
    for id in ids:
        df_id = df[df['id'] == id].copy()
        X.append(
            df_id.iloc[[0], 1:-2])
        # print(X)
        df_id.sort_values(by='t', inplace=True)
        ts.append(df_id['t'].values.reshape(-1))
        ys.append(df_id['y'].values.reshape(-1))
    X = pd.concat(X, axis=0, ignore_index=True)
    return X, ts, ys



def get_real_pharma_dataset(data_folder='data'):
    max_t = 12.5

    df = pd.read_csv(os.path.join(data_folder, "tacrolimus", "tac_pccp_mr4_250423.csv"))
    dosage_rows = df[df['DOSE'] != 0]
    assert dosage_rows['visit_id'].is_unique
    df.drop(columns=['DOSE', 'EVID','II', 'AGE'], inplace=True) # we drop age because many missing values. the other columns are not needed
    df.drop(index=dosage_rows.index, inplace=True) # drop dosage rows
    # Merge df with dosage rows on visit_id
    df = df.merge(dosage_rows[['visit_id', 'DOSE']], on='visit_id', how='left') # add dosage as a feature
    df.loc[df['TIME'] >= 168, 'TIME'] -= 168 # subtract 168 from time to get time since last dosage
    missing_24h = df[(df['TIME'] == 0) & (df['DV'] == 0)].index
    df.drop(index=missing_24h, inplace=True) # drop rows where DV is 0 and time is 0 - they correspond to missing 24h measurements

    dv_0 = df[df['TIME'] == 0][['visit_id', 'DV']]
    assert dv_0['visit_id'].is_unique
    df = df.merge(dv_0, on='visit_id', how='left', suffixes=('', '_0')) # add DV_0 as a feature

    more_than_t = df[df['TIME'] > max_t].index
    df.drop(index=more_than_t, inplace=True) # drop rows where time is greater than max_t

    df.dropna(inplace=True) # drop rows with missing values

    df = df[['visit_id'] + ['DOSE', 'DV_0', 'SEXE', 'POIDS', 'HT', 'HB', 'CREAT', 'CYP', 'FORMULATION'] + ['TIME', 'DV']]

    feature_names = ['DOSE', 'DV_0', 'SEX', 'WEIGHT', 'HT', 'HB', 'CREAT', 'CYP', 'FORM']

    df.columns = ['id'] + feature_names + ['t', 'y']

    X, ts, ys = extract_data_from_one_dataframe(df)

    global_t = np.array([0,0.33,0.67,1,1.5,2,3,4,6,9,12])

    new_ys = []
    new_ts = []

    for i in range(len(ts)):
        # interpolate the data
        t = ts[i]
        y = ys[i]
        y_interp = interp1d(t, y, kind='linear', fill_value='extrapolate')(global_t)
        new_ys.append(y_interp)
        new_ts.append(global_t)
    ts = np.stack(new_ts, axis=0)/12
    ys = np.stack(new_ys, axis=0)/20
    ys = np.expand_dims(ys,-1)
    X.iloc[:,1] = X.iloc[:,1]/20

    # Convert categorical features to integers
    X['SEX'] = X['SEX'].apply(np.int64).astype('category')
    X['CYP'] = X['SEX'].apply(np.int64).astype('category')
    X['FORM'] = X['SEX'].apply(np.int64).astype('category')

    return Dataset('tacrolimus-real', X, ts, ys, t_range=(0,1), x0={0:1})


def get_pk_dataset(n_samples, n_measurements, noise_std, seed=0):

    tac = TacrolimusPK()
    covariates = tac.get_random_covariates(n_samples, seed=seed)

    ys = []
    ts = []
    for i in range(n_samples):
        t = np.linspace(0,24,n_measurements)
        y = tac.predict(covariates[[i],:],t)
        ys.append(y)
        ts.append(t)

    ys = np.stack(ys, axis=0) / 20
    ts = np.stack(ts, axis=0) / 24

    # add noise
    gen = np.random.default_rng(seed)
    ys = ys + gen.normal(0, noise_std, size=ys.shape)

    covariates[:,1] = covariates[:,1] / 20

    df = pd.DataFrame(covariates, columns=tac.get_feature_names())

    df['SEX'] = df['SEX'].astype('category')
    df['CYP'] = df['CYP'].astype('category')
    df['FORM'] = df['FORM'].astype('category')

    dataset_name = f"PK__n_samples={n_samples}__n_measurements={n_measurements}__noise_std={noise_std}"

    return Dataset(dataset_name, df, ts, np.expand_dims(ys,-1), t_range=(0,1), x0={0:1})



def get_bike_dataset():
    
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    bike_sharing = fetch_ucirepo(id=275) 
    
    # data (as pandas dataframes) 
    X = bike_sharing.data.features.copy()
    y = bike_sharing.data.targets.copy()

    col_names = list(X.columns)
    col_names[0] = 'id'
    col_names[4] = 't'
    X.columns = col_names
    X['y'] = y['cnt']
    X['x0'] = pd.Series([0.0]*len(X))
    col_names = ['id', 'x0', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed','t', 'y']
    X = X[col_names]  

    # For each unique id, take the average of the columns and replace
    # the original data with the average values
    unique_ids = X['id'].unique()
    for id in unique_ids:
        mask = X['id'] == id
        x0 = X.loc[mask,'y'].iloc[0]
        if mask.sum() != 24:
            X.drop(X[mask].index, inplace=True)
            continue
        X.loc[mask, 'weathersit'] = X.loc[mask, 'weathersit'].mean().round()
        X.loc[mask, 'temp'] = X.loc[mask, 'temp'].mean()
        X.loc[mask, 'hum'] = X.loc[mask, 'hum'].mean()
        X.loc[mask, 'windspeed'] = X.loc[mask, 'windspeed'].mean()
        X.loc[mask, 'x0'] = x0
    X.drop(columns=['atemp', 'yr','holiday','weekday'], inplace=True)

    X, ts, ys = extract_data_from_one_dataframe(X)

    ts = np.stack(ts, axis=0) / 23
    ys = np.expand_dims(np.stack(ys, axis=0),-1) / 500
    X['x0'] = X['x0'] / 500

    X['mnth'] = X['mnth'].astype('float')
    X['workingday'] = X['workingday'].astype('category')
    X['weathersit'] = X['weathersit'].astype('category')

    dataset_name = "bike-sharing"

    return Dataset(dataset_name, X, ts, ys, t_range=(0,1), x0={0:0})


def _get_beta_dataset_base(name, n_samples, n_measurements, noise_std, seed=0, alpha_range=(1.5,3.0), beta_range=(1.5,3.0)):

    gen = np.random.default_rng(seed)
    alphas = gen.uniform(alpha_range[0],alpha_range[1],n_samples)
    betas = gen.uniform(beta_range[0],beta_range[1],n_samples)

    X = pd.DataFrame({'alpha':alphas, 'beta':betas})
    ts = [np.linspace(0,1,n_measurements) for i in range(len(X))]
    ys = [np.array([beta.pdf(t,alpha, betap) for t in np.linspace(0,1,n_measurements)]) for alpha, betap in zip(X['alpha'], X['beta'])]

    # add noise
    for i in range(len(ys)):
        ys[i] = np.expand_dims(ys[i],-1)
        ys[i] += gen.normal(0, noise_std, size=ys[i].shape)
    

    X = X[['alpha','beta']]

    dataset_name = f"{name}__n_samples={n_samples}__n_measurements={n_measurements}__noise_std={noise_std}"

    return Dataset(dataset_name, X, ts, ys, t_range=(0,1), x0={0:0.0})

def get_beta_dataset(n_samples, n_measurements, noise_std, seed=0):

    alpha_range = (1.5,3.0)
    beta_range = (1.5,3.0)
    n_samples_per_dim = int(np.sqrt(n_samples))
    n_samples = n_samples_per_dim**2
    name = "Beta"
    return _get_beta_dataset_base(name, n_samples, n_measurements, noise_std, seed, alpha_range, beta_range)


def get_beta_2_dataset(n_samples, n_measurements, noise_std, seed=0):

    name = "Beta_2"
    alpha_range = (3.0,5.0)
    beta_range = (3.0,5.0)

    return _get_beta_dataset_base(name, n_samples, n_measurements, noise_std, seed, alpha_range, beta_range)


def get_synthetic_tumor_dataset(n_samples, n_measurements, noise_std, seed=0):

    time_horizon = 1.0
    equation = 'wilkerson'
    dataset_class = SyntheticTumorDataset(
        n_samples=n_samples,
        n_time_steps=n_measurements,
        time_horizon=time_horizon,
        noise_std=noise_std,
        seed=seed,
        equation=equation
    )

    X, ts, ys = dataset_class.get_X_ts_ys()

    for i in range(len(ys)):
        ys[i] = np.expand_dims(ys[i],-1)

    dataset_name = f"synthetic-tumor__n_samples={n_samples}__n_measurements={n_measurements}__noise_std={noise_std}"
    return Dataset(dataset_name, X, ts, ys, t_range=(0,1), x0={0:2})

# Originally implemented in https://github.com/krzysztof-kacprzyk/TIMEVIEW
class SyntheticTumorDataset():

    def __init__(self, **args):
        self.args = args
        X, ts, ys = SyntheticTumorDataset.synthetic_tumor_data(
                        n_samples = self.args['n_samples'],
                        n_time_steps = self.args['n_time_steps'],
                        time_horizon = self.args['time_horizon'],
                        noise_std = self.args['noise_std'],
                        seed = self.args['seed'],
                        equation = self.args['equation'])
        if self.args['equation'] == "wilkerson":
            self.X = pd.DataFrame(X, columns=["age", "weight", "initial_tumor_volume", "dosage"])
        elif self.args['equation'] == "geng":
            self.X = pd.DataFrame(X, columns=["age", "weight", "initial_tumor_volume", "start_time", "dosage"])

        self.ts = ts
        self.ys = ys
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        if self.args['equation'] == "wilkerson":
            return {
                "age": (20, 80),
                "weight": (40, 100),
                "initial_tumor_volume": (0.1, 0.5),
                "dosage": (0.0, 1.0)
                }
        elif self.args['equation'] == "geng":
            return {
                "age": (20, 80),
                "weight": (40, 100),
                "initial_tumor_volume": (0.1, 0.5),
                "start_time": (0.0, 1.0),
                "dosage": (0.0, 1.0)
                }

    def get_feature_names(self):
        if self.args['equation'] == "wilkerson":
            return ["age", "weight", "initial_tumor_volume", "dosage"]
        elif self.args['equation'] == "geng":
            return ["age", "weight", "initial_tumor_volume", "start_time", "dosage"]



    def _tumor_volume(t, age, weight, initial_tumor_volume, start_time, dosage):
        """
        Computes the tumor volume at times t based on the tumor model under chemotherapy described in the paper.

        Args:
            t: numpy array of real numbers that are the times at which to compute the tumor volume
            age: a real number that is the age
            weight: a real number that is the weight
            initial_tumor_volume: a real number that is the initial tumor volume
            start_time: a real number that is the start time of chemotherapy
            dosage: a real number that is the chemotherapy dosage
        Returns:
            Vs: numpy array of real numbers that are the tumor volumes at times t
        """

        RHO_0=2.0

        K_0=1.0
        K_1=0.01

        BETA_0=50.0

        GAMMA_0=5.0

        V_min=0.001

        # Set the parameters of the tumor model
        rho=RHO_0 * (age / 20.0) ** 0.5
        K=K_0 + K_1 * (weight)
        beta=BETA_0 * (age/20.0) ** (-0.2)

        # Create chemotherapy function
        def C(t):
            return np.where(t < start_time, 0.0, dosage * np.exp(- GAMMA_0 * (t - start_time)))

        def dVdt(V, t):
            """
            This is the tumor model under chemotherapy.
            Args:
                V: a real number that is the tumor volume
                t: a real number that is the time
            Returns:
                dVdt: a real number that is the rate of change of the tumor volume
            """

            dVdt=rho * (V-V_min) * V * np.log(K / V) - beta * V * C(t)

            return dVdt

        # Integrate the tumor model
        V=odeint(dVdt, initial_tumor_volume, t)[:, 0]
        return V


    def _tumor_volume_2(t, age, weight, initial_tumor_volume, dosage):
        """
        Computes the tumor volume at times t based on the tumor model under chemotherapy described in the paper.

        Args:
            t: numpy array of real numbers that are the times at which to compute the tumor volume
            age: a real number that is the age
            weight: a real number that is the weight
            initial_tumor_volume: a real number that is the initial tumor volume
            start_time: a real number that is the start time of chemotherapy
            dosage: a real number that is the chemotherapy dosage
        Returns:
            Vs: numpy array of real numbers that are the tumor volumes at times t
        """

        G_0=2.0
        D_0=180.0
        PHI_0=10

        # Set the parameters of the tumor model
        # rho = RHO_0 * (age / 20.0) ** 0.5
        # K = K_0 + K_1 * (weight)
        # beta = BETA_0 * (age/20.0) ** (-0.2)

        g=G_0 * (age / 20.0) ** 0.5
        d=D_0 * dosage/weight
        # sigmoid function
        phi=1 / (1 + np.exp(-dosage*PHI_0))

        return initial_tumor_volume * (phi*np.exp(-d * t) + (1-phi)*np.exp(g * t))


    def synthetic_tumor_data(n_samples,  n_time_steps, time_horizon=1.0, noise_std=0.0, seed=0, equation="wilkerson"):
        """
        Creates synthetic tumor data based on the tumor model under chemotherapy described in the paper.

        We have five static features:
            1. age
            2. weight
            3. initial tumor volume
            4. start time of chemotherapy (only for Geng et al. model)
            5. chemotherapy dosage

        Args:
            n_samples: an integer that is the number of samples
            noise_std: a real number that is the standard deviation of the noise
            seed: an integer that is the random seed
        Returns:
            X: a numpy array of shape (n_samples, 4)
            ts: a list of n_samples 1D numpy arrays of shape (n_time_steps,)
            ys: a list of n_samples 1D numpy arrays of shape (n_time_steps,)
        """
        TUMOR_DATA_FEATURE_RANGES={
            "age": (20, 80),
            "weight": (40, 100),
            "initial_tumor_volume": (0.1, 0.5),
            "start_time": (0.0, 1.0),
            "dosage": (0.0, 1.0)
        }

        # Create the random number generator
        gen=np.random.default_rng(seed)

        # Sample age
        age=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['age'][0], TUMOR_DATA_FEATURE_RANGES['age'][1], size=n_samples)
        # Sample weight
        weight=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['weight'][0], TUMOR_DATA_FEATURE_RANGES['weight'][1], size=n_samples)
        # Sample initial tumor volume
        tumor_volume=gen.uniform(TUMOR_DATA_FEATURE_RANGES['initial_tumor_volume']
                                [0], TUMOR_DATA_FEATURE_RANGES['initial_tumor_volume'][1], size=n_samples)
        # Sample start time of chemotherapy
        start_time=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['start_time'][0], TUMOR_DATA_FEATURE_RANGES['start_time'][1], size=n_samples)
        # Sample chemotherapy dosage
        dosage=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['dosage'][0], TUMOR_DATA_FEATURE_RANGES['dosage'][1], size=n_samples)

        # Combine the static features into a single array
        if equation == "wilkerson":
            X=np.stack((age, weight, tumor_volume, dosage), axis=1)
        elif equation == "geng":
            X=np.stack((age, weight, tumor_volume, start_time, dosage), axis=1)

        # Create the time points
        ts=[np.linspace(0.0, time_horizon, n_time_steps)
            for i in range(n_samples)]

        # Create the tumor volumes
        ys=[]

        for i in range(n_samples):

            # Unpack the static features
            if equation == "wilkerson":
                age, weight, tumor_volume, dosage=X[i, :]
            elif equation == "geng":
                age, weight, tumor_volume, start_time, dosage=X[i, :]

            if equation == "wilkerson":
                ys.append(SyntheticTumorDataset._tumor_volume_2(
                    ts[i], age, weight, tumor_volume, dosage))
            elif equation == "geng":
                ys.append(SyntheticTumorDataset._tumor_volume(ts[i], age, weight,
                        tumor_volume, start_time, dosage))

            # Add noise to the tumor volumes
            ys[i] += gen.normal(0.0, noise_std, size=n_time_steps)

        return X, ts, ys


