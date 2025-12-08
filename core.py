import numpy as np
import pickle,json,os,warnings,shutil,zipfile,urllib.request,json
from ._version import __version__

try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False
try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.colors import ListedColormap
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    has_sklearn = True
except ImportError:
    has_sklearn = False

NEST_DIR = os.path.dirname(os.path.abspath(__file__))

loaded_isochrones = {}

def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

warnings.showwarning = custom_warning

def import_cmaps():
    cmaps = {}
    if has_matplotlib:
        cmaps_colors = dict(np.load(os.path.join(NEST_DIR, 'cmaps.npz')))
        for name, colors in cmaps_colors.items():
            cmaps[name] = ListedColormap(colors)
    return cmaps

cmaps = import_cmaps()

def sanitize_input(input):
    if input is not None and type(input) is not list:
        if hasattr(input,'ndim') and input.ndim == 0:
                input = input.item()
        if hasattr(input,'tolist'):
            input = input.tolist()
        else:
            input = [input]
    return input

def get_mode(arr,min_age=0,max_age=14,nbins=280):
    #TODO: choose number of bins appropriately
    hist, bins = np.histogram(arr,bins=nbins,range=(min_age,max_age))
    return bins[np.argmax(hist)] + (bins[1]-bins[0])/2

def download_isochrones(verbose=True):
    if input('Isochrone curves for plots do not exist. Download them ? (27.9Mb) [Y/n] (default: Y)') not in ['Y','y','']:
            return None
    iso_url = "https://github.com/star-age/star-age.github.io/archive/refs/heads/main.zip"
    iso_dir = os.path.join(NEST_DIR, 'isochrones')
    tmp_zip = os.path.join(NEST_DIR, 'isochrones_tmp.zip')

    if verbose:
        print("Downloading isochrones from GitHub...")
    if has_requests and has_tqdm:
        response = requests.get(iso_url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(tmp_zip, 'wb') as file, tqdm(
            desc="Downloading isochrones",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        urllib.request.urlretrieve(iso_url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
        members = [m for m in zip_ref.namelist() if m.startswith('star-age.github.io-main/isochrones/')]
        zip_ref.extractall(NEST_DIR, members)

    src = os.path.join(NEST_DIR, 'star-age.github.io-main', 'isochrones')
    if os.path.exists(iso_dir):
        shutil.rmtree(iso_dir)
    shutil.move(src, iso_dir)
    with open(os.path.join(iso_dir, 'version.txt'), 'w') as f:
        f.write(__version__)
    shutil.rmtree(os.path.join(NEST_DIR, 'star-age.github.io-main'))
    os.remove(tmp_zip)
    if verbose:
        print("Isochrones downloaded and extracted.")

def get_isochrones(model):
    if model.model_name in loaded_isochrones:
        return loaded_isochrones[model.model_name]
    if os.path.exists(os.path.join(NEST_DIR, 'isochrones')) == False:
        download_isochrones(verbose=model.verbose)
    
    matching_version = True
    if os.path.exists(os.path.join(NEST_DIR, 'isochrones/version.txt')):
        with open(os.path.join(NEST_DIR, 'isochrones/version.txt'),'r') as f:
            isochrone_version = f.read().strip()
            if isochrone_version != __version__:
                matching_version = False
    else:
        matching_version = False
    if not matching_version and model.verbose:
        warnings.warn('Isochrone version does not match NEST version. You may want to update the isochrones by running NEST.download_isochrones()')

    isochrone_path = os.path.join(NEST_DIR, 'isochrones', 'isochrones_' + model.model_name + '.json')
    if os.path.exists(isochrone_path):
        loaded_isochrones[model.model_name] = json.load(open(isochrone_path, 'r'))
        return loaded_isochrones[model.model_name]
    return None

def available_models():
    model_list = ['BaSTI','PARSEC','MIST','Geneva','Dartmouth','YaPSI','BaSTI_HST']
    model_sources = [
        'http://basti-iac.oa-abruzzo.inaf.it/',
        'https://stev.oapd.inaf.it/PARSEC/',
        'https://waps.cfa.harvard.edu/MIST/',
        'https://www.unige.ch/sciences/astro/evolution/en/database/syclist/',
        'https://rcweb.dartmouth.edu/stellar/',
        'http://www.astro.yale.edu/yapsi/',
        'http://basti-iac.oa-abruzzo.inaf.it/'
    ]
    for model,source in zip(model_list,model_sources):
        print(model + 'Model (' + source + ')')

class PopulationAge:
    __array_priority__ = 1000
    def __init__(self,age,age_error):
        self.age50 = age
        self.age16 = age_error[0]
        self.age84 = age_error[1]
        self._data = [self.age50,self.age16,self.age84]

    def __getitem__(self, idx):
        return self._data[idx]
    
    def __radd__(self, other):
        if isinstance(other, list):
            return other + list(self)
        return NotImplemented

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self._data, dtype=dtype)
        if copy is False:
            return arr
        return arr.copy()
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        arrays = [np.asarray(i) if isinstance(i, PopulationAge) else i
                for i in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)
        return result

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, PopulationAge) for t in types):
            return NotImplemented
        arrays = [np.asarray(arg) for arg in args]
        return func(*arrays, **kwargs)

    def __setitem__(self, idx, value):
        self._data[idx] = value
        self.age50, self.age16, self.age84 = self._data

    def __len__(self):
        return 3

    def __iter__(self):
        yield from self._data

    def __contains__(self, x):
        return x in self._data

    def __repr__(self):
        return self.__str__()

    def _repr_latex_(self):
        return self.__str__()

    def __str__(self):
        up = self.age84 - self.age50
        down = self.age50 - self.age16
        down_s = f"-{down:.2f}"
        up_s = f"+{up:.2f}"
        return f"$\\tau={self.age50:.2f}_{{{down_s}}}^{{{up_s}}}\\,\\mathrm{{Gyr}}$"


class AgeModel:
    def __init__(self,model_name,use_sklearn=True,use_tqdm=True,photometric_type=None,verbose=True):
        self.model_name = model_name
        self.use_sklearn = use_sklearn and has_sklearn
        self.use_tqdm = use_tqdm and has_tqdm
        self.verbose = verbose
        if photometric_type == None:
            photometric_type = 'Gaia'
        self.photometric_type = photometric_type
        domain_path = os.path.join(NEST_DIR, 'domain.pkl')
        domain = pickle.load(open(domain_path, 'rb'))
        if model_name in domain:
            self.domain = domain[model_name]
            self.space_col = np.array(self.domain['spaces'][0])
            self.space_mag = np.array(self.domain['spaces'][1])
            self.space_met = np.array(self.domain['spaces'][2])
            self.domain = self.domain['grid']
            self.domain = np.array(self.domain,dtype=bool)
        else:
            self.domain = None
            self.space_col = None
            self.space_mag = None
            self.space_met = None
        self.neural_networks = {}
        self.scalers = {}
        self.samples = None
        self.ages = None
        self.medians = None
        self.means = None
        self.modes = None
        self.stds = None
        self.pop_age = None
        self.pop_age_error = None
        self.load_neural_network(self.model_name)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _str = self.model_name + ' Age Model'
        if self.photometric_type == 'Gaia':
            _str += ', Gaia photometry: mag=MG, col=(GBP-GRP).'
        elif self.photometric_type == 'HST':
            _str += ', HST photometry: mag=F814W, col=(F606W-F814W).'
        return _str
    
    def load_mlp_and_scaler(self, filename):
        with open(filename + '.mlp', "r") as f:
            payload = json.load(f)
        nn_payload = payload['mlp']
        meta = nn_payload["meta"]
        mlp = MLPRegressor(hidden_layer_sizes=tuple(meta["hidden_layer_sizes"]),
                        activation=meta["activation"],
                        solver=meta["solver"],
                        alpha=meta.get("alpha", 0.0001))

        mlp.coefs_ = [np.array(c, dtype=float) for c in nn_payload["coefs"]]
        mlp.intercepts_ = [np.array(b, dtype=float) for b in nn_payload["intercepts"]]

        mlp.n_layers_ = len(mlp.coefs_) + 1
        mlp.n_outputs_ = mlp.coefs_[-1].shape[1]
        mlp.out_activation_ = meta.get("out_activation_", "identity")
        mlp.n_features_in_ = mlp.coefs_[0].shape[0]
        mlp.n_iter_ = 1

        scaler_payload = payload['scaler']

        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_payload["mean_"], dtype=float)
        scaler.scale_ = np.array(scaler_payload["scale_"], dtype=float)
        scaler.n_features_in_ = len(scaler.mean_)

        return {'NN':mlp, 'Scaler':scaler}

    def load_neural_network(self, model_name):
        if self.use_sklearn:
            model_path_full = os.path.join(NEST_DIR, 'models', f'{model_name}')
            model_path_reduced = os.path.join(NEST_DIR, 'models', f'{model_name}_BPRP')
            if os.path.exists(model_path_full + '.mlp'):
                nn = self.load_mlp_and_scaler(os.path.join(NEST_DIR, 'models', model_name))
                self.neural_networks['full'] = nn['NN']
                self.scalers['full'] = nn['Scaler']
            if os.path.exists(model_path_reduced + '.mlp'):
                nn = self.load_mlp_and_scaler(os.path.join(NEST_DIR, 'models', model_name + '_BPRP'))
                self.neural_networks['reduced'] = nn['NN']
                self.scalers['reduced'] = nn['Scaler']
        else:
            model_path_full = os.path.join(NEST_DIR, 'models', f'{model_name}.mlp')
            model_path_reduced = os.path.join(NEST_DIR, 'models', f'{model_name}_BPRP.mlp')
            if os.path.exists(model_path_full):
                nn_json = json.load(open(model_path_full, 'r'))
                self.neural_networks['full'] = {
                    'weights':nn_json['mlp']['coefs'],
                    'biases':nn_json['mlp']['intercepts']
                }
                self.scalers['full'] = {
                    'means':nn_json['scaler']['mean_'],
                    'stds':nn_json['scaler']['scale_']
                }
            if os.path.exists(model_path_reduced):
                nn_json = json.load(open(model_path_reduced, 'r'))
                self.neural_networks['reduced'] = {
                    'weights':nn_json['mlp']['coefs'],
                    'biases':nn_json['mlp']['intercepts']
                }
                self.scalers['reduced'] = {
                    'means':nn_json['scaler']['mean_'],
                    'stds':nn_json['scaler']['scale_']
                }

    def ages_prediction(self,
                        met,mag,col,
                        emet=None,emag=None,ecol=None,
                        GBP=None,GRP=None,
                        eGBP=None,eGRP=None,
                        n=1,
                        store_samples=True,
                        min_age=0,max_age=14):
        
        met = sanitize_input(met)
        mag = sanitize_input(mag)
        col = sanitize_input(col)
        emet = sanitize_input(emet)
        emag = sanitize_input(emag)
        ecol = sanitize_input(ecol)
        GBP = sanitize_input(GBP)
        GRP = sanitize_input(GRP)
        eGBP = sanitize_input(eGBP)
        eGRP = sanitize_input(eGRP)

        if np.isnan(met).any():
            raise ValueError('Metallicity (met) cannot contain NaN values')
        if np.isnan(mag).any():
            raise ValueError('Absolute Magnitude (mag) cannot contain NaN values')
        if np.isnan(col).any():
            raise ValueError('Color (col) cannot contain NaN values')
        if emet is not None and np.isnan(emet).any():
            raise ValueError('Metallicity error (emet) cannot contain NaN values')
        if emag is not None and np.isnan(emag).any():
            raise ValueError('Absolute Magnitude error (emag) cannot contain NaN values')
        if ecol is not None and np.isnan(ecol).any():
            raise ValueError('Color error (ecol) cannot contain NaN values')
        if GBP is not None and np.isnan(GBP).any():
            raise ValueError('Gaia GBP magnitude (GBP) cannot contain NaN values')
        if GRP is not None and np.isnan(GRP).any():
            raise ValueError('Gaia GRP magnitude (GRP) cannot contain NaN values')
        if eGBP is not None and np.isnan(eGBP).any():
            raise ValueError('Gaia GBP magnitude error (eGBP) cannot contain NaN values')
        if eGRP is not None and np.isnan(eGRP).any():
            raise ValueError('Gaia GRP magnitude error (eGRP) cannot contain NaN values')

        if store_samples and n*len(met) > 1e6 and self.verbose:
            warnings.warn('Storing samples for {} stars with {} samples for each will take a lot of memory. Consider setting store_samples=False to only store mean,median,mode and std of individual age distributions.'.format(len(met),n))

        inputs = [input for input in [met,mag,col,emet,emag,ecol,GBP,GRP,eGBP,eGRP] if input is not None]
        if len(set(map(len,inputs))) != 1:
            raise ValueError('All input arrays must have the same length')

        is_reduced = True
        has_errors = False
        if GBP is not None and GRP is not None:
            is_reduced = False
            if emet is not None and emag is not None and ecol is not None and eGBP is not None and eGRP is not None:
                has_errors = True
        else:
            if emet is not None and emag is not None and ecol is not None:
                has_errors = True
        
        if n > 1 and not has_errors:
            raise ValueError('For more than one sample, errors must be provided')
        
        if is_reduced:
            X = np.array([met, mag, col])
            X_errors = np.array([emet, emag, ecol])
        else:
            X = np.array([met, mag, GBP, GRP, col])
            X_errors = np.array([emet, emag, eGBP, eGRP, ecol])

        X = X.T
        X_errors = X_errors.T

        if X.shape[1] == 3:
            if self.neural_networks.get('reduced') is None:
                raise ValueError('Reduced neural network not available for this model')
            scaler = self.scalers['reduced']
            neural_network = self.neural_networks['reduced']
        else:
            if self.neural_networks.get('full') is None:
                raise ValueError('Full neural network not available for this model')
            scaler = self.scalers['full']
            neural_network = self.neural_networks['full']

        if store_samples:
            self.ages = np.zeros((X.shape[0],n))
            self.samples = np.zeros((X.shape[0],n,X.shape[1]))
        else:
            self.ages = None
            self.samples = np.zeros((X.shape[0],X.shape[1]))
            self.medians = np.zeros(X.shape[0])
            self.means = np.zeros(X.shape[0])
            self.modes = np.zeros(X.shape[0])
            self.stds = np.zeros(X.shape[0])

        if self.use_tqdm and (n > 1 or X.shape[0] > 1):
            loop = tqdm(range(X.shape[0]))
        else:
            loop = range(X.shape[0])
        for i in loop:
            if n > 1:
                X_i = np.random.normal(X[i],X_errors[i],(n,X.shape[1]))
                if store_samples:
                    self.samples[i] = X_i
            else:
                X_i = X[i].reshape(1,-1)
                if store_samples:
                    self.samples[i] = X_i
            
            ages = self.propagate(X_i,neural_network,scaler)
            if store_samples:
                self.ages[i] = ages
            else:
                median = np.median(ages)
                mean = np.mean(ages)
                mode = get_mode(ages,min_age,max_age)
                std = np.std(ages)
                self.medians[i] = median
                self.means[i] = mean
                self.modes[i] = mode
                self.stds[i] = std
                self.samples[i] = X[i]

        if store_samples:
            return self.ages
        else:
            return {'mean':self.means,'median':self.medians,'mode':self.modes,'std':self.stds}

    def check_domain(self,met,mag,col,emet=None,emag=None,ecol=None,use_tqdm=True):
        if self.domain is None:
            raise ValueError('No domain defined for this model')
        met = sanitize_input(met)
        mag = sanitize_input(mag)
        col = sanitize_input(col)
        emet = sanitize_input(emet)
        emag = sanitize_input(emag)
        ecol = sanitize_input(ecol)
        
        has_errors = emet != None and emag != None and ecol != None
        
        in_domain = np.zeros(len(met),dtype=bool)

        if self.use_tqdm and len(met) > 1 and use_tqdm:
            loop = tqdm(range(len(met)))
        else:
            loop = range(len(met))

        for i in loop:
            if has_errors:
                errors = [ecol[i],emag[i],emet[i]]
            else:
                errors = [0,0,0]
            min_i_col = np.maximum(np.digitize(col[i] - errors[0],self.space_col) - 1,0)
            max_i_col = np.minimum(np.digitize(col[i] + errors[0],self.space_col) - 1,self.space_col.size-2)
            min_i_mag = np.maximum(np.digitize(mag[i] - errors[1],self.space_mag) - 1,0)
            max_i_mag = np.minimum(np.digitize(mag[i] + errors[1],self.space_mag) - 1,self.space_mag.size-2)
            min_i_met = np.maximum(np.digitize(met[i] - errors[2],self.space_met) - 1,0)
            max_i_met = np.minimum(np.digitize(met[i] + errors[2],self.space_met) - 1,self.space_met.size-2)
            in_domain[i] = bool(np.any(self.domain[min_i_col:max_i_col+1,min_i_mag:max_i_mag+1,min_i_met:max_i_met+1]) == 1)
        
        return in_domain
    
    def population_age(self,nbins=280,min_age=0,max_age=14,check_domain=True,n_mc=100,use_tqdm=True,epsilon=None):
        if self.ages is None:
            raise ValueError('No samples have been stored yet. Make sure to run ages_prediction() with store_samples=True')
        
        if check_domain and self.samples is not None:
            in_domain = self.check_domain(
                np.median(self.samples[:,:,0],axis=1),
                np.median(self.samples[:,:,1],axis=1),
                np.median(self.samples[:,:,2],axis=1),
                use_tqdm=False
            )
            ages = self.ages[in_domain]
        else:
            ages = self.ages

        ages = np.array([age for age in ages if np.any(np.isnan(age)) == False])

        if len(ages) == 0:
            raise ValueError('No valid age samples available to compute population age (maybe all stars are out of domain ?)')
        
        if ages.shape[1] == 1:
            return np.median(ages)

        pdfs = []

        loop = range(n_mc)
        if self.use_tqdm and use_tqdm:
            loop = tqdm(loop)
        
        if epsilon is None:
            epsilon = np.clip(1/ages.shape[0],0.001,0.1)

        individual_pdfs = []
        for i in range(ages.shape[0]):
            age = ages[i]
            pdf, bins = np.histogram(age,bins=nbins,range=(min_age,max_age))
            pdf = pdf.astype(float)
            pdf += epsilon
            individual_pdfs.append(pdf)

        for i in loop:
            random_star_indices = np.random.choice(list(range(ages.shape[0])), size=ages.shape[0], replace=True)
            random_star_indices = random_star_indices.astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                arr = np.array(individual_pdfs)
                selected = arr[random_star_indices]
                global_pdf = np.prod(selected, axis=0)
            if not np.allclose(global_pdf, global_pdf[0]):
                pdfs.append(bins[np.argmax(global_pdf)] + (bins[1]-bins[0])/2)

        if len(pdfs) == 0:
            if self.verbose:
                warnings.warn('No valid population age could be computed from the samples')
            self.pop_age = np.nan
            self.pop_age_error = (np.nan,np.nan)
            return PopulationAge(np.nan,(np.nan,np.nan))

        population_age = np.median(pdfs)
        population_age_p16 = np.percentile(pdfs,16)
        population_age_p84 = np.percentile(pdfs,84)

        self.pop_age = population_age
        self.pop_age_error = (population_age_p16,population_age_p84)

        return PopulationAge(population_age, (population_age_p16, population_age_p84))
    
    def propagate(self,X,neural_network,scaler):
        if self.use_sklearn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                X = scaler.transform(X)
                return neural_network.predict(X)
        else:
            weights = neural_network['weights']
            biases = neural_network['biases']
            means = scaler['means']
            stds = scaler['stds']
            outputs = []
            for x in X:
                a = (x - means)/stds
                output = self.predict_nn(a,weights,biases)
                outputs.append(output)
            return np.array(outputs)

    def relu(self,x):
        return np.maximum(0,x)

    def dot(self,x,y):
        x_dot_y = 0
        for i in range(len(x)):
            x_dot_y += x[i]*y[i]
        return x_dot_y

    def predict_nn(self,X,weights,biases):
        a = X
        for i in range(len(weights)):
            a = self.dot(a,np.array(weights[i])) + np.array(biases[i])
            a = self.relu(a)
        return a[0]
    
    def mean_ages(self):
        if self.ages is None:
            raise ValueError('No age predictions have been made yet')
        self.means = np.mean(self.ages,axis=1)
        return self.means

    def median_ages(self):
        if self.ages is None:
            raise ValueError('No age predictions have been made yet')
        self.medians = np.median(self.ages,axis=1)
        return self.medians

    def mode_ages(self):
        if self.ages is None:
            raise ValueError('No age predictions have been made yet')
        modes = []
        min_age = max(0,self.ages.min())
        max_age = max(14,self.ages.max())

        for i in range(len(self.ages)):
            modes.append(get_mode(self.ages[i],min_age,max_age))
        self.modes = np.array(modes)
        return self.modes
    
    def std_ages(self):
        if self.ages is None:
            raise ValueError('No age predictions have been made yet')
        self.stds = np.std(self.ages,axis=1)
        return self.stds
    
    def HR_diagram(self,
                   isochrone_met=0,
                   plot_isochrone=True,plot_stars=True,plot_isochrone_uncertainty=True,
                   isochrone_ages=None,isochrone_ages_std=None,
                   age_type='median',
                   check_domain=True,
                   fig=None,ax=None,
                   star_cmap=cmaps['tempo_R'],
                   isochrone_cmap=cmaps['acton'],
                   loc_legend='best',
                   axis_fontsize=12,
                   legend_fontsize=10,
                   colorbar_fontsize=10,
                   selected_isochrone_linewidth=2,
                   uncertainty_alpha=0.25,
                   colorbar_lims=None,
                   colorbar_loc='right',
                   colorbar_pad=0.02,
                   colorbar=True,
                   n_mc=100,
                   epsilon=None,
                   **kwargs):
        if has_matplotlib == False:
            raise ImportError('matplotlib is required for HR diagram plotting')

        if type(star_cmap) is str:
            star_cmap = plt.get_cmap(star_cmap)
        if type(isochrone_cmap) is str:
            isochrone_cmap = plt.get_cmap(isochrone_cmap)

        if isochrone_ages is None:
            isochrone_ages = []
        if isochrone_ages_std is None:
            isochrone_ages_std = []
        
        if age_type not in ('median','mean','mode'):
            if self.verbose:
                print('Age type not available, using median instead')
            age_type = 'median'
        
        if plot_stars:
            if self.samples is not None and len(self.samples) > 0 and self.ages is not None and len(self.ages) == len(self.samples):
                if age_type == 'median':
                    age = self.median_ages()
                elif age_type == 'mean':
                    age = self.mean_ages()
                elif age_type == 'mode':
                    age = self.mode_ages()
                met = np.median(self.samples,axis=1)[:,0]
                mag = np.median(self.samples,axis=1)[:,1]
                col = np.median(self.samples,axis=1)[:,2]
            elif self.medians is not None and len(self.medians) > 0:
                if age_type == 'median':
                    age = self.medians
                elif age_type == 'mean':
                    age = self.means
                elif age_type == 'mode':
                    age = self.modes
                met = self.samples[:,0]
                mag = self.samples[:,1]
                col = self.samples[:,2]
            else:
                met = None
                mag = None
                col = None
                age = None
                plot_stars = False

            if check_domain and met is not None:
                in_domain = self.check_domain(met,mag,col,use_tqdm=False)
                age = age[in_domain]
                met = np.array(sanitize_input(met))
                mag = np.array(sanitize_input(mag))
                col = np.array(sanitize_input(col))
                met = met[in_domain]
                mag = mag[in_domain]
                col = col[in_domain]

        new_fig = False
        if fig is None or ax is None:
            new_fig = True
            if plot_stars == False:
                figsize = (8,7)
            else:
                figsize = (10,7)
            fig,ax = plt.subplots(figsize=figsize)
            if self.photometric_type == 'Gaia':
                ax.set_xlabel(r'$(G_{BP}-G_{RP})_0$ [mag]', fontsize=axis_fontsize)
                ax.set_ylabel(r'$M_G$ [mag]', fontsize=axis_fontsize)
            elif self.photometric_type == 'HST':
                ax.set_xlabel(r'$(F606W-F814W)$ [mag]', fontsize=axis_fontsize)
                ax.set_ylabel(r'$F606W$ [mag]', fontsize=axis_fontsize)
            ax.tick_params(labelsize=axis_fontsize,axis='both')
        isochrones = get_isochrones(self)
        if isochrones is None:
            raise ValueError('Isochrones not available for this model')
        if type(isochrone_met) is not str:
            if np.issubdtype(type(isochrone_met), np.number) is False:
                raise ValueError('isochrone_met must be a float or int')
            isochrone_met = str(round(isochrone_met))
        if isochrone_met not in isochrones.keys():
            if self.verbose:
                print('Metallicity not available for this model (-2,-1,0), using [M/H]=0 instead')
            isochrone_met = '0'
        isochrones = isochrones[isochrone_met]
        
        lines = []
        ages = []
        colors = []
        has_labels = False

        if 'c' not in kwargs and 'color' not in kwargs:
            kwargs['color'] = 'k'
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['linewidth'] = 0.5
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.25
        for i,isochrone in enumerate(isochrones):
            iso_age = isochrone['age']
            iso_mag = isochrone['MG']
            iso_col = isochrone['BP-RP']
            
            if i == 0 and 'label' in kwargs:
                label = kwargs.pop('label')
                line, = ax.plot(iso_col, iso_mag, label=label, **kwargs)
                has_labels = True
            else:
                line, = ax.plot(iso_col, iso_mag, **kwargs)
            lines.append(line)
            ages.append(iso_age)
            colors.append('k')

        vmin = None
        vmax = None
        cb = None

        if plot_stars and len(age) > 0:
            scatter = ax.scatter(
                col,
                mag,
                c=age,
                cmap=star_cmap,
                zorder=4,
                marker='*'
            )
            ages_arr = np.array(age)
            data_vmin = np.nanmin(ages_arr)
            data_vmax = np.nanmax(ages_arr)

            if data_vmin < 0 or data_vmax > 14:
                vmin, vmax = 0.0, 14.0
            else:
                vmin, vmax = float(data_vmin), float(data_vmax)

            if vmin >= vmax:
                vmin, vmax = max(0.0, vmin - 0.5), min(14.0, vmax + 0.5)
                if vmin >= vmax:
                    vmin, vmax = 0.0, 14.0
            
            if colorbar_lims is not None and (type(colorbar_lims) is tuple or type(colorbar_lims) is list) and len(colorbar_lims) == 2:
                vmin, vmax = colorbar_lims

            scatter.set_clim(vmin, vmax)
            if colorbar:
                cb = plt.colorbar(scatter,location=colorbar_loc,pad=colorbar_pad)
                cb.set_label('Age [Gyr]', fontsize=colorbar_fontsize)
                cb.ax.tick_params(labelsize=colorbar_fontsize)

        if isinstance(isochrone_ages, (float, int, np.floating, np.integer)):
            isochrone_ages = [float(isochrone_ages)]
        if isinstance(isochrone_ages_std, (float, int, np.floating, np.integer)):
            isochrone_ages_std = [float(isochrone_ages_std)]
        while len(isochrone_ages_std) < len(isochrone_ages):
            isochrone_ages_std += [None]
        if plot_isochrone:
            if self.ages is not None and self.ages.shape[1] > 1:
                if self.pop_age is None:
                    self.population_age(check_domain=check_domain,n_mc=n_mc,epsilon=epsilon,use_tqdm=False)
                if self.pop_age is not None:
                    isochrone_ages += [self.pop_age]
                    isochrone_ages_std += [self.pop_age_error]
            elif age is not None:
                isochrone_ages += [np.median(age)]
                isochrone_ages_std += [np.std(age)]

        if len(isochrone_ages) > 0:
            if vmin is None or vmax is None:
                if len(isochrone_ages) > 1:
                    vmin = min(isochrone_ages)
                    vmax = max(isochrone_ages)
                else:
                    vmin = max(0, isochrone_ages[0] - 0.5)
                    vmax = min(14, isochrone_ages[0] + 0.5)
            for i, (isochrone_age, isochrone_age_std) in enumerate(zip(isochrone_ages,isochrone_ages_std)):
                if np.isnan(isochrone_age):
                    continue
                isochrone = self.get_closest_isochrone(isochrone_age)
                color = isochrone_cmap((i+1)/(len(isochrone_ages)+1))

                if isochrone_age_std is not None and plot_isochrone_uncertainty:
                    if type(isochrone_age_std) is tuple or type(isochrone_age_std) is list:
                        isochrone_low = self.get_closest_isochrone(isochrone_age_std[0])
                        isochrone_upp = self.get_closest_isochrone(isochrone_age_std[1])
                        label = r'$\tau$' + f'={isochrone_age:.2f} ' + r'$^{+' + f'{isochrone_age_std[1]-isochrone_age:.2f}' + r'}_{-' + f'{isochrone_age - isochrone_age_std[0]:.2f}' + r'}$ Gyr'
                    else:
                        label = r'$\tau$' + f'={isochrone_age:.2f} ' + r'$\pm$' + f' {isochrone_age_std:.2f} Gyr'
                        isochrone_low = self.get_closest_isochrone(isochrone_age - isochrone_age_std)
                        isochrone_upp = self.get_closest_isochrone(isochrone_age + isochrone_age_std)
                    verts = np.concatenate([
                        np.column_stack([isochrone_low['BP-RP'], isochrone_low['MG']]),
                        np.column_stack([isochrone_upp['BP-RP'][::-1], isochrone_upp['MG'][::-1]])
                    ])
                    poly = Polygon(verts, closed=True, facecolor=color, alpha=uncertainty_alpha, edgecolor=None, zorder=6)
                    ax.add_patch(poly)
                else:
                    label = r'$\tau$' + f'={isochrone_age:.2f} Gyr'

                line, = ax.plot(
                    isochrone['BP-RP'],
                    isochrone['MG'],
                    color=color,
                    linewidth=selected_isochrone_linewidth,
                    zorder=5,
                    label=label
                )

                has_labels = True
                lines.append(line)
                ages.append(isochrone_age)
                colors.append(color)

        annot = ax.annotate("", xy=(0,0), xytext=(0,15), textcoords="offset points",
                            ha='center',
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"),zorder=5)
        annot.set_visible(False)
        if has_labels:
            ax.legend(loc=loc_legend, fontsize=legend_fontsize)

        def update_annot(line, event, age, color):
            x, y = event.xdata, event.ydata
            annot.xy = (x, y)
            annot.set_text(f"Age: {age:.2f} Gyr")
            annot.set_color(color)
            annot.get_bbox_patch().set_edgecolor(color)
            annot.get_bbox_patch().set_facecolor('white')
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                #First loop on non-black isochrones to prioritize them
                for line, age, color in zip(lines, ages, colors):
                    if color != 'k':
                        cont, ind = line.contains(event)
                        if cont:
                            update_annot(line, event, age, color)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                            return
                for line, age, color in zip(lines, ages, colors):
                    if color == 'k':
                        cont, ind = line.contains(event)
                        if cont:
                            update_annot(line, event, age, color)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                            return
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        if new_fig:
            ax.set_xlim(-1,3)
            ax.set_ylim(10,-5)
            plt.tight_layout()
            return fig,ax
    
    def get_closest_isochrone(self,age_target):
        isos_dic = get_isochrones(self)['0']
        isos_dic = sorted(isos_dic, key=lambda x: x['age'])

        if self.model_name == 'BaSTI':#Can interpolate isochrones for BaSTI model
            # Flatten all isochrones into lists
            ages_list = []
            bp_rp_list = []
            mg_list = []
            m_list = []
            for iso in isos_dic[::-1]:
                n = len(iso['MG'])
                ages_list.extend([iso['age']] * n)
                bp_rp_list.extend(iso['BP-RP'])
                mg_list.extend(iso['MG'])
                m_list.extend(iso['M'])

            ages_arr = np.array(ages_list)
            bp_rp_arr = np.array(bp_rp_list)
            mg_arr = np.array(mg_list)
            m_arr = np.array(m_list)

            unique_ages = np.unique(ages_arr)
            age0, age1 = 0, 0
            for i, age in enumerate(unique_ages[1:]):
                if age >= age_target:
                    age0 = unique_ages[i]
                    age1 = age
                    break

            if age_target < age0:
                age_target = age0
            if age_target > age1:
                age_target = age1

            # Get indices for iso0 and iso1
            idx0 = np.where(ages_arr == age0)[0]
            idx1 = np.where(ages_arr == age1)[0]

            # Interpolate between iso0 and iso1
            bp_rp0 = bp_rp_arr[idx0]
            mg0 = mg_arr[idx0]
            m0 = m_arr[idx0]

            bp_rp1 = bp_rp_arr[idx1]
            mg1 = mg_arr[idx1]
            m1 = m_arr[idx1]

            # Make sure arrays are the same length for interpolation
            n_points = min(len(bp_rp0), len(bp_rp1))
            bp_rp0 = bp_rp0[:n_points]
            mg0 = mg0[:n_points]
            m0 = m0[:n_points]
            bp_rp1 = bp_rp1[:n_points]
            mg1 = mg1[:n_points]
            m1 = m1[:n_points]

            frac = (age_target - age0) / (age1 - age0) if age1 != age0 else 0

            bp_rp_interp = bp_rp0 + (bp_rp1 - bp_rp0) * frac
            mg_interp = mg0 + (mg1 - mg0) * frac
            m_interp = (m0 + m1) * frac
            age_interp = np.full(n_points, age_target)

            # Special handling for age_target < 0.5
            if age_target < 0.5:
                for i in range(n_points):
                    if mg_interp[i] >= 1:
                        old_mg = mg_interp[i]
                        old_bp_rp = bp_rp_interp[i]

                        # Find closest MG in iso0 and iso1
                        idx0_closest = np.abs(mg0 - old_mg).argmin()
                        idx1_closest = np.abs(mg1 - old_mg).argmin()

                        iso0_mg = mg0[idx0_closest]
                        iso1_mg = mg1[idx1_closest]
                        iso0_bp_rp = bp_rp0[idx0_closest]
                        iso1_bp_rp = bp_rp1[idx1_closest]

                        factor = old_mg - 1
                        factor = min(max(factor, 0), 1)

                        mg_interp[i] = old_mg * (1 - factor) + (iso0_mg + (iso1_mg - iso0_mg) * frac) * factor
                        bp_rp_interp[i] = old_bp_rp * (1 - factor) + (iso0_bp_rp + (iso1_bp_rp - iso0_bp_rp) * frac) * factor
        else:
            # Find closest isochrone
            closest_iso = min(isos_dic, key=lambda x: abs(x['age'] - age_target))
            bp_rp_interp = np.array(closest_iso['BP-RP'])
            mg_interp = np.array(closest_iso['MG'])
            age_interp = np.full(len(bp_rp_interp), closest_iso['age'])
        # Return as a dict of arrays
        return {
            'age': age_interp,
            'BP-RP': bp_rp_interp,
            'MG': mg_interp
        }

class BaSTIModel(AgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__('BaSTI',*args, **kwargs)

class PARSECModel(AgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__('PARSEC',*args, **kwargs)

class MISTModel(AgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__('MIST',*args, **kwargs)

class GenevaModel(AgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__('Geneva',*args, **kwargs)

class DartmouthModel(AgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__('Dartmouth',*args, **kwargs)

class YaPSIModel(AgeModel):
    def __init__(self, *args, **kwargs):
        super().__init__('YaPSI',*args, **kwargs)

class BaSTI_HSTModel(AgeModel):
    def __init__(self,*args,photometric_type='HST',**kwargs):
        super().__init__('BaSTI_HST',*args, **kwargs, photometric_type=photometric_type)

class BaSTI_HST_enhancedModel(AgeModel):
    def __init__(self,*args,photometric_type='HST',**kwargs):
        super().__init__('BaSTI_HST_enhanced',*args, **kwargs, photometric_type=photometric_type)