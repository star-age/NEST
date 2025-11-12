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
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

NEST_DIR = os.path.dirname(os.path.abspath(__file__))

loaded_isochrones = {}

def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

warnings.showwarning = custom_warning

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

def download_isochrones():
    if input('Isochrone curves for plots do not exist. Download them ? (27.9Mb) [Y/n] (default: Y)') not in ['Y','y','']:
            return None
    iso_url = "https://github.com/star-age/star-age.github.io/archive/refs/heads/main.zip"
    iso_dir = os.path.join(NEST_DIR, 'isochrones')
    tmp_zip = os.path.join(NEST_DIR, 'isochrones_tmp.zip')

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
    print("Isochrones downloaded and extracted.")

def get_isochrones(model):
    if model.model_name in loaded_isochrones:
        return loaded_isochrones[model.model_name]
    if os.path.exists(os.path.join(NEST_DIR, 'isochrones')) == False:
        download_isochrones()
    
    matching_version = True
    if os.path.exists(os.path.join(NEST_DIR, 'isochrones/version.txt')):
        with open(os.path.join(NEST_DIR, 'isochrones/version.txt'),'r') as f:
            isochrone_version = f.read().strip()
            if isochrone_version != __version__:
                matching_version = False
    else:
        matching_version = False
    if not matching_version:
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

class AgeModel:
    def __init__(self,model_name,use_sklearn=True,use_tqdm=True,photometric_type=None):
        self.model_name = model_name
        self.use_sklearn = use_sklearn
        self.use_tqdm = use_tqdm
        if photometric_type == None:
            photometric_type = 'Gaia'
        self.photometric_type = photometric_type
        if not has_tqdm and self.use_tqdm:
            self.use_tqdm = False
        domain_path = os.path.join(NEST_DIR, 'domain.pkl')
        domain = pickle.load(open(domain_path, 'rb'))
        if model_name in domain:
            self.domain = domain[model_name]
            self.space_col = self.domain['spaces'][0]
            self.space_mag = self.domain['spaces'][1]
            self.space_met = self.domain['spaces'][2]
            self.domain = self.domain['grid']
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

    def load_neural_network(self, model_name):
        if self.use_sklearn:
            model_path_full = os.path.join(NEST_DIR, 'models', f'{model_name}.sav')
            model_path_reduced = os.path.join(NEST_DIR, 'models', f'{model_name}_BPRP.sav')
            if os.path.exists(model_path_full):
                nn = pickle.load(open(model_path_full, 'rb'))
                self.neural_networks['full'] = nn['NN']
                self.scalers['full'] = nn['Scaler']
            if os.path.exists(model_path_reduced):
                nn = pickle.load(open(model_path_reduced, 'rb'))
                self.neural_networks['reduced'] = nn['NN']
                self.scalers['reduced'] = nn['Scaler']
        else:
            model_path_full = os.path.join(NEST_DIR, 'models', f'NN_{model_name}.json')
            model_path_reduced = os.path.join(NEST_DIR, 'models', f'NN_{model_name}_BPRP.json')
            if os.path.exists(model_path_full):
                json_nn = json.load(open(model_path_full, 'r'))
                self.neural_networks['full'] = {
                    'weights':json_nn['weights'],
                    'biases':json_nn['biases']
                }
                self.scalers['full'] = {
                    'means':json_nn['means'],
                    'stds':json_nn['stds']
                }
            if os.path.exists(model_path_reduced):
                json_nn = json.load(open(model_path_reduced, 'r'))
                self.neural_networks['reduced'] = {
                    'weights':json_nn['weights'],
                    'biases':json_nn['biases']
                }
                self.scalers['reduced'] = {
                    'means':json_nn['means'],
                    'stds':json_nn['stds']
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

        if store_samples and n*len(met) > 1e6:
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

    def check_domain(self,met,mag,col,emet=None,emag=None,ecol=None):
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

        if self.use_tqdm and len(met) > 1:
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
    
    def population_age(self,nbins=280,min_age=0,max_age=14):
        if self.ages is None:
            raise ValueError('No samples have been stored yet. Make sure to run ages_prediction() with store_samples=True')
        global_pdf = np.ones(nbins)
        for i in range(self.ages.shape[0]):
            ages = self.ages[i]
            pdf, bins = np.histogram(ages,bins=nbins,range=(min_age,max_age))
            pdf = pdf.astype(float)
            pdf += .01
            global_pdf *= pdf
        return bins[np.argmax(global_pdf)] + (bins[1]-bins[0])/2
    
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
    
    def HR_diagram(self,isochrone_met=0,plot_isochrone=True,plot_stars=True,age_type='median',check_domain=True,fig=None,ax=None,**kwargs):
        if has_matplotlib == False:
            raise ImportError('matplotlib is required for HR diagram plotting')
        
        if age_type not in ('median','mean','mode'):
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
                in_domain = self.check_domain(met,mag,col)
                age = age[in_domain]
                met = np.array(sanitize_input(met))
                mag = np.array(sanitize_input(mag))
                col = np.array(sanitize_input(col))
                met = met[in_domain]
                mag = mag[in_domain]
                col = col[in_domain]

        if plot_stars == False and type(plot_isochrone) is bool and plot_isochrone:
            plot_isochrone = False

        new_fig = False
        if fig is None or ax is None:
            new_fig = True
            if plot_stars == False and plot_isochrone == False:
                figsize = (8,7)
            else:
                figsize = (10,7)
            fig,ax = plt.subplots(figsize=figsize)
            if self.photometric_type == 'Gaia':
                ax.set_xlabel(r'$(G_{BP}-G_{RP})_0$ [mag]')
                ax.set_ylabel(r'$M_G$ [mag]')
            elif self.photometric_type == 'HST':
                ax.set_xlabel(r'$(F606W-F814W)$ [mag]')
                ax.set_ylabel(r'$F606W$ [mag]')

        isochrones = get_isochrones(self)
        if isochrones is None:
            raise ValueError('Isochrones not available for this model')
        if type(isochrone_met) is not str:
            if np.issubdtype(type(isochrone_met), np.number) is False:
                raise ValueError('isochrone_met must be a float or int')
            isochrone_met = str(round(isochrone_met))
        if isochrone_met not in isochrones.keys():
            print('Metallicity not available for this model (-2,-1,0), using [M/H]=0 instead')
            isochrone_met = '0'
        isochrones = isochrones[isochrone_met]
        
        lines = []
        ages = []
        colors = []
        if 'c' not in kwargs and 'color' not in kwargs:
            kwargs['color'] = 'k'
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['linewidth'] = 0.5
        for i,isochrone in enumerate(isochrones):
            iso_age = isochrone['age']
            iso_mag = isochrone['MG']
            iso_col = isochrone['BP-RP']
            
            if i == 0 and 'label' in kwargs:
                label = kwargs.pop('label')
                line, = ax.plot(iso_col, iso_mag, label=label, **kwargs)
            else:
                line, = ax.plot(iso_col, iso_mag, **kwargs)
            lines.append(line)
            ages.append(iso_age)
            colors.append('k')

        isochrone_ages = []
        vmin = None
        vmax = None
        cb = None

        if plot_stars:
            scatter = ax.scatter(
                col,
                mag,
                c=age,
                cmap='viridis',
                zorder=4,
                marker='*'
            )
            ages_arr = np.array(age)
            data_vmin = np.nanmin(ages_arr)
            data_vmax = np.nanmax(ages_arr)
            # If any values fall outside [0,14], restrict the colorbar to [0,14]
            if data_vmin < 0 or data_vmax > 14:
                vmin, vmax = 0.0, 14.0
            else:
                vmin, vmax = float(data_vmin), float(data_vmax)
            # Ensure a valid range
            if vmin >= vmax:
                vmin, vmax = max(0.0, vmin - 0.5), min(14.0, vmax + 0.5)
                if vmin >= vmax:
                    vmin, vmax = 0.0, 14.0
            scatter.set_clim(vmin, vmax)
            cb = plt.colorbar(scatter, label='Age [Gyr]')
            if (type(plot_isochrone) is bool and plot_isochrone != False) or type(plot_isochrone) in (float,int,list):
                #if self.model_name != 'BaSTI':
                    #raise ValueError('Plotting individual isochrones is only available for the BaSTI model at the moment')
                if type(plot_isochrone) is bool and plot_isochrone:
                    if self.ages is not None and self.ages.shape[1] > 1:
                        isochrone_ages = [self.population_age()]
                    else:
                        isochrone_ages = [np.median(age)]
        if type(plot_isochrone) is float or type(plot_isochrone) is int:
            isochrone_ages = [float(plot_isochrone)]
        elif type(plot_isochrone) is list:
            isochrone_ages = plot_isochrone
        
        if len(isochrone_ages) > 0:
            if vmin is None or vmax is None:
                if len(isochrone_ages) > 1:
                    vmin = min(isochrone_ages)
                    vmax = max(isochrone_ages)
                else:
                    vmin = max(0, isochrone_ages[0] - 0.5)
                    vmax = min(14, isochrone_ages[0] + 0.5)
            for isochrone_age in isochrone_ages:
                isochrone = self.get_closest_isochrone(isochrone_age)
                norm = plt.Normalize(vmin, vmax)
                cmap = plt.get_cmap('viridis')
                color = cmap(norm(isochrone_age))
                line, = ax.plot(
                    isochrone['BP-RP'],
                    isochrone['MG'],
                    color=color,
                    linewidth=2,
                    zorder=5,
                    label=f'Isochrone (age={isochrone_age:.2f} Gyr)'
                )
                lines.append(line)
                ages.append(isochrone_age)
                colors.append(color)
            
            if cb is None:
                norm = plt.Normalize(vmin, vmax)
                mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
                mappable.set_array([vmin,vmax])
                cb = plt.colorbar(mappable, ax=ax, label='Age [Gyr]')

        annot = ax.annotate("", xy=(0,0), xytext=(0,15), textcoords="offset points",
                            ha='center',
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"),zorder=5)
        annot.set_visible(False)
        ax.legend()

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
            m_interp = np.array(closest_iso['M'])
            age_interp = np.full(len(bp_rp_interp), closest_iso['age'])
        # Return as a dict of arrays
        return {
            'age': age_interp,
            'BP-RP': bp_rp_interp,
            'MG': mg_interp,
            'M': m_interp
        }

class BaSTIModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True):
        super().__init__('BaSTI',use_sklearn,use_tqdm)

class PARSECModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True):
        super().__init__('PARSEC',use_sklearn,use_tqdm)

class MISTModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True):
        super().__init__('MIST',use_sklearn,use_tqdm)

class GenevaModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True):
        super().__init__('Geneva',use_sklearn,use_tqdm)

class DartmouthModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True):
        super().__init__('Dartmouth',use_sklearn,use_tqdm)

class YaPSIModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True):
        super().__init__('YaPSI',use_sklearn,use_tqdm)

class BaSTI_HSTModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True,photometric_type='HST'):
        super().__init__('BaSTI_HST',use_sklearn,use_tqdm,photometric_type)

class BaSTI_HST_enhancedModel(AgeModel):
    def __init__(self,use_sklearn=True,use_tqdm=True,photometric_type='HST'):
        super().__init__('BaSTI_HST_enhanced',use_sklearn,use_tqdm,photometric_type)