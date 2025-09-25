import numpy as np
import pickle,json,os,warnings,shutil,zipfile,urllib.request,json

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
    if input('Isochrone curves for plots do not exist. Download them ? (15.6Mb) [Y/n] (default: Y)') in ['n', 'N']:
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
    shutil.rmtree(os.path.join(NEST_DIR, 'star-age.github.io-main'))
    os.remove(tmp_zip)
    print("Isochrones downloaded and extracted.")

def get_isochrones(model):
    if model.model_name in loaded_isochrones:
        return loaded_isochrones[model.model_name]
    if os.path.exists(os.path.join(NEST_DIR, 'isochrones')) == False:
        download_isochrones()
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

        self.ages = np.zeros((X.shape[0],n))
        if store_samples:
            self.samples = np.zeros((X.shape[0],n,X.shape[1]))
        else:
            self.samples = None
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
            a = self.dot(a,weights[i]) + biases[i]
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
    
    def HR_diagram(self,met=None,mag=None,col=None,age=None,isochrone_met=0,age_type='median',check_domain=True,fig=None,ax=None,**kwargs):
        if has_matplotlib == False:
            raise ImportError('matplotlib is required for HR diagram plotting')
        
        if age_type not in ('median','mean','mode'):
            print('Age type not available, using median instead')
            age_type = 'median'
        
        if met is not None and mag is not None and col is not None:
            if age is None:
                result = self.ages_prediction(met,mag,col,store_samples=False)
                age = result[age_type]    
        elif met is None and mag is None and col is None and age is None:
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
        else:
            raise ValueError('Not a valid combination of arguments.')

        if check_domain and met is not None:
            in_domain = self.check_domain(met,mag,col)
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
            fig,ax = plt.subplots(figsize=(10,7))
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
            if type (isochrone_met) is not float and type(isochrone_met) is not int:
                raise ValueError('isochrone_met must be a float or int')
            isochrone_met = str(round(isochrone_met))
        if isochrone_met not in isochrones.keys():
            print('Metallicity not available for this model (-2,-1,0), using [M/H]=0 instead')
            isochrone_met = '0'
        isochrones = isochrones[isochrone_met]
        
        lines = []
        ages = []
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

        if mag is not None and col is not None:
            scatter = ax.scatter(
                col,
                mag,
                c=age,
                cmap='viridis',
                zorder=4,
                marker='*'
            )
            plt.colorbar(scatter,label='Age [Gyr]')

        annot = ax.annotate("", xy=(0,0), xytext=(0,15), textcoords="offset points",
                            ha='center',
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"),zorder=5)
        annot.set_visible(False)

        def update_annot(line, event, age):
            x, y = event.xdata, event.ydata
            annot.xy = (x, y)
            annot.set_text(f"Age: {age:.2f} Gyr")
            annot.get_bbox_patch().set_facecolor('white')
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                for line, age in zip(lines, ages):
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(line, event, age)
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
            plt.show()

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