import numpy as np
import pickle,json,os
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

def available_models():
    model_list = ['BaSTI','BaSTI_HST','PARSEC','MIST','SYCLIST','Dartmouth','YaPSI']
    model_sources = [
        'http://basti-iac.oa-abruzzo.inaf.it/',
        'http://basti-iac.oa-abruzzo.inaf.it/',
        'https://stev.oapd.inaf.it/PARSEC/',
        'https://waps.cfa.harvard.edu/MIST/',
        'https://www.unige.ch/sciences/astro/evolution/en/database/syclist',
        'https://rcweb.dartmouth.edu/stellar/',
        'http://www.astro.yale.edu/yapsi/'
    ]
    for model,source in zip(model_list,model_sources):
        print(model + 'Model (' + source + ')')

class AgeModel:
    def __init__(self,model_name,use_sklearn=True,use_tqdm=True):
        self.model_name = model_name
        self.use_sklearn = use_sklearn
        self.use_tqdm = use_tqdm
        if not has_tqdm and self.use_tqdm:
            self.use_tqdm = False
        self.space_col = np.linspace(-1,3,101)
        self.space_mag = np.linspace(-5,10,101)
        self.space_met = np.linspace(-3,0.5,11)
        self.domain = np.load('domain.npy',allow_pickle=True).item()[model_name]
        self.neural_networks = {}
        self.scalers = {}
        self.age_predictions = None
        self.load_neural_network(model_name)

    def __str__(self):
        return self.model_name + ' Age Model'

    def load_neural_network(self, model_name):
        if self.use_sklearn:
            if os.path.exists('models/NN_{}.sav'.format(model_name)):
                self.neural_networks['full'] = pickle.load(open('models/NN_{}.sav'.format(model_name), 'rb'))
                self.scalers['full'] = pickle.load(open('models/scaler_{}.sav'.format(model_name), 'rb'))
            if os.path.exists('models/NN_{}_BPRP.sav'.format(model_name)):
                self.neural_networks['reduced'] = pickle.load(open('models/NN_{}_BPRP.sav'.format(model_name), 'rb'))
                self.scalers['reduced'] = pickle.load(open('models/scaler_{}_BPRP.sav'.format(model_name), 'rb'))
        else:
            if os.path.exists('models/NN_{}.json'.format(model_name)):
                json_nn = json.load(open('models/NN_{}.json'.format(model_name), 'r'))
                self.neural_networks['full'] = {
                    'weights':json_nn['weights'],
                    'biases':json_nn['biases']
                }
                self.scalers['full'] = {
                    'means':json_nn['means'],
                    'stds':json_nn['stds']
                }
            if os.path.exists('models/NN_{}_BPRP.json'.format(model_name)):
                json_nn = json.load(open('models/NN_{}_BPRP.json'.format(model_name), 'r'))
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
                        n=1):
        if type(met) is not list:
            met = [met]
        if type(mag) is not list:
            mag = [mag]
        if type(col) is not list:
            col = [col]
        if type(emet) is not list and emet is not None:
            emet = [emet]
        if type(emag) is not list and emag is not None:
            emag = [emag]
        if type(ecol) is not list and ecol is not None:
            ecol = [ecol]
        if type(GBP) is not list and GBP is not None:
            GBP = [GBP]
        if type(GRP) is not list and GRP is not None:
            GRP = [GRP]
        if type(eGBP) is not list and eGBP is not None:
            eGBP = [eGBP]
        if type(eGRP) is not list and eGRP is not None:
            eGRP = [eGRP]
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

        self.age_predictions = np.zeros((X.shape[0],n))

        if self.use_tqdm and (n > 1 or X.shape[0] > 1):
            loop = tqdm(range(X.shape[0]))
        else:
            loop = range(X.shape[0])
        for i in loop:
            if n > 1:
                X_i = np.random.normal(X[i],X_errors[i],(n,X.shape[1]))
            else:
                X_i = X[i].reshape(1,-1)
            self.age_predictions[i] = self.propagate(X_i,neural_network,scaler)
        
        return self.age_predictions
    
    def check_domain(self,met,mag,col,emet=None,emag=None,ecol=None):
        if type(met) is not list:
            met = [met]
        if type(mag) is not list:
            mag = [mag]
        if type(col) is not list:
            col = [col]
        
        has_errors = emet != None and emag != None and ecol != None

        if has_errors:
            if type(emet) is not list:
                emet = [emet]
            if type(emag) is not list:
                emag = [emag]
            if type(ecol) is not list:
                ecol = [ecol]
        
        in_domain = np.zeros(len(met),dtype=bool)
        for i in range(len(met)):
            if has_errors:
                errors = [ecol[i],emag[i],emet[i]]
            else:
                errors = [0,0,0]
            min_i_col = np.maximum(np.digitize(col[i] - errors[0],self.space_col) - 1,0)
            max_i_col = np.minimum(np.digitize(col[i] + errors[0],self.space_col) - 1,99)
            min_i_mag = np.maximum(np.digitize(mag[i] - errors[1],self.space_mag) - 1,0)
            max_i_mag = np.minimum(np.digitize(mag[i] + errors[1],self.space_mag) - 1,99)
            min_i_met = np.maximum(np.digitize(met[i] - errors[2],self.space_met) - 1,0)
            max_i_met = np.minimum(np.digitize(met[i] + errors[2],self.space_met) - 1,9)

            in_domain[i] = bool(np.any(self.domain[min_i_col:max_i_col+1,min_i_mag:max_i_mag+1,min_i_met:max_i_met+1]) == 1)
        return in_domain
    
    def propagate(self,X,neural_network,scaler):
        if self.use_sklearn:
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
                for i in range(len(weights)):
                    a = self.dot(a,np.array(weights[i])) + biases[i]
                    a = self.relu(a)
                outputs.append(a[0])
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
        if self.age_predictions is None:
            raise ValueError('No age predictions have been made yet')
        return np.mean(self.age_predictions,axis=1)

    def median_ages(self):
        if self.age_predictions is None:
            raise ValueError('No age predictions have been made yet')
        return np.median(self.age_predictions,axis=1)

    def mode_ages(self):
        if self.age_predictions is None:
            raise ValueError('No age predictions have been made yet')
        #TODO: choose number of bins appropriately
        modes = []
        min_age = max(0,self.age_predictions.min())
        max_age = max(14,self.age_predictions.max())

        for i in range(len(self.age_predictions)):
            hist, bins = np.histogram(self.age_predictions[i],bins=100,range=(min_age,max_age))
            modes.append(bins[np.argmax(hist)] + (bins[1]-bins[0])/2)
        return np.array(modes)
    
    def std_ages(self):
        if self.age_predictions is None:
            raise ValueError('No age predictions have been made yet')
        return np.std(self.age_predictions,axis=1)
    
class BaSTIModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('BaSTI',use_sklearn)

class BaSTI_HSTModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('BaSTI_HST',use_sklearn)

class PARSECModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('PARSEC',use_sklearn)

class MISTModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('MIST',use_sklearn)

class SYCLISTModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('SYCLIST',use_sklearn)

class DartmouthModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('Dartmouth',use_sklearn)

class YaPSIModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('YaPSI',use_sklearn)

#TODO: add flavors to models (e.g. trained on cut CMD for optimal performance)