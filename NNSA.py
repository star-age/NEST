import numpy as np
import pickle,json

def available_models():
    model_list = ['BaSTI','PARSEC','MIST','SYCLIST','Dartmouth','YaPSI']
    print('Available models: ' + ', '.join([model + 'Model' for model in model_list]))

class AgeModel:
    def __init__(self,model_name,use_sklearn=True):
        self.use_sklearn = use_sklearn
        self.neural_networks = {}
        self.scalers = {}
        self.model_name = ''
        self.age_predictions = None
        self.load_neural_network(model_name)

    def __str__(self):
        return self.model_name + ' Age Model'

    def load_neural_network(self, model_name):
        self.model_name = model_name
        if self.use_sklearn:
            self.neural_networks['full'] = pickle.load(open('models/NN_{}.sav'.format(model_name), 'rb'))
            self.scalers['full'] = pickle.load(open('models/scaler_{}.sav'.format(model_name), 'rb'))
            self.neural_networks['reduced'] = pickle.load(open('models/NN_{}_BPRP.sav'.format(model_name), 'rb'))
            self.scalers['reduced'] = pickle.load(open('models/scaler_{}_BPRP.sav'.format(model_name), 'rb'))
        else:
            json_nn = json.load(open('models/NN_{}.json'.format(model_name), 'r'))
            self.neural_networks['full'] = {
                'weights':json_nn['weights'],
                'biases':json_nn['biases']
            }
            self.scalers['full'] = {
                'means':json_nn['means'],
                'stds':json_nn['stds']
            }
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
                        MH,MG,BPRP,
                        eMH=None,eMG=None,eBPRP=None,
                        GBP=None,GRP=None,
                        eGBP=None,eGRP=None,
                        n=1):
        if type(MH) is not list:
            MH = [MH]
        if type(MG) is not list:
            MG = [MG]
        if type(BPRP) is not list:
            BPRP = [BPRP]
        if type(eMH) is not list and eMH is not None:
            eMH = [eMH]
        if type(eMG) is not list and eMG is not None:
            eMG = [eMG]
        if type(eBPRP) is not list and eBPRP is not None:
            eBPRP = [eBPRP]
        if type(GBP) is not list and GBP is not None:
            GBP = [GBP]
        if type(GRP) is not list and GRP is not None:
            GRP = [GRP]
        if type(eGBP) is not list and eGBP is not None:
            eGBP = [eGBP]
        if type(eGRP) is not list and eGRP is not None:
            eGRP = [eGRP]
        inputs = [input for input in [MH,MG,BPRP,eMH,eMG,eBPRP,GBP,GRP,eGBP,eGRP] if input is not None]
        if len(set(map(len,inputs))) != 1:
            raise ValueError('All input arrays must have the same length')

        is_reduced = True
        has_errors = False
        if GBP is not None and GRP is not None:
            is_reduced = False
            if eMH is not None and eMG is not None and eBPRP is not None and eGBP is not None and eGRP is not None:
                has_errors = True
        else:
            if eMH is not None and eMG is not None and eBPRP is not None:
                has_errors = True
        
        if n > 1 and not has_errors:
            raise ValueError('For more than one sample, errors must be provided')
        
        if is_reduced:
            X = np.array([MH, MG, BPRP])
            X_errors = np.array([eMH, eMG, eBPRP])
        else:
            X = np.array([MH, MG, GBP, GRP, BPRP])
            X_errors = np.array([eMH, eMG, eGBP, eGRP, eBPRP])

        X = X.T
        X_errors = X_errors.T

        if X.shape[1] == 3:
            scaler = self.scalers['reduced']
            neural_network = self.neural_networks['reduced']
        else:
            scaler = self.scalers['full']
            neural_network = self.neural_networks['full']

        self.age_predictions = np.zeros((X.shape[0],n))
        for i in range(X.shape[0]):
            if n > 1:
                X_i = np.random.normal(X[i],X_errors[i],(n,X.shape[1]))
            else:
                X_i = X[i].reshape(1,-1)
            self.age_predictions[i] = self.propagate(X_i,neural_network,scaler)
        
        return self.age_predictions
    
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
        for i in range(len(self.age_predictions)):
            hist, bins = np.histogram(self.age_predictions[i],bins=140,range=(0,14))
            modes.append(bins[np.argmax(hist)] + (bins[1]-bins[0])/2)
        return np.array(modes)
    
    def std_ages(self):
        if self.age_predictions is None:
            raise ValueError('No age predictions have been made yet')
        return np.std(self.age_predictions,axis=1)
    
class BaSTIModel(AgeModel):
    def __init__(self,use_sklearn=True):
        super().__init__('BaSTI',use_sklearn)

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

#TODO: add flavors to models (i.e. trained on cut CMD for optimal performance)
#TODO: add alpha shapes to get bounds of each model