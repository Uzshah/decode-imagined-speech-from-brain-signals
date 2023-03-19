import scipy as sp
import numpy as np
import pandas as pd
import scipy.io
import os
from scipy.signal import butter, lfilter
import spect_features as spect
from features import features


class utils:
    def get_features(self, mat, subset="Train"):
        if subset == "Train":
            sub = "epo_train"
        else:
            sub = "epo_validation"
        features = mat[0][sub][0][0][4].T
        for i in range(14):
            temp = mat[i+1][sub][0][0][4].T
            features = np.concatenate((features, temp), axis=0)
        return features
    
    def get_labels(self, mat, subset="Train"):
        if subset == "Train":
            sub = "epo_train"
        else:
            sub = "epo_validation"
        labels = mat[0][sub][0][0][5].T
        for i in range(14):
            temp = mat[i+1][sub][0][0][5].T
            labels = np.concatenate((labels, temp), axis=0)
        return labels
    
    def get_channel(self, mat, subset="Train"):
        if subset == "Train":
            sub = "epo_train"
        else:
            sub = "epo_validation"
        cha_name = mat[0][sub][0][0][0][0]
        cha_name = [f[0] for f in cha_name]
        cha_index = [f for f in range(1, len(cha_name)+1)]
        dic = dict(zip(cha_name, cha_index))
        return dic
    
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
        
    def extract_all(self, subset = "Train"):
        if os.path.exists(subset) & os.path.isfile(subset+"/Data_Sample01.mat"):
            path = [str(subset)+'/Data_Sample0'+str(x)+'.mat' for x in range(10) if x!=0]
            path += [str(subset)+'/Data_Sample1'+str(x)+'.mat' for x in range(6)]
            all_data = dict()
            for i, p in enumerate(path):
                all_data[i] = scipy.io.loadmat(p)
            
            features = self.get_features(all_data, subset)
            labels = self.get_labels(all_data, subset)
            channels = self.get_channel(all_data, subset)
            return features, labels, channels
        else:
            print(f"folder {subset} is not exists or Folder is empty")
            
          
    def statistical_features(self, raw_data, fs = 256, counter = 0):
        output = [] 
        for raw in raw_data:
            output.extend(features.hist(raw))
            output.append(features.interq_range(raw))
            output.append(features.kurtosis(raw))
            output.append(features.skewness(raw))
            output.append(features.calc_max(raw))
            output.append(features.calc_min(raw))
            output.append(features.calc_mean(raw))
            output.append(features.calc_median(raw))
            output.append(features.mean_abs_deviation(raw))
            output.append(features.median_abs_deviation(raw))
            output.append(features.calc_std(raw))
            output.append(features.calc_var(raw))
            output.extend(features.ecdf(raw))
            output.extend(features.ecdf_percentile(raw))
            output.extend(features.ecdf_percentile_count(raw))
        if counter%1000==0:
            print("Trail ", counter)
        return np.array(output)
    
    def temporal_features(self, raw_data, fs = 256, counter = 0):
        output = []
        for raw in raw_data:
            output.append(features.calc_centroid(raw, fs))
            output.append(features.mean_abs_diff(raw))
            output.append(features.mean_diff(raw))
            output.append(features.median_abs_diff(raw))
            output.append(features.median_diff(raw))
            output.append(features.distance(raw))
            output.append(features.sum_abs_diff(raw))
            output.append(features.slope(raw))
            output.append(features.auc(raw, fs))
            output.append(features.pk_pk_distance(raw))
            output.append(features.entropy(raw))
            output.append(features.neighbourhood_peaks(raw))
        if counter%1000==0:
            print("Trail ", counter)
        return np.array(output)
    
    def spectral_features(self, raw_data, fs = 256, counter = 0):
        output = []
        for raw in raw_data:
            output.append(spect.spectral_distance(raw, fs))
            output.append(spect.fundamental_frequency(raw, fs))
            output.append(spect.max_power_spectrum(raw, fs))
            output.append(spect.max_frequency(raw, fs))
            output.append(spect.median_frequency(raw, fs))
            output.append(spect.spectral_decrease(raw, fs))
            output.append(spect.spectral_kurtosis(raw, fs))
            output.append(spect.spectral_skewness(raw, fs))
            output.append(spect.spectral_spread(raw, fs))
            output.append(spect.spectral_slope(raw, fs))
            output.append(spect.spectral_variation(raw, fs))
            output.append(spect.spectral_positive_turning(raw, fs))
            output.append(spect.spectral_roll_on(raw, fs))
            output.append(spect.power_bandwidth(raw, fs))
            output.append(spect.fft_mean_coeff(raw, fs))
            output.append(spect.spectral_entropy(raw, fs))
        if counter%1000==0:
            print("Trail ", counter)
        return np.array(output)
    
    
    def get_ryhtm(self, data, rythm = "delta"):
        if rythm == 'delta':
            delta_features = np.asarray([self.bandpass(feature, 0.5, 4, 256) for feature in data])
            return delta_features
        elif rythm == 'theta':
            delta_features = np.asarray([self.bandpass(feature, 4, 8, 256) for feature in data])
            return delta_features
        elif rythm == 'alpha':
            delta_features = np.asarray([self.bandpass(feature, 8, 12, 256) for feature in data])
            return delta_features
        elif rythm == 'beta':
            delta_features = np.asarray([self.bandpass(feature, 12, 30, 256) for feature in data])
            return delta_features
        elif rythm == 'gamma':
            delta_features = np.asarray([self.bandpass(feature, 30, 100, 256) for feature in data])
            return delta_features
        else:
            print("Invalid rythm name or spelling mistake, please select one of the following \n[delta, theta, alpha, beta, gamma]")
            