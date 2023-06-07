import serial
import numpy as np
import pandas as pd
from scipy import signal

COM_PORT = 'COM7'
BAUD_RATE = 9600
ser = serial.Serial(COM_PORT, BAUD_RATE)

def digital_filter(EEG_wave, samp_freq=1000):
    sos_delta = signal.butter(5, 4, 'lowpass', fs=samp_freq, output='sos')
    output_delta = signal.sosfilt(sos_delta, EEG_wave)
    # prevent transient response from affecting the output wave range
    mean_delta, std_delta = np.mean(output_delta), np.std(output_delta)
    output_delta[np.absolute(output_delta - mean_delta) > 3 * std_delta] = mean_delta

    sos_alpha1 = signal.butter(5, [8, 10], 'bandpass', fs=samp_freq, output='sos')
    output_alpha1 = signal.sosfilt(sos_alpha1, EEG_wave)
    mean_alpha1, std_alpha1 = np.mean(output_alpha1), np.std(output_alpha1)
    output_alpha1[np.absolute(output_alpha1 - mean_alpha1) > 3 * std_alpha1] = mean_alpha1

    sos_alpha2 = signal.butter(5, [10, 12], 'bandpass', fs=samp_freq, output='sos')
    output_alpha2 = signal.sosfilt(sos_alpha2, EEG_wave)
    mean_alpha2, std_alpha2 = np.mean(output_alpha2), np.std(output_alpha2)
    output_alpha2[np.absolute(output_alpha2 - mean_alpha2) > 3 * std_alpha2] = mean_alpha2

    sos_beta1 = signal.butter(5, [12, 20], 'bandpass', fs=samp_freq, output='sos')
    output_beta1 = signal.sosfilt(sos_beta1, EEG_wave)
    mean_beta1, std_beta1 = np.mean(output_beta1), np.std(output_beta1)
    output_beta1[np.absolute(output_beta1 - mean_beta1) > 3 * std_beta1] = mean_beta1

    sos_beta2 = signal.butter(5, [20, 29], 'bandpass', fs=samp_freq, output='sos')
    output_beta2 = signal.sosfilt(sos_beta2, EEG_wave)
    mean_beta2, std_beta2 = np.mean(output_beta2), np.std(output_beta2)
    output_beta2[np.absolute(output_beta2 - mean_beta2) > 3 * std_beta2] = mean_beta2

    sos_gamma1 = signal.butter(5, [25, 50], 'bandpass', fs=samp_freq, output='sos')
    output_gamma1 = signal.sosfilt(sos_gamma1, EEG_wave)
    mean_gamma1, std_gamma1 = np.mean(output_gamma1), np.std(output_gamma1)
    output_gamma1[np.absolute(output_gamma1 - mean_gamma1) > 3 * std_gamma1] = mean_gamma1

    sos_gamma2 = signal.butter(5, [50, 100], 'bandpass', fs=samp_freq, output='sos')
    output_gamma2 = signal.sosfilt(sos_gamma2, EEG_wave)
    mean_gamma2, std_gamma2 = np.mean(output_gamma2), np.std(output_gamma2)
    output_gamma2[np.absolute(output_gamma2 - mean_gamma2) > 3 * std_gamma2] = mean_gamma2

    sos_theta = signal.butter(5, [4, 7], 'bandpass', fs=samp_freq, output='sos')
    output_theta = signal.sosfilt(sos_theta, EEG_wave)
    mean_theta, std_theta = np.mean(output_theta), np.std(output_theta)
    output_theta[np.absolute(output_theta - mean_theta) > 3 * std_theta] = mean_theta
    
    return {'Alpha1': output_alpha1, 
            'Alpha2': output_alpha2, 
            'Beta1': output_beta1, 
            'Beta2': output_beta2, 
            'Gamma1': output_gamma1, 
            'Gamma2': output_gamma2, 
            'Theta': output_theta, 
            'Delta': output_delta}

def main():
    EEG_wave = []

    try:
        while True:
            while ser.in_waiting:
                data_raw = ser.readline()
                data = data_raw.decode()    # convert to utf-8
                
                # print(data)
                data_str = data.replace('\n', '')
                data_str = data_str.replace('\r', '')
                data_str = data_str.replace(' ', '')
                if not data_str:
                    continue
                # print(data_str)
                value = int(data_str.split('.')[0])
                EEG_wave.append(value)
                
                num_point = len(EEG_wave)
                print(f'{num_point}, {value}')
                
                if num_point >= 2000:
                    # del EEG_wave[0]
                    
                    EEG_np = np.array(EEG_wave)
                    
                    wave_dict = digital_filter(EEG_np)
                    
                    # print(wave_dict['Alpha1'][500:550])
                    
                    df = pd.DataFrame(wave_dict)
                    df.to_csv('realtime_wave.csv')
                    EEG_wave.clear()

    except KeyboardInterrupt:
        ser.close()

if __name__ == '__main__':
    main()