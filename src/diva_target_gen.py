import math
import sys
import os
from os import path
import shutil
import csv
import torch
import parselmouth
from parselmouth.praat import call
import glob
import numpy as np
import parselmouth
import statistics
import diva_targets

from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Helper scripts from: https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb

# This is the function to measure source acoustics using default male parameters.
import diva_utils


def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID)  # read the sound
    duration = call(sound, "Get total duration")  # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
    startTime = call(sound, "Get start time")
    endTime = call(sound, "Get end time")
    numTimeSteps = (endTime - startTime) / 0.05
    tmin = startTime
    tmax = endTime

    minF0 = call(pitch, "Get minimum", tmin, tmax, "Hertz", "Parabolic")
    maxF0 = call(pitch, "Get maximum", tmin, tmax, "Hertz", "Parabolic")
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return duration, meanF0, minF0, maxF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


# This function measures formants using Formant Position formula
def measureFormants(sound, wave_file, f0min, f0max):
    sound = parselmouth.Sound(sound)  # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    pt2 = call(pitch, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

    # calculate mean formants across pulses
    if len(f1_list) == 0:
        return None

    f1_mean = statistics.mean(f1_list)

    if len(f2_list) == 0:
        return None
    f2_mean = statistics.mean(f2_list)

    if len(f3_list) == 0:
        return None
    f3_mean = statistics.mean(f3_list)

    if len(f4_list) == 0:
        return None
    f4_mean = statistics.mean(f4_list)

    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    # Altered to return standard deviation -- Sean Kinahan
    f1_stdev = statistics.stdev(f1_list)
    f2_stdev = statistics.stdev(f2_list)
    f3_stdev = statistics.stdev(f3_list)
    f4_stdev = statistics.stdev(f4_list)

    return f1_mean, f2_mean, f3_mean, f4_mean, f1_stdev, f2_stdev, f3_stdev, f4_stdev


def measureFormantsAlternate(sound, wave_file, f0min, f0max):
    sound = parselmouth.Sound(sound)  # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

    midPt = math.floor(numPoints / 2)

    # calculate mean formants across pulses
    if len(f1_list) == 0:
        return None
    f1_mean = statistics.mean(f1_list)

    f1_start_interval = f1_list[0:19]
    f1_mid_interval = f1_list[midPt - 10:midPt + 10]
    f1_end_interval = f1_list[numPoints - 20:numPoints]

    f1_start_mean = statistics.mean(f1_start_interval)
    f1_start_stdev = statistics.stdev(f1_start_interval)

    f1_mid_mean = statistics.mean(f1_mid_interval)
    f1_mid_stdev = statistics.stdev(f1_mid_interval)

    f1_end_mean = statistics.mean(f1_end_interval)
    f1_end_stdev = statistics.mean(f1_end_interval)

    if len(f2_list) == 0:
        return None
    f2_mean = statistics.mean(f2_list)

    f2_start_interval = f2_list[0:19]
    f2_mid_interval = f2_list[midPt - 10:midPt + 10]
    f2_end_interval = f2_list[numPoints - 20:numPoints]

    f2_start_mean = statistics.mean(f2_start_interval)
    f2_start_stdev = statistics.stdev(f2_start_interval)

    f2_mid_mean = statistics.mean(f2_mid_interval)
    f2_mid_stdev = statistics.stdev(f2_mid_interval)

    f2_end_mean = statistics.mean(f2_end_interval)
    f2_end_stdev = statistics.stdev(f2_end_interval)

    if len(f3_list) == 0:
        return None
    f3_mean = statistics.mean(f3_list)

    f3_start_interval = f3_list[0:19]
    f3_mid_interval = f3_list[midPt - 10:midPt + 10]
    f3_end_interval = f3_list[numPoints - 20:numPoints]

    f3_start_mean = statistics.mean(f3_start_interval)
    f3_start_stdev = statistics.stdev(f3_start_interval)

    f3_mid_mean = statistics.mean(f3_mid_interval)
    f3_mid_stdev = statistics.stdev(f3_mid_interval)

    f3_end_mean = statistics.mean(f3_end_interval)
    f3_end_stdev = statistics.stdev(f3_end_interval)

    f1_means = (f1_start_mean, f1_mid_mean, f1_end_mean)
    f1_stdevs = (f1_start_stdev, f1_mid_stdev, f1_end_stdev)

    f2_means = (f2_start_mean, f2_mid_mean, f2_end_mean)
    f2_stdevs = (f2_start_stdev, f2_mid_stdev, f2_end_stdev)

    f3_means = (f3_start_mean, f3_mid_mean, f3_end_mean)
    f3_stdevs = (f3_start_stdev, f3_mid_stdev, f3_end_stdev)

    return f1_means, f1_stdevs, f2_means, f2_stdevs, f3_means, f3_stdevs


class Target_Gen():

    def __init__(self):
        self.diva_targets = diva_targets.diva_targets()

    def pull_timesteps(self, pm_obj, f_no, start, mid, end):
        start = pm_obj.get_value_at_time(f_no, start)
        mid = pm_obj.get_value_at_time(f_no, mid)
        end = pm_obj.get_value_at_time(f_no, end)
        return start, mid, end

    def extract(self, file):
        sound = parselmouth.Sound(file.path)
        pitch = sound.to_pitch()
        formant = sound.to_formant_burg()
        start_time = sound.trange[0]
        end_time = sound.trange[1]
        mid_time = end_time / 2
        # Extract F0 - F3

        results = measurePitch(file.path, 75.0, 600.0, "Hertz")
        duration = results[0]
        duration_ms = int(math.ceil(duration*1000))
        meanF0 = results[1]
        minF0 = results[2]
        maxF0 = results[3]
        stdevF0 = results[4]

        form_results = measureFormantsAlternate(sound, None, 75.0, 600.0)
        if form_results is None:
            return False
        f1_means, f1_stdevs, f2_means, f2_stdevs, f3_means, f3_stdevs = form_results

        (f1_start_mean, f1_mid_mean, f1_end_mean) = f1_means
        (f1_start_stdev, f1_mid_stdev, f1_end_stdev) = f1_stdevs

        (f2_start_mean, f2_mid_mean, f2_end_mean) = f2_means
        (f2_start_stdev, f2_mid_stdev, f2_end_stdev) = f2_stdevs

        (f3_start_mean, f3_mid_mean, f3_end_mean) = f3_means
        (f3_start_stdev, f3_mid_stdev, f3_end_stdev) = f3_stdevs

        span_width = 100
        width_factor = span_width / 2
        width_factor2 = span_width*2

        formant_pattern = "{0} {1} {2}"
        z_score_25 = -0.675
        z_score_75 = 0.675

        f0_25 = 0
        f0_75 = meanF0 + (z_score_75 * stdevF0)

        f1_start_25 = f1_start_mean + (z_score_25 * f1_start_stdev)
        f1_mid_25 = f1_mid_mean + (z_score_25 * f1_mid_stdev)
        f1_end_25 = f1_end_mean + (z_score_25 * f1_end_stdev)

        # f1_25 = formant_pattern.format(f1_start_25, f1_mid_25, f1_end_25)
        #f1_25 = min(f1_start_25, f1_mid_25, f1_end_25)
        f1_25 = f1_mid_25 - width_factor

        f1_start_75 = f1_start_mean + (z_score_75 * f1_start_stdev)
        f1_mid_75 = f1_mid_mean + (z_score_75 * f1_mid_stdev)
        f1_end_75 = f1_end_mean + (z_score_75 * f1_end_stdev)

        # f1_75 = formant_pattern.format(f1_start_75, f1_mid_75, f1_end_75)
        #f1_75 = max(f1_start_75, f1_mid_75, f1_end_75)
        f1_75 = f1_mid_75 + width_factor

        f2_start_25 = f2_start_mean + (z_score_25 * f2_start_stdev)
        f2_mid_25 = f2_mid_mean + (z_score_25 * f2_mid_stdev)
        f2_end_25 = f2_end_mean + (z_score_25 * f2_end_stdev)

        # f2_25 = formant_pattern.format(f2_start_25, f2_mid_25, f2_end_25)
        #f2_25 = min(f2_start_25, f2_mid_25, f2_end_25)
        f2_25 = f2_mid_25 - width_factor

        f2_start_75 = f2_start_mean + (z_score_75 * f2_start_stdev)
        f2_mid_75 = f2_mid_mean + (z_score_75 * f2_mid_stdev)
        f2_end_75 = f2_end_mean + (z_score_75 * f2_end_stdev)

        # f2_75 = formant_pattern.format(f2_start_75, f2_mid_75, f2_end_75)
        #f2_75 = max(f2_start_75, f2_mid_75, f2_end_75)
        f2_75 = f2_mid_75 + width_factor

        f3_start_25 = f3_start_mean + (z_score_25 * f3_start_stdev)
        f3_mid_25 = f3_mid_mean + (z_score_25 * f3_mid_stdev)
        f3_end_25 = f3_end_mean + (z_score_25 * f3_end_stdev)

        # f3_25 = formant_pattern.format(f3_start_25, f3_mid_25, f3_end_25)
        #f3_25 = min(f3_start_25, f3_mid_25, f3_end_25)
        f3_25 = f3_mid_25 - width_factor2

        f3_start_75 = f3_start_mean + (z_score_75 * f3_start_stdev)
        f3_mid_75 = f3_mid_mean + (z_score_75 * f3_mid_stdev)
        f3_end_75 = f3_end_mean + (z_score_75 * f3_end_stdev)

        # f3_75 = formant_pattern.format(f3_start_75, f3_mid_75, f3_end_75)
        #f3_75 = max(f3_start_75, f3_mid_75, f3_end_75)
        f3_75 = f3_mid_75 + width_factor2

        pressure_min = 0.75
        pressure_max = 1.0
        voicing_min = 0.75
        voicing_max = 1
        PA_pharyngeal_control = 0
        PA_pharyngeal_min = -1
        PA_pharyngeal_max = -0.25

        PA_uvular_control = 0
        PA_uvular_min = -1
        PA_uvular_max = -0.25

        PA_velar_control = 0
        PA_velar_min = -1
        PA_velar_max = -0.25

        PA_palatal_control = 0
        PA_palatal_min = -1
        PA_palatal_max = -0.25

        PA_alveolardental_control = 0
        PA_alveolardental_min = -1
        PA_alveolardental_max = -0.25

        PA_labial_control = 0
        PA_labial_min = -1
        PA_labial_max = -0.25
        wrapper = 0

        name = file.name

        #length = duration_ms
        length = 2000
        interpolation = 'spline'
        F0_control = 0
        F1_control = 0
        F2_control = 0
        F3_control = 0

        minF0 = 0

        ctr = 1
        diva_target_filename = "diva_000001.txt"
        while path.exists(diva_target_filename):
            file_root = os.path.splitext(diva_target_filename)[0]
            file_num_str = file_root[10:]
            file_root = file_root[:10]
            existing_num = int(file_num_str)
            new_num = existing_num + 1
            file_root = file_root + str(new_num)
            diva_target_filename = file_root + '.txt'

        # now the name should be free, write out the file
        with open(diva_target_filename, "w") as f:
            f.write("#name\n")
            f.write(name + "\n")
            f.write("#length\n")
            f.write(str(length) + "\n")
            f.write("#interpolation\nspline\n")
            f.write("#F0_control\n")
            f.write(str(F0_control) + "\n")
            f.write("#F0_min\n")
            f.write(str(minF0) + "\n")
            f.write("#F0_max\n")
            f.write(str(maxF0) + "\n")
            f.write("#F1_control\n")
            f.write(str(F1_control) + "\n")
            f.write("#F1_min\n")
            f.write(str(f1_25) + "\n")
            f.write("#F1_max\n")
            f.write(str(f1_75) + "\n")
            f.write("#F2_control\n")
            f.write(str(F2_control) + "\n")
            f.write("#F2_min\n")
            f.write(str(f2_25) + "\n")
            f.write("#F2_max\n")
            f.write(str(f2_75) + "\n")
            f.write("#F3_control\n")
            f.write(str(F3_control) + "\n")
            f.write("#F3_min\n")
            f.write(str(f3_25) + "\n")
            f.write("#F3_max\n")
            f.write(str(f3_75) + "\n")
            f.write("#pressure_control\n")
            f.write("0" + "\n")
            f.write("#pressure_min\n")
            f.write(str(pressure_min) + "\n")
            f.write("#pressure_max\n")
            f.write(str(pressure_max) + "\n")
            f.write("#voicing_control\n")
            f.write("0\n")
            f.write("#voicing_min\n")
            f.write(str(voicing_min) + "\n")
            f.write("#voicing_max\n")
            f.write(str(voicing_max) + "\n")
            f.write("#PA_pharyngeal_control\n")
            f.write(str(PA_pharyngeal_control) + "\n")
            f.write("#PA_pharyngeal_min\n")
            f.write(str(PA_pharyngeal_min) + "\n")
            f.write("#PA_pharyngeal_max\n")
            f.write(str(PA_pharyngeal_max) + "\n")
            f.write("#PA_uvular_control\n")
            f.write(str(PA_uvular_control) + "\n")
            f.write("#PA_uvular_min\n")
            f.write(str(PA_uvular_min) + "\n")
            f.write("#PA_uvular_max\n")
            f.write(str(PA_uvular_max) + "\n")
            f.write("#PA_velar_control\n")
            f.write(str(PA_velar_control) + "\n")
            f.write("#PA_velar_min\n")
            f.write(str(PA_velar_min) + "\n")
            f.write("#PA_velar_max\n")
            f.write(str(PA_velar_max) + "\n")
            f.write("#PA_palatal_control\n")
            f.write(str(PA_palatal_control) + "\n")
            f.write("#PA_palatal_min\n")
            f.write(str(PA_palatal_min) + "\n")
            f.write("#PA_palatal_max\n")
            f.write(str(PA_palatal_max) + "\n")
            f.write("#PA_alveolardental_control\n")
            f.write(str(PA_alveolardental_control) + "\n")
            f.write("#PA_alveolardental_min\n")
            f.write(str(PA_alveolardental_min) + "\n")
            f.write("#PA_alveolardental_max\n")
            f.write(str(PA_alveolardental_max) + "\n")
            f.write("#PA_labial_control\n")
            f.write(str(PA_labial_control) + "\n")
            f.write("#PA_labial_min\n")
            f.write(str(PA_labial_min) + "\n")
            f.write("#PA_labial_max\n")
            f.write(str(PA_labial_max) + "\n")
            f.write("#wrapper\n")
            f.write("0\n")

        prod_info = self.diva_targets.init_struct()
        prod_info["name"] = name
        prod_info["length"] = length
        prod_info["interpolation"] = 'spline'

        prod_info["F0_control"] = F0_control
        prod_info["F0_min"] = minF0
        prod_info["F0_max"] = maxF0
        prod_info["F1_control"] = F1_control
        prod_info["F1_min"] = f1_25
        prod_info["F1_max"] = f1_75
        prod_info["F2_control"] = F2_control
        prod_info["F2_min"] = f2_25
        prod_info["F2_max"] = f2_75
        prod_info["F3_control"] = F3_control
        prod_info["F3_min"] = f3_25
        prod_info["F3_max"] = f3_75
        prod_info["pressure_control"] = 0
        prod_info["pressure_min"] = pressure_min
        prod_info["pressure_max"] = pressure_max
        prod_info["voicing_control"] = 0
        prod_info["voicing_min"] = voicing_min
        prod_info["voicing_max"] = voicing_max
        prod_info["PA_pharyngeal_control"] = PA_pharyngeal_control
        prod_info["PA_pharyngeal_min"] = PA_pharyngeal_min
        prod_info["PA_pharyngeal_max"] = PA_pharyngeal_max
        prod_info["PA_uvular_control"] = PA_uvular_control
        prod_info["PA_uvular_min"] = PA_uvular_min
        prod_info["PA_uvular_max"] = PA_uvular_max
        prod_info["PA_velar_control"] = PA_velar_control
        prod_info["PA_velar_min"] = PA_velar_min
        prod_info["PA_velar_max"] = PA_velar_max
        prod_info["PA_palatal_control"] = PA_palatal_control
        prod_info["PA_palatal_min"] = PA_palatal_min
        prod_info["PA_palatal_max"] = PA_palatal_max
        prod_info["PA_alveolardental_control"] = PA_alveolardental_control
        prod_info["PA_alveolardental_min"] = PA_alveolardental_min
        prod_info["PA_alveolardental_max"] = PA_alveolardental_max
        prod_info["PA_labial_control"] = PA_labial_control
        prod_info["PA_labial_min"] = PA_labial_min
        prod_info["PA_labial_max"] = PA_labial_max
        prod_info["wrapper"] = 0

        timeseries = self.diva_targets.timeseries(prod_info, doheader=True)
        t_dict = {'timeseries':timeseries}
        dest_file = file_root + '.mat'
        diva_utils.write_file(dest_file, t_dict)

        # src_file = "default_target.mat"
        # shutil.copy(src_file, dest_file)

        with open('diva.csv', "r") as target_list_file:
            csvreader = csv.reader(target_list_file)
            rows = []
            for row in csvreader:
                rows.append(row)

            last_idx = len(rows) - 1
            last_row = rows[last_idx]
            id_no = int(last_row[0]) + 1

        with open('diva.csv', "a+") as target_list_file:
            target_list_file.write(str(id_no) + ',' + name + "\n")

        if path.exists(diva_target_filename):
            print("Target extraction complete: " + name)

        return True

    def is_sustained(self, file):
        if '_phrase.wav' in file.name:
            return False
        else:
            return True

    def is_static_pitch(self, file):
        if '_lhl' in file.name:
            return False
        else:
            return True

    def is_male(self, sub_directory):
        if '_m_' in sub_directory.name:
            return True
        else:
            return False

    def is_normal(self, sub_directory):
        if '_p_' in sub_directory.name:
            return False
        else:
            return True

    def generate(self, input_directory):
        if not os.path.isdir(input_directory):
            print("ERROR: Invalid input directory: " + str(input_directory))
        ctr = 0
        for subd in os.scandir(input_directory):
            if self.is_normal(subd):
                for inner_subd in os.scandir(subd):
                    for file in os.scandir(inner_subd):
                        if self.is_sustained(file):
                            if self.is_static_pitch(file):
                                self.extract(file)
                                ctr += 1
                                if ctr >= 500:
                                    print("{0} targets processed. Exiting...".format(ctr))
                                    sys.exit(0)


if __name__ == '__main__':
    target_gen = Target_Gen()
    target_gen.generate('./data/ori/')
