from unittest import TestCase
import diva_utils
import os
import csv
import torch
import torch.nn as nn


class TestMotorCortex(TestCase):

    def test_runs(self):
        continue_prompt = input("Do you want to test DIVA_Mat vs DIVA_Py? (0/1):")
        if int(continue_prompt) == 1:
            folder_name = input("Please enter folder containing test data...")
            matlab_out_name = input("Please enter name of the .mat file containing export data...")
            num_runs = input("Please enter the number of test points...")
            num_runs = int(num_runs)
            exported_data = diva_utils.read_file_parameter_alternate(folder_name + '/' + matlab_out_name + '.mat',
                                                                     "runs")
            print("LOSS FOR SAMPLES VT0-VT19: " + folder_name + "/" + matlab_out_name)
            losses = []
            loss = nn.MSELoss()
            loss
            for i in range(num_runs):
                pytorch_out_name = 'VT_' + str(i)
                pytorch_out_filename = folder_name + '/' + pytorch_out_name + '.mat'
                if os.path.exists(pytorch_out_filename):
                    mat_run_data = exported_data[0, i]
                    py_run_data = diva_utils.read_file_parameter_alternate(pytorch_out_filename, pytorch_out_name)
                    mat_run_data = torch.from_numpy(mat_run_data)
                    py_run_data = torch.from_numpy(py_run_data)
                    min = torch.min(mat_run_data)
                    max = torch.max(mat_run_data)
                    print("Min: " + str(min.item()))
                    print("Max: " + str(max.item()))

                    loss_result = loss(mat_run_data, py_run_data)
                    print(str(loss_result.item()))
                    losses.append([loss_result.item()])
                else:
                    print("MISSING FILE: " + pytorch_out_filename)
            file = open(folder_name + '/' + 'test_runs_out.csv', 'w+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(losses)
        self.assertTrue(True)

    def test_py_runs(self):
        continue_prompt = input("Do you want to test DIVA_Py vs DIVA_Py? (0/1):")
        if int(continue_prompt) == 1:
            folder_name = input("Please enter folder containing test data...")
            num_runs = input("Please enter the number of test points...")
            num_runs = int(num_runs)
            losses = []
            for i in range(num_runs):
                pytorch_out_name = 'VT_' + str(i)
                pytorch_out_name_2 = 'VT_' + str(i) + ' (2)'
                pytorch_out_filename = folder_name + '/' + pytorch_out_name + '.mat'
                pytorch_out_filename2 = folder_name + '/' + pytorch_out_name_2 + '.mat'
                if os.path.exists(pytorch_out_filename):
                    py_run_data = diva_utils.read_file_parameter_alternate(pytorch_out_filename, pytorch_out_name)
                    py_run_data2 = diva_utils.read_file_parameter_alternate(pytorch_out_filename2, pytorch_out_name)
                    py_run_data = torch.from_numpy(py_run_data)
                    py_run_data2 = torch.from_numpy(py_run_data2)
                    loss = nn.MSELoss()
                    loss_result = loss(py_run_data, py_run_data2)
                    print(str(loss_result.item()))
                    losses.append([loss_result.item()])
                else:
                    print("MISSING FILE: " + pytorch_out_filename)
            file = open(folder_name + '/' + 'test_runs_out.csv', 'w+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(losses)
        self.assertTrue(True)

    def test_compare_runs(self):
        continue_prompt = input("Do you want to test DIVA_Mat vs DIVA_Mat? (0/1):")
        if int(continue_prompt) == 1:
            folder_name = input("Please enter folder containing test data...")
            matlab_out_name = input("Please enter name of the .mat file containing export data...")
            matlab_out_name2 = input("Please enter name of the second .mat file containing export data...")
            num_runs = input("Please enter the number of test points...")
            num_runs = int(num_runs)
            exported_data = diva_utils.read_file_parameter_alternate(folder_name + '/' + matlab_out_name + '.mat',
                                                                     "runs")
            exported_data_2 = diva_utils.read_file_parameter_alternate(folder_name + '/' + matlab_out_name2 + '.mat',
                                                                       "runs")
            print("LOSS FOR SAMPLES VT0-VT19: " + folder_name + "/" + matlab_out_name)
            losses = []
            for i in range(num_runs):
                mat_run_data = exported_data[0, i]
                mat_run_data_2 = exported_data_2[0, i]
                mat_run_data = torch.from_numpy(mat_run_data)
                mat_run_data_2 = torch.from_numpy(mat_run_data_2)
                loss = nn.MSELoss()
                loss_result = loss(mat_run_data, mat_run_data_2)
                print(str(loss_result.item()))
                losses.append([loss_result.item()])
            file = open(folder_name + '/' + 'test_runs_out.csv', 'w+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(losses)
            self.assertTrue(os.path.exists(folder_name + '/' + 'test_runs_out.csv'))
