import cv2
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from DataLoader import DataLoader
from MetricTester import MetricTester


if __name__ == '__main__':
    """ ---------------- """
    """ For a specific file, enter an ID to DataLoader. For all: leave it blank or write None"""
    InceptionData = DataLoader()
    Tester = MetricTester(test_data=InceptionData)
    Tester.get_roc_curve()
    Tester.plot_roc_curve()
