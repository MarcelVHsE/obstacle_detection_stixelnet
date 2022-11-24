from DataLoader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import auc


class MetricTester:
    def __init__(self, test_data=DataLoader(80)):
        self.TestData = test_data
        self.roc_curve = []
        self.auc = 0.0
        with open('config.yaml') as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        self.num_gt = self.get_num_gt()

    def get_roc_curve(self):
        """
        Method to vary the tolerance of an accepted stixel_pos match starting with 0 and ending with the highest diff
        from GT to predict. Tolerance steps in 8 px.
        :return: a list of x|y coordinates while each point describes one tolerance step and x means fpr and y: tpr
        """
        # Prepare the GT label
        full_target_list = np.asarray(self.TestData.target[..., 1]).astype(int)
        # Prepare the Prediction label, returns the index of the max (prob.) of every col
        full_prediction_list = np.argmax(self.TestData.predict, axis=1)

        # Apply the difference
        diff_list = []
        for i in range(len(full_target_list)):
            diff_list.append((abs(full_target_list[i]-full_prediction_list[i]), self.TestData.target[i][0]))
        diff_list = np.asarray(diff_list)
        # iterate over the tolerances and count the num of tp and fp and increase the tolerance until max
        for tolerance in range(int(np.amax(diff_list[..., 0])+1)):
            # e.g.  diff = 87 and tolerance = 0: 87-0 is > 0 means fp ... tolerance = 87 means <0: tp+1
            tp = 0
            fp = 0
            for stixel_pred in diff_list:
                # Ground Truth available
                if stixel_pred[1] == 1:
                    # Within tolerance
                    if stixel_pred[0] - tolerance <= 0:
                        tp += 1
                    else:
                        # predicts outside the tolerance
                        fp += 1
                else:
                    # if it predicts something without having GT
                    fp += 1
            self.roc_curve.append((fp, tp))
        self.roc_curve = np.asarray(self.roc_curve)
        return self.roc_curve

    @staticmethod
    def __get_auc(x_vals, y_vals):
        """
        Calculates the area-under-curve (AUC) for the current ROC
        :return: a float number with ther area-under-curve
        """
        calc_auc = auc(x_vals, y_vals)
        return calc_auc

    def plot_roc_curve(self):
        fpr = []
        tpr = []
        for pt in self.roc_curve:
            fpr.append(pt[0]/(self.config['label_size']['x']*self.TestData.num_datasets))
            tpr.append(pt[1]/self.num_gt)
        self.auc = self.__get_auc(fpr, tpr)
        # plot the roc curve for the model
        plt.figure(figsize=(20, 12))
        plt.plot([0, 1], [1, 0], color="navy", linestyle="--")
        plt.plot(fpr, tpr, color="darkorange", label='ROC AUC=%.3f' % self.auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_num_gt(self):
        num_gt = 0
        for gt_avail in self.TestData.target:
            if gt_avail[0] == 1:
                # if GT is available, overtake the stixel_pos
                num_gt += 1
        return num_gt


"""
    # returns the index of the max (prob.) of every col
    predictions = np.argmax(self.TestData.predict, axis=1)
    max_val_list = []
    for i in range(len(testData.predict)):
        # creates a list with the maximum probability value and the related index e.g.  (...,[0.61][94])
        max_val_list.append((np.amax(testData.predict[i]), predictions[i]))
    # Convert to numpy
    max_val_list = np.asarray(max_val_list)

    y_list = []
    score_list = []
    # Apply the matching logic: when is a prediction true?
    # plausibility check whether the lib is use correctly
    num_gt = 0
    count_correct_pred = 0
    # Go over the existing GT pts
    for i in range(len(testData.target)):
        # if GT is available...
        if testData.target[i][0] == 1:
            y_list.append(1)
            num_gt += 1
            # if the max prediction is within range ... + tolerance - check index
            if testData.target[i][1] - gt_tolerance <= int(max_val_list[i][1]) <= testData.target[i][1] + gt_tolerance:
                if max_val_list[i][1] > probability_threshold:
                    score_list.append(1)
                    count_correct_pred += 1
            else:
                score_list.append(max_val_list[i][0])
                print("Pred: " + str(max_val_list[i][1]) + "\t Targ: " + str(testData.target[i][1]) + "\t Colu: " + str(i))
        else:
            y_list.append(0)
            score_list.append(max_val_list[i][0])

        score_list.append(abs(testData.target[i][1] - int(max_val_list[i][1])))

    # ROC -example-
    # the metric compares correct or not correct predictions, based on the probability
    #     fpr: false-positive-rate
    #     tpr: true-positive-rate
    #     auc: area-under-curve

    fpr, tpr, thresholds = metrics.roc_curve(y_list, score_list, pos_label=1)
    auc = roc_auc_score(y_list, score_list)
    plot_roc_curve(fpr, tpr, auc)

    print("Percentage of GT available: " + str(100/len(testData.target)*num_gt) + " %")
    print("Percentage of GT hit: " + str(100/num_gt*count_correct_pred) + " %")
"""