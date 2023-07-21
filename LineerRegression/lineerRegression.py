from typing import List

class lineerRegression:
    def __init__(self):
        self.learning_rate = 0.000005
        self.epoch = 1000
        self.coef_height = 1
        self.coef_weight = 2
        self.coef_single = 0
        self.loss_over_time = []
        self.accuracy_over_time = []

    #x is stands for height, y is stands for weight, z is target.
    def fit(self, features: List[List[int]], z_train: List[int], feature_test: List[List[int]], target_test: List[int]):
        coef_height = self.coef_height
        coef_weight = self.coef_weight
        coef_single = self.coef_single
        num_of_values = len(z_train)

        for i in range(self.epoch):
            calculated_BMI = [(coef_height * row[0] + coef_weight * row[1] + coef_single) for row in features]
            calculated_BMI_test = [(coef_height * row[0] + coef_weight * row[1] + coef_single) for row in feature_test]
            
            loss = sum([(calculated - target)**2 for calculated, target in zip(calculated_BMI_test, target_test)]) / num_of_values
            accuracy = sum([abs(calculated - target) for calculated, target in zip(calculated_BMI_test, target_test)]) / num_of_values
            self.loss_over_time.append(loss)
            self.accuracy_over_time.append(accuracy)

            coef_height = coef_height - self.learning_rate * (2 * sum([(cal_BMI - z) * row[0] for cal_BMI, row, z in zip(calculated_BMI, features, z_train)]) / num_of_values)
            coef_weight = coef_weight - self.learning_rate * (2 * sum([(cal_BMI - z) * row[1] for cal_BMI, row, z in zip(calculated_BMI, features, z_train)]) / num_of_values)
            coef_single = coef_single - self.learning_rate * (2 * sum([(cal_BMI - z) for cal_BMI, z in zip(calculated_BMI, z_train)]) / num_of_values)
        self.coef_height = coef_height
        self.coef_weight = coef_weight
        self.coef_single = coef_single

    def predict(self, features):
        return [self.coef_height * row[0] + self.coef_weight * row[1] + self.coef_single for row in features], self.loss_over_time, self.accuracy_over_time