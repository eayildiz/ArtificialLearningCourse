class lineerRegression():
    def __init__():
        learning_rate = 0.000005
        epoch = 1000
        coef_height = 1
        coef_weight = 2
        coef_single = 0

    #x is stands for height, y is stands for weight, z is target.
    def fit(x_train, y_train, z_train):
        coef_height = 1
        coef_weight = 2
        coef_single = 0
        num_of_values = len(z_train)

        for i in range(epoch):
            calculated_BMI = [(coef_height * curr_height + coef_weight * curr_weight + coef_single) for curr_height, curr_weight in zip(x_train,y_train)]
            loss = sum([(calculated - target)**2 for calculated, target in zip(calculated_BMI, z_train)]) / num_of_values

            coef_height = coef_height - learning_rate * (2 * sum([(coef_height * x + coef_weight * y + coef_single - z) * x for x, y, z in zip(x_train,y_train, z_train)]) / num_of_values)
            coef_weight = coef_weight - learning_rate * (2 * sum([(coef_height * x + coef_weight * y + coef_single - z) * y for x, y, z in zip(x_train,y_train, z_train)]) / num_of_values)
            coef_single = coef_single - learning_rate * (2 * sum([(coef_height * x + coef_weight * y + coef_single - z) for x, y, z in zip(x_train,y_train, z_train)]) / num_of_values)
        self.coef_height = coef_height
        self.coef_weight = coef_weight
        self.coef_single = coef_single

    def predict(x_test, y_test):
        result = [self.coef_height * curr_height + self.coef_weight + curr_weight + coef_single for curr_height, curr_weight in zip(x_test, y_test)]