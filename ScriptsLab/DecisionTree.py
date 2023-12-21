import numpy as np


class DecisionTree:
    def __init__(self, min_samples, max_depth, data):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.data = data
        self.train_init()

    def get_max_mse(self, samples, mse, param):
        sorted_samples = sorted(samples, key=lambda x: self.data[x][param])

        max_mse = -1e8
        max_val = -1

        for i in range(self.min_samples - 1, len(samples) - self.min_samples):
            if self.data[sorted_samples[i]][param] == self.data[sorted_samples[i + 1]][param]:
                continue

            predict1 = sum(self.data[sorted_samples[j]]["bmi"] for j in range(i + 1))
            predict2 = sum(self.data[sorted_samples[j]]["bmi"] for j in range(i + 1, len(samples)))

            predict1 /= i + 1
            predict2 /= len(samples) - i - 1

            mse1 = sum((self.data[sorted_samples[j]]["bmi"] - predict1) ** 2 for j in range(i + 1))
            mse2 = sum((self.data[sorted_samples[j]]["bmi"] - predict2) ** 2 for j in range(i + 1, len(samples)))

            mse1 /= i + 1
            mse2 /= len(samples) - i - 1

            curr_val = mse - mse1 - mse2
            if curr_val > max_mse:
                max_mse = curr_val
                max_val = (self.data[sorted_samples[i + 1]][param] + self.data[sorted_samples[i]][param]) / 2

        return {'val': max_val, 'mse': max_mse}

    def get_samples(self, val, param, is_left):
        return [i for i, row in enumerate(self.data) if
                (is_left and row[param] <= val) or (not is_left and row[param] > val)]

    def train(self, node, depth):
        samples = node["samples"]
        bmi_values = np.array([self.data[i]["bmi"] for i in samples])

        node["predict"] = np.mean(bmi_values)
        node["mse"] = np.mean((bmi_values - node["predict"]) ** 2)
        node["isLeaf"] = False

        if depth == self.max_depth or len(samples) == self.min_samples:
            return

        val1 = self.get_max_mse(samples, node["mse"], "age")
        val2 = self.get_max_mse(samples, node["mse"], "gender")

        if val1["val"] == -1 and val2["val"] == -1:
            node["isLeaf"] = True
            return

        max_val, param = (val1, "age") if val1["mse"] >= val2["mse"] else (val2, "gender")

        node["val"] = max_val["val"]
        node["param"] = param

        node["left"] = {"samples": self.get_samples(max_val["val"], param, True)}
        self.train(node["left"], depth + 1)

        node["right"] = {"samples": self.get_samples(max_val["val"], param, False)}
        self.train(node["right"], depth + 1)

    def train_init(self):
        self.tree = {}
        self.tree["samples"] = []

        for i in range(len(self.data)):
            self.tree["samples"].append(i)

        self.train(self.tree, 1)

    def get_prediction(self, age, gender):
        elem = {"age": age, "gender": gender}
        current = self.tree
        res = 0
        depth = 1

        while True:
            res = current["predict"]

            if (current["isLeaf"] or
                    depth == self.max_depth or
                    len(current["samples"]) <= self.min_samples):
                break

            if elem[current["param"]] <= current["val"]:
                current = current["left"]
            else:
                current = current["right"]

            depth += 1

        return res