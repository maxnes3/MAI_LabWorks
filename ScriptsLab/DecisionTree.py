class DecisionTree:
    def __init__(self, min_samples, max_depth, data):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.data = data
        self.trainInit()

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
        node["predict"] = 0
        node["mse"] = 0
        node["isLeaf"] = False

        for i in node["samples"]:
            node["predict"] += self.data[i]["bmi"]

        node["predict"] /= len(node["samples"])

        for i in node["samples"]:
            diff = (self.data[i]["bmi"] - node["predict"]) ** 2
            node["mse"] += diff

        node["mse"] /= len(node["samples"])

        if depth == self.max_depth:
            return

        if len(node["samples"]) == self.min_samples:
            return

        val1 = self.get_max_mse(node["samples"], node["mse"], "age")
        val2 = self.get_max_mse(node["samples"], node["mse"], "gender")

        if val1["val"] == -1 and val2["val"] == -1:
            node["isLeaf"] = True
            return

        maxVal = val1
        param = "age"

        if val2["mse"] > val1["mse"]:
            maxVal = val2
            param = "gender"

        node["val"] = maxVal["val"]
        node["param"] = param

        node["left"] = {}
        node["left"]["samples"] = self.get_samples(maxVal["val"], param, True)

        self.train(node["left"], depth + 1)

        node["right"] = {}
        node["right"]["samples"] = self.get_samples(maxVal["val"], param, False)

        self.train(node["right"], depth + 1)

    def trainInit(self):
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

            if current["isLeaf"]:
                break

            if depth == self.max_depth:
                break

            if len(current["samples"]) <= self.min_samples:
                break

            if elem[current["param"]] <= current["val"]:
                current = current["left"]
            else:
                current = current["right"]

            depth += 1

        return res