import random

import numpy as np
from os.path import exists


def generateDataSetSet(numberOfDataSets):
    tmp = []
    for i in range(numberOfDataSets):
        tmp.append(i + 1)
    random.shuffle(tmp)
    return tmp


class DataLoader:
    """
    Creates a loader for data. Excepts numpy arrays of dictionaries:

    {
        "image": array of image pixels,
        "action": action taken
    }

    Actions:
        0 = forward
        1 = backwards
        2 = left
        3 = right

    Excepts that data is saved in a separate np files for each run named "run_x.np", where x is the count of the run.
    Loads 4 files at once until all the files has been loaded.
    """

    def __init__(self, path, datasets):
        super().__init__()
        self._file_path = path
        self._run_count = 1
        self._datasets = generateDataSetSet(datasets)
        self._data = self.collectData()
        if self._data is None:
            print("Data set not found")
            raise StopIteration
        self._first = self.checkFirstAction()

    def __iter__(self):
        return self

    def __next__(self):
        if self._first == len(self._data):
            self._data = self.collectData()
            if self._data is None:
                raise StopIteration
            self._first = self.checkFirstAction()
        data = self._data[self._first]
        self._first = self._first + 1
        return data

    def checkForNextArray(self):
        try:
            fileNumber = self._datasets.pop()
        except IndexError:
            print("No more data left")
            return None
        file_exists = exists(self._file_path + str(fileNumber) + ".npy")
        if file_exists:
            print("Loading dataset number", fileNumber)
            data = np.load(self._file_path + str(fileNumber) + ".npy", allow_pickle=True)
            self._run_count = self._run_count + 1
            return data
        else:
            print("No more data left. Loaded ", self._run_count - 1, " data arrays")
            return None

    def checkFirstAction(self):
        for i in range(len(self._data)):
            action = self._data[i]["action"]
            if action is not None:
                return i

    def checkLength(self):
        return len(self._data) - self._first

    def collectData(self):
        data = self.checkForNextArray()
        if data is None:
            return None

        withoutNones = []
        zeros = []
        ones = []
        twos = []
        trees = []

        for j in range(3):
            for i in data:
                if i["action"] is not None:
                    withoutNones.append(i)
                    if i["action"] == "0":
                        zeros.append(i)
                    if i["action"] == "1":
                        ones.append(i)
                    if i["action"] == "2":
                        twos.append(i)
                    if i["action"] == "3":
                        trees.append(i)

            data = self.checkForNextArray()
            if data is None:
                break

        finalData = []

        minimum = min((len(zeros), len(twos), len(trees)))
        random.shuffle(zeros)
        random.shuffle(twos)
        random.shuffle(trees)

        for i in range(minimum):
            finalData.append(zeros[i])
            finalData.append(twos[i])
            finalData.append(trees[i])
        random.shuffle(finalData)

        del zeros
        del twos
        del trees

        while len(finalData) == 0:
            print("finalData == 0, trying again...")
            self.collectData()

        return finalData
