
class Dataset:
    def __init__(self):
        self.feature_vectors = None #TODO choose datastructure
        self.images = []
        self.lables = []
        self.information = {}

    def import(self, file):
        """
        Imports saved
        :param file:
        :return:
        """
        #TODO
        pass

    def export(self, file):
        #TODO
        pass

    def get(self, index):
        """
        Returns the feature vector and label for the given image
        :param index:
        :return:
        """
        feature = self.feature_vectors[index]
        lable = self.lables[index]
        return feature, lable