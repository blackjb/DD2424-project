
from dataset import Dataset

class Parser:
    """
    Abstract Parser class
    """
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

    def import(file):
        """
        Imports image files runs image preprocessing and returns a Dataset object containing the imported images
        :return: dataset - Dataset object containing the imported iamges
        """
        raise NotImplementedError()

class ImageNetParser(Parser):
    """
    Parser for the image net dataset
    """
    def import(file):
        dataset = Dataset()
        return dataset
