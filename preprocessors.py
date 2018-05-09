
class Preprocessor
    """
    Abstract Preprocessor class
    """
    def transform(self, img):
        raise NotImplementedError()


class GreyScalePreprocessor(Preprocessor):

    def transform(self, img):
        #TODO
        pass