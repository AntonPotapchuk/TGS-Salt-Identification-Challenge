from abc import ABC, abstractmethod


class ModelBase(ABC):
    def __init__(self, dropout, last_activation='sigmoid'):
        self.custom_objects = {}
        width = self.get_image_size()
        self.model = self._create_model(input_shape=[width, width, self.get_number_of_channels()], dropout=dropout,
                                        last_activation=last_activation)

    @staticmethod
    @abstractmethod
    def get_image_size():
        raise NotImplementedError("")

    def get_number_of_channels(self):
        return 3

    @abstractmethod
    def _create_model(self, input_shape, dropout=0.0, last_activation='sigmoid'):
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def get_image_preprocessor():
        return None

    def __getattr__(self, name):
        return getattr(self.model, name)
