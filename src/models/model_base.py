from abc import ABC, abstractmethod
from models.metrics import mean_prec_iou
from models.losses import jaccard_distance_loss, dice_loss, lovasz_hinge, mean_iou_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


class ModelBase(ABC):
    def __init__(self, dropout, last_activation='sigmoid'):
        self.custom_objects = {}
        width = self.get_image_size()
        self.model = self._create_model(input_shape=[width, width, 3], dropout=dropout,
                                        last_activation=last_activation)

    @abstractmethod
    def get_image_size(self):
        raise NotImplementedError("")

    @abstractmethod
    def _create_model(self, input_shape, dropout=0.0, last_activation='sigmoid'):
        raise NotImplementedError("")

    @abstractmethod
    def get_image_preprocessor(self):
        return None

    def __get_loss(self, loss):
        if loss == 'binary_crossentropy':
            return loss
        if loss == 'dice_loss':
            self.custom_objects[dice_loss.__name__] = dice_loss
            return dice_loss
        if loss == 'jaccard_distance_loss':
            self.custom_objects[jaccard_distance_loss.__name__] = jaccard_distance_loss
            return jaccard_distance_loss
        if loss == 'lovasz_hinge':
            self.custom_objects[lovasz_hinge.__name__] = lovasz_hinge
            return lovasz_hinge
        if loss == 'mean_iou_loss':
            self.custom_objects[mean_iou_loss.__name__] = mean_iou_loss
            return mean_iou_loss
        raise Exception("Unknown loss: " + str(loss))

    def __get_metric(self, metric):
        if metric == 'accuracy':
            return metric
        if metric == 'mean_prec_iou':
            self.custom_objects[mean_prec_iou.__name__] = mean_prec_iou
            return mean_prec_iou
        raise Exception("Unknown metric: " + str(metric))

    def __get_metrics(self, metrics):
        return [self.__get_metric(metric) for metric in metrics]

    def compile(self, optimizer, loss='binary_crossentropy', metrics='mean_prec_iou'):
        if not isinstance(metrics, list):
            metrics = [metrics]
        loss = self.__get_loss(loss)
        metrics = self.__get_metrics(metrics)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_generator(self, train_gen, epochs=1, shuffle=True, val_gen=None, steps_per_epoch=None,
                      validation_steps=None, lr_patience=None, lr_alpha=0.2, early_stopping=None,
                      model_path="model.h4", tensorboard_dir=None,
                      save_not_best_only=False):

        callbacks = self.__get_callbacks(model_path, lr_patience, lr_alpha, early_stopping, tensorboard_dir,
                                         save_not_best_only)
        return self.model.fit_generator(train_gen, epochs=epochs, shuffle=shuffle,
                                        validation_data=val_gen, callbacks=callbacks, verbose=2,
                                        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    @staticmethod
    def __get_callbacks(model_path, lr_patience=3, lr_alpha=0.2, early_stopping=10, tensorboard_dir=None,
                        save_not_best_only=False):
        callbacks = []
        if early_stopping is not None and early_stopping > 0:
            callbacks.append(EarlyStopping(patience=early_stopping, verbose=1))
        if tensorboard_dir is not None:
            callbacks.append(TensorBoard(tensorboard_dir))
        if model_path is not None:
            monitor = 'val_loss'
            if save_not_best_only:
                monitor = 'loss'
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True, monitor=monitor))
        if lr_patience is not None and lr_patience > 0:
            callbacks.append(ReduceLROnPlateau(monitor='loss', factor=lr_alpha, patience=lr_patience))
        return callbacks

    def evaluate(self, x, y, verbose=0):
        return self.model.evaluate(x, y, verbose=verbose)

    def load_weights(self, file_path):
        self.model.load_weights(file_path)

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def summary(self):
        print(self.model.summary())

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        return self.model.predict(x, batch_size, verbose=verbose, steps=steps)
