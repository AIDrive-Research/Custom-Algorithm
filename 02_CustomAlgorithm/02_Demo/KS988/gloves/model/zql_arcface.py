import cv2
import numpy as np

from logger import LOGGER
from model import OnnxModel


class Model(OnnxModel):
    default_args = {
        'img_size': 256
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])
        self.mean = [127.5, 127.5, 127.5]
        self.std = [128.0, 128.0, 128.0]

    def _load_args(self, args):
        try:
            self.img_size = args.get('img_size', self.default_args['img_size'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, data, **kwargs):
        """
        arcface特征提取
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = None
        if self.status:
            try:
                # Do image preprocess
                raw_width, raw_height = data.shape[1], data.shape[0]
                if max(data.shape[:2]) != self.img_size:
                    scale = self.img_size / max(data.shape[:2])
                    if raw_height > raw_width:
                        data = cv2.resize(data, (int(raw_width * scale), self.img_size))
                    else:
                        data = cv2.resize(data, (self.img_size, int(raw_height * scale)))
                data, _, _ = self._letterbox(data, (self.img_size, self.img_size), mean=self.mean, std=self.std)
                outputs = self._onnx_infer('model', data)
                # Do postprocess
                output = outputs[0].reshape(1, -1)
                norm = np.linalg.norm(output, ord=2, axis=1, keepdims=True)
                feature = (output / norm).reshape(-1)
                infer_result = feature.tolist()
            except:
                LOGGER.exception('infer')
        return infer_result
