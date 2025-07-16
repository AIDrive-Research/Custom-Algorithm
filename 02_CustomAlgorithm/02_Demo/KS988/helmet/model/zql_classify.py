import numpy as np

from logger import LOGGER
from model import OnnxModel


class Model(OnnxModel):
    default_args = {
        'img_size': 224
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['model'])
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]

    def _load_args(self, args):
        try:
            self.img_size = args.get('img_size', self.default_args['img_size'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    @staticmethod
    def __softmax(x):
        # 计算指数
        exp_x = np.exp(x)
        # 对每行求和
        sum_exp_x = np.sum(exp_x)
        # 计算softmax
        softmax_output = exp_x / sum_exp_x
        return softmax_output

    def infer(self, data, **kwargs):
        """
        图像分类
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = None
        if self.status:
            try:
                # Do image preprocess
                image = data
                image, _, _ = self._letterbox(image, (self.img_size, self.img_size), mean=self.mean, std=self.std, stretch=True)
                outputs = self._onnx_infer('model', image)
                output = self.__softmax(outputs[0]).reshape(-1)
                obj = {
                    'output': output.tolist()
                }
                infer_result = obj
            except:
                LOGGER.exception('infer')
        return infer_result
