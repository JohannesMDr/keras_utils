import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback

model_checkpoint = ModelCheckpoint(
	filepath=os.path.join(path_model, 'unet3-1_{epoch:02d}.h5'),
	period=1,
	save_weights_only=True)
					
					
# ref: https://qiita.com/typecprint/items/3ef54ce47e32e286d092
class LossHistory(Callback):
	def __init__(self):
		# コンストラクタに保持用の配列を宣言しておく
		self.train_acc = []
		self.train_loss = []
		self.val_acc = []
		self.val_loss = []

	def on_epoch_end(self, epoch, logs={}):
		# 配列にEpochが終わるたびにAppendしていく
		self.train_acc.append(logs['acc'])
		self.val_acc.append(logs['val_acc'])
		self.train_loss.append(logs['loss'])
		self.val_loss.append(logs['val_loss'])

		# 保存
		np.save('history.npy', np.array([self.train_acc, self.val_acc, self.train_loss, self.val_loss]))

		# グラフ描画部
		plt.clf()
		plt.figure(num=1, clear=True)
		plt.title('accuracy')
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.plot(self.train_acc, label='train')
		plt.plot(self.val_acc, label='validation')
		plt.legend()
		plt.savefig('history.png')
		plt.pause(0.1)

# cb_my = LossHistory()


# ref: https://qiita.com/shoji9x9/items/896204303a7a56321d4c
from keras.callbacks import Callback
from hyperdash import Experiment

class Hyperdash(Callback):
    def __init__(self, entries, exp):
        super(Hyperdash, self).__init__()
        self.entries = entries
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        for entry in self.entries:
            log = logs.get(entry)            
            if log is not None:
                self.exp.metric(entry, log)

exp = Experiment("unet3-1")
hd_callback = Hyperdash(["val_loss", "loss", "val_accuracy", "accuracy"], exp)
