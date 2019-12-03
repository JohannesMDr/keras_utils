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

# cb_my = loss_history.LossHistory()
