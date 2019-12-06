# RAdam
# https://github.com/CyberZHG/keras-radam
! pip install keras-rectified-adam

from keras_radam import RAdam
radam = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
model.compile(optimizer=radam, loss='mse')
