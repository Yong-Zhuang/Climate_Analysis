import math
import numpy as np

from keras import backend as K
from keras.callbacks import Callback

class ActRecTrainingUtils(object):
    def __init__(self):
        pass

    def get_one_cycle_lr_fn(self,total_steps, max_lr, tail_len):
        if tail_len >= total_steps:
            raise ValueError("Total number of steps should be longer than the tail.")
        
        cycle_len = total_steps - tail_len
        max_lr_step = math.floor(cycle_len / 2)
        initial_lr = max_lr / 10
        final_lr = max_lr / 1000    
        
        linear_change = (max_lr - initial_lr) / max_lr_step
        tail_linear_change = (final_lr - initial_lr) / tail_len
        
        neg_intercept = max_lr + (linear_change * max_lr_step)
        tail_intercept = initial_lr - (tail_linear_change * cycle_len)
        
        def one_cycle_fn(epoch):
            if epoch <= max_lr_step:
                lr = linear_change * epoch + initial_lr
            elif epoch > max_lr_step and epoch <= cycle_len:
                lr = -1 * linear_change * epoch + neg_intercept
            else:
                lr = tail_linear_change * epoch + tail_intercept
                
            return lr
        
        return one_cycle_fn  

    def get_one_cycle_momentum_fn(self,total_steps, max_lr, tail_len, max_momentum=0.95, min_momentum=0.85):
        if tail_len >= total_steps:
            raise ValueError("Total number of steps should be longer than the tail.")
            
        if min_momentum >= max_momentum:
            raise ValueError("Maximum momentum is less than minimum momentum.")
        
        cycle_len = total_steps - tail_len
        max_lr_step = math.floor(cycle_len / 2)

        momentum_linear_change = (max_momentum - min_momentum) / max_lr_step
        momentum_rising_intercept = min_momentum - momentum_linear_change * max_lr_step
        
        def one_cycle_mom_fn(epoch):
            if epoch <= max_lr_step:
                mom = -1 * momentum_linear_change * epoch + max_momentum
            elif epoch > max_lr_step and epoch <= cycle_len:
                mom = (momentum_linear_change * epoch) + momentum_rising_intercept
            else:
                mom = max_momentum
                
            return mom
        
        return one_cycle_mom_fn  

class MomentumScheduler(Callback):
    def __init__(self, schedule, verbose=0):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute.')
        momentum = float(K.get_value(self.model.optimizer.momentum))
        try:  # new API
            momentum = self.schedule(epoch, momentum)
        except TypeError:  # old API for backward compatibility
            momentum = self.schedule(epoch)
        if not isinstance(momentum, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                            'should be float.')
        K.set_value(self.model.optimizer.momentum, momentum)
        if self.verbose > 0:
            print('\nEpoch %05d: Momentum scheduler setting momentum '
                'to %s.' % (epoch + 1, momentum))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['momentum'] = K.get_value(self.model.optimizer.momentum)