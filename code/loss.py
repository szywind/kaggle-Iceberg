from keras import backend as K

def focus_loss(y_true, y_pred):
    gamma = 2.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    loss = y_true * K.log(y_pred + K.epsilon()) * (1 - y_pred + K.epsilon()) ** gamma + \
           (1 - y_true) * K.log(1 - y_pred + K.epsilon()) * (y_pred + K.epsilon()) ** gamma
    return -K.mean(loss)


def focal_loss(y_true, y_pred, gamma=2.):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    epsilon = K.epsilon()

    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = K.pow(1.0 - y_pred, gamma)

    loss = - K.sum(loss * y_true * K.log(y_pred), axis=-1)
    return loss

# https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
def focal_loss(gamma=2, alpha=2):
	def focal_loss_fixed(y_true, y_pred):
		if(K.backend()=="tensorflow"):
			import tensorflow as tf
			pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
		if(K.backend()=="theano"):
			import theano.tensor as T
			pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
	return focal_loss_fixed