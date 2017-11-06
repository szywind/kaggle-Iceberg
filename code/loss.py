from keras import backend as K

def focus_loss(y_true,y_pred):
    gamma = 2.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    loss = y_true*K.log(y_pred+K.epsilon())*(1-y_pred+K.epsilon())**gamma + \
           (1-y_true)*K.log(1-y_pred+K.epsilon())*(y_pred+K.epsilon())**gamma
    return -K.mean(loss)