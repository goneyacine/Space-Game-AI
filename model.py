import tensorflow as tf

class model:
    #max_obsticals_count refers to the max number of obsticals the model can observe and it defines the number of inputs in the layer
    def __init__(self,lr=1e-4,batch_norm=False,max_obsticals_count=5,activation='relu',load_from_disk = False,model_path='model.keras',output_activation='softmax') -> None:
        self.lr = lr
        if not load_from_disk:
         self.__init_model__(batch_norm=batch_norm
                                          ,max_obsticals_count=max_obsticals_count,activation=activation,output_activation=output_activation)
        else:
            self.model = self.load(model_path=model_path)
    #max_obsticals_count refers to the max number of obsticals the model can observe and it defines the number of inputs in the layer    
    def __init_model__(self,batch_norm=False,max_obsticals_count=5,activation='relu',output_activation='softmax'):
        input = tf.keras.Input(max_obsticals_count * 2 + 1)
        layer1 = tf.keras.layers.Dense(64,activation=activation)(input)
        if batch_norm:
         layer1 = tf.keras.layers.BatchNormalization()(layer1)
        layer2 = tf.keras.layers.Dense(32,activation=activation)
        if batch_norm:
         layer2 = tf.keras.layers.BatchNormalization()(layer2)
        output = tf.keras.layers.Dense(3,activation=output_activation)(layer2)
        self.model = output 
        
    def update(self):
        pass
    def compute_reward(self):
        pass
    def save(self,output_path='model.keras'):
        pass
    def load(self,model_path='model.keras'):
        pass