import tensorflow as tf
import os
import tensorflow_probability as tfp
import numpy as np
class pg_model:
    #max_obsticals_count refers to the max number of obsticals the model can observe and it defines the number of inputs in the layer
    def __init__(self,lr=1e-1,batch_norm=False,max_obsticals_count=20,activation='relu',load_from_disk = False,model_path='model.keras',output_activation='softmax',learning_rate=1e-4) -> None:
        self.lr = lr
        self.max_obsticals_count = max_obsticals_count
        if not load_from_disk:
         self.__init_model__(batch_norm=batch_norm
                                          ,max_obsticals_count=max_obsticals_count,activation=activation,output_activation=output_activation)
        else:
            self.model = self.load(model_path=model_path)
        self.init_optimizer(learning_rate=learning_rate)
    #max_obsticals_count refers to the max number of obsticals the model can observe and it defines the number of inputs in the layer    
    def __init_model__(self,batch_norm=False,max_obsticals_count=5,activation='relu',output_activation='softmax'):
        input = tf.keras.Input(max_obsticals_count * 2 + 1)
        x = tf.keras.layers.Dense(64,activation=activation)(input)
        if batch_norm:
         x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(32,activation=activation)(x)
        if batch_norm:
         x = tf.keras.layers.BatchNormalization()(x)
        output = tf.keras.layers.Dense(3,activation=output_activation)(x)
        self.model = tf.keras.models.Model(inputs=input,outputs=output) 
        
    def init_optimizer(self,learning_rate=1e-4):    
        self.optimizer =  tf.keras.optimizers.SGD(learning_rate)
    def update(self,states,actions,rewards,auto_save=False,output_path='model.keras'):
        
        for state,reward, action in zip(states,rewards, actions):
         with tf.GradientTape() as tape:
           p = self.model(state.reshape(1,-1),training=True)
           dist = tfp.distributions.Categorical(p)
           log_prob = dist.log_prob(action)
           loss = -log_prob*reward
           gradients = tape.gradient(loss, self.model.trainable_variables)
         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if auto_save:
            self.save(output_path=output_path)
             
    def compute_reward(self,state,action,previous_reward,discount_factor=0.6,hit=False):
        reward = previous_reward * discount_factor
        if hit:
            reward = -10
            return reward
        if action!= 0:
           reward -= 0.4
        for i in range(self.max_obsticals_count):
            player_x_pos = state[0]
            obstical_x_pos = state[i*2 +1]
            obstical_y_pos = state[i*2 +2]
            a = obstical_y_pos ** 2 * 10
            b = abs(obstical_x_pos - player_x_pos) - 1
            reward += a * b
        return reward
    def save(self,output_path='model.keras'):
         if not os.path.exists(output_path):
            os.mkdir(output_path)
         self.model.save(output_path)
    def load(self,model_path='model.keras'):
        self.model = tf.keras.models.load_model(model_path)