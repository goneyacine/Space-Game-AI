import tensorflow as tf
import os
import tensorflow_probability as tfp
import numpy as np
class pg_model:
    #max_obsticals_count refers to the max number of obsticals the model can observe and it defines the number of inputs in the layer
    def __init__(self,lr=1e-3,batch_norm=False,max_obsticals_count=20,activation='relu',load_from_disk = False,model_path='model.keras',output_activation='softmax',learning_rate=1e-4) -> None:
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
        
    def init_optimizer(self,learning_rate=1e-2):    
        initial_learning_rate = learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.5,
        staircase=True)

        self.optimizer =  tf.keras.optimizers.SGD(lr_schedule)
    def update(self,states,actions,rewards,auto_save=False,output_path='C:',old_model=None,discount_factor=0.9):
        rewards = rewards.numpy()
        i = len(rewards)-2
        while(i > 0):
            rewards[i] += rewards[i+1] * discount_factor
            print(rewards[i])
            i -=1
        rewards = tf.convert_to_tensor(rewards)        
        @tf.function
        def step(state,action,reward,old_model=None):
          with tf.GradientTape() as tape:
           if(old_model == None):
               old_model = self.model
           p = old_model(state,training=True)
           dist = tfp.distributions.Categorical(p)
           log_prob = dist.log_prob(action)
           loss = -log_prob*reward
          gradients = tape.gradient(loss, self.model.trainable_variables)
          self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        for state,reward, action in zip(states,rewards, actions):
            state = state.reshape(1,-1)
            step(state=state,reward=reward,action=action,old_model=old_model)
        if auto_save:
            self.save(output_path=os.path.join(output_path,'model.keras'))
    #computes the imediate reward         
    def compute_reward(self,state,action,screen_size=500,obsticals_count=20,hit=False):
        reward = .3
        if hit:
            reward = -10
            return reward
        if action!= 0:
           reward -= 0.04
        for i in range(self.max_obsticals_count):
            player_x_pos = state[0]
            obstical_x_pos = state[i*2 +1]
            obstical_y_pos = state[i*2 +2]
            a = ((abs(obstical_x_pos - player_x_pos)) / screen_size)**3 - 0.13
            b = -((obstical_y_pos - (screen_size/2)) / screen_size)**3
            reward += a + b 
        return reward / obsticals_count *100
    def save(self,output_path='model.keras'):
         if not os.path.exists(output_path):
            os.mkdir(output_path)
         self.model.save(output_path)
    def load(self,model_path='model.keras'):
        self.model = tf.keras.models.load_model(model_path)