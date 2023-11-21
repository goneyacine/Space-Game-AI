
import pygame
import math
import random
import numpy as np
import tensorflow as tf
import model
import tensorflow_probability as tfp
def start_game():
 pg_model = model.pg_model()
 MAX_OBSTICALS = 20
 PLAYER_POS_RANGE = (20,480) 
 OBSITCALS_X_POS_RANGE = 500
 OBSTICAL_MAX_Y_POS = 600
 OBSTICAL_MAX_SPEED = 0.5
 OBSTICAL_MIN_SPEED = 0.05
 OBSTICAL_SIZE = 10
 PLAYER_SIZE = 15
 PLAYER_SPEED = .5
 SCREEN = pygame.display.set_mode([500, 500])
 player_pos = [200,450]
 obsticals = []
 running = True
 restart = False
 steps_count = 0
 states = []
 actions = []
 rewards = tf.constant([], dtype=tf.float32)
 dist = None
 font = pygame.font.Font(None, 36) 
 while running and not restart: 
     
  #take action
  old_state = np.array([player_pos[0]])
  if(len(obsticals) > 0):
   for obstical in obsticals:
        old_state = np.append(old_state,[obstical[0],obstical[1]])
   dist = tfp.distributions.Categorical(probs=pg_model.model(old_state.reshape(1, -1),training=True),dtype=tf.float32)
   action = dist.sample()[0].numpy()
   if action == 1  and player_pos[0] > PLAYER_POS_RANGE[0]:
        player_pos[0] -= PLAYER_SPEED
   elif action == 2 and player_pos[0] < PLAYER_POS_RANGE[1]:
        player_pos[0] += PLAYER_SPEED
  #Obsticals update
  #destroy old obsticals & update positions
  for obstical in obsticals:
      obstical[1] += obstical[2] 
      if obstical[1] > OBSTICAL_MAX_Y_POS:
          obsticals.remove(obstical)
  #spawn new obsticals
  while len(obsticals) < MAX_OBSTICALS:
      obsticals.append([random.random() * OBSITCALS_X_POS_RANGE,-30,random.random() * OBSTICAL_MAX_SPEED + OBSTICAL_MIN_SPEED])
  
  #Input update
  for event in pygame.event.get():
      if event.type == pygame.QUIT:
          running = False
  keys = pygame.key.get_pressed()
  if keys[pygame.K_LEFT] and player_pos[0] > PLAYER_POS_RANGE[0]:
      player_pos[0] -= PLAYER_SPEED
  if keys[pygame.K_RIGHT] and player_pos[0] < PLAYER_POS_RANGE[1]:
      player_pos[0] += PLAYER_SPEED
  #Collision detection 
  for obstical in obsticals:
      if math.sqrt((player_pos[0] - obstical[0]) ** 2 + (player_pos[1] - obstical[1]) ** 2 ) < PLAYER_SIZE + OBSTICAL_SIZE:
          restart = True
          break
  #Rendering
  SCREEN.fill((255, 255, 255))
  #render player
  pygame.draw.circle(SCREEN, (0, 0, 255), (player_pos[0], player_pos[1]), PLAYER_SIZE)
  #render obsticals
  for obstical in obsticals:
      pygame.draw.circle(SCREEN,(255,0,0),(obstical[0],obstical[1]),OBSTICAL_SIZE)
  
  pygame.display.flip()
  new_state = np.array([player_pos[0]])
  for obstical in obsticals:
        new_state = np.append(new_state,[obstical[0],obstical[1]])
  if(dist != None):
   actions.append(action)
   states.append(old_state)
   reward = None
   if(rewards.shape[0]>0):
    reward =  tf.constant([pg_model.compute_reward(state=new_state,action=action,previous_reward=rewards.numpy()[steps_count-2]) ],dtype=tf.float32)
   else:
    reward =  tf.constant([pg_model.compute_reward(state=new_state,action=action,previous_reward=0) ],dtype=tf.float32)
   text_surface = font.render(str(reward.numpy()[0]), True, (255, 255, 255)) 
   SCREEN.blit(text_surface, (500 // 2 - text_surface.get_width() // 2, 500 // 2 - text_surface.get_height() // 2))
   rewards = tf.concat([rewards, reward], axis=0)
  steps_count += 1
 if not running:
  pygame.quit()
 elif restart :
     print(len(actions))
     print(steps_count)
     pg_model.update(states=states,actions=actions,rewards=rewards)
     start_game()


pygame.init()
start_game()