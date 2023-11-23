
import pygame
import math
import random
import numpy as np
import tensorflow as tf
import model
import tensorflow_probability as tfp
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="Space Game AI",
    
    # track hyperparameters and run metadata
    config={
         'discount factor':0.9,
          'episodes steps':100
    }
)

config = wandb.config

MAX_OBSTICALS = 20
PLAYER_POS_RANGE = (20,480) 
OBSITCALS_X_POS_RANGE = 500
OBSTICAL_MAX_Y_POS = 600
OBSTICAL_MAX_SPEED = 0.5
OBSTICAL_MIN_SPEED = 0.05
OBSTICAL_SIZE = 10
PLAYER_SIZE = 15
PLAYER_SPEED = .5
max_score = 0

def render(screen,obsticals,player_pos):
  screen.fill((50, 30, 30))
  #render player
  pygame.draw.circle(screen, (220, 220, 220), (player_pos[0], player_pos[1]), PLAYER_SIZE)
  #render obsticals
  for obstical in obsticals:
      pygame.draw.circle(screen,(255,210,210),(obstical[0],obstical[1]),OBSTICAL_SIZE)
      
def update_obsticals(obsticals):
     #destroy old obsticals & update positions
  for obstical in obsticals:
      obstical[1] += obstical[2] 
      if obstical[1] > OBSTICAL_MAX_Y_POS:
          obsticals.remove(obstical)
  #spawn new obsticals
  while len(obsticals) < MAX_OBSTICALS:
      obsticals.append([random.random() * OBSITCALS_X_POS_RANGE,-30,random.random() * OBSTICAL_MAX_SPEED + OBSTICAL_MIN_SPEED])
  return obsticals
def move_player(player_pos,action):
   keys = pygame.key.get_pressed()
   if (action == 1 or keys[pygame.K_LEFT])  and player_pos[0] > PLAYER_POS_RANGE[0]:
        player_pos[0] -= PLAYER_SPEED
   elif (action == 2 or keys[pygame.K_RIGHT])and player_pos[0] < PLAYER_POS_RANGE[1]:
        player_pos[0] += PLAYER_SPEED
   return player_pos
def call_model(model,state):
   dist = tfp.distributions.Categorical(probs=model.model(state.reshape(1, -1),training=True),dtype=tf.float32)
   action = dist.sample()[0].numpy()    
   return action
def get_state(player_pos,obsticals):
  state= np.array([player_pos[0]-250])
  if(len(obsticals) > 0):
   for obstical in obsticals:
        state = np.append(state,[obstical[0]-250,obstical[1]-250])
   return state
def check_collisions(obsticals,player_pos):
     for obstical in obsticals:
      if math.sqrt((player_pos[0] - obstical[0]) ** 2 + (player_pos[1] - obstical[1]) ** 2 ) < PLAYER_SIZE + OBSTICAL_SIZE:
          return True
     return False
def start_game():
 score = 0
 pg_model = model.pg_model()
 screen = pygame.display.set_mode([500, 500])
 player_pos = [200,450]
 obsticals = []
 running = True
 restart = False
 steps_count = 0
 states = []
 actions = []
 episodes = []
 global max_score
 rewards = tf.constant([], dtype=tf.float32)
 font = pygame.font.SysFont('arial', 20)
 while running and not restart:
  score += 1e-2
  if score > max_score:
   max_score = score
  for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
  #take action
  action = 0
  old_state = []
  if(len(obsticals) > 0):
   old_state = get_state(player_pos=player_pos,obsticals=obsticals)
   action = call_model(model=pg_model,state=old_state)
  #move player
  player_pos = move_player(player_pos=player_pos,action=action)
  #Obsticals update
  obsticals = update_obsticals(obsticals=obsticals)
  #Collision detection 
  restart = check_collisions(player_pos=player_pos,obsticals=obsticals)
  render(player_pos=player_pos,obsticals=obsticals,screen=screen)
 
  
  new_state = get_state(player_pos=player_pos,obsticals=obsticals)
  if(len(old_state) > 0):
   actions.append(action)
   states.append(old_state)
   reward = pg_model.compute_reward(state=new_state,action=action,obsticals_count=MAX_OBSTICALS)
   text_surface = font.render( "immediate reward " + str(int(reward)), True, (0, 255, 0)) 
   screen.blit(text_surface,(10,10))
   text_surface = font.render( "score " + str(int(score)), True, (0, 255, 0)) 
   screen.blit(text_surface,(10,30))
   text_surface = font.render( "max score " + str(int(max_score)), True, (0, 255, 0)) 
   screen.blit(text_surface,(10,50))
   reward =  tf.constant([reward ],dtype=tf.float32)
   rewards = tf.concat([rewards, reward], axis=0)
   steps_count += 1
   pygame.display.flip()
   if(steps_count >= 100):
    #pg_model.update(states=states,actions=actions,rewards=rewards)
    episodes.append({'states':states,'actions':actions,'rewards':rewards})
    steps_count = 0
    states = []
    actions = []
    rewards = tf.constant([], dtype=tf.float32)
 if not running:
  pygame.quit()
 elif restart :
     wandb.log({"score":score})
     old_model = pg_model.model
     episodes.append({'states':states,'actions':actions,'rewards':rewards})
     for episode in episodes:
      pg_model.update(states=episode['states'],actions=episode['actions'],rewards=episode['rewards'],old_model=old_model)
     episodes = []
     start_game()


pygame.init()
start_game()