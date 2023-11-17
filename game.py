
import pygame
import math
import random

def start_game():
 MAX_OBSTICALS = 5
 PLAYER_POS_RANGE = (20,480) 
 OBSITCALS_X_POS_RANGE = 500
 OBSTICAL_MAX_Y_POS = 600
 OBSTICAL_MAX_SPEED = 0.2
 OBSTICAL_MIN_SPEED = 0.05
 OBSTICAL_SIZE = 25
 PLAYER_SIZE = 15
 PLAYER_SPEED = .3
 SCREEN = pygame.display.set_mode([500, 500])
 player_pos = [200,450]
 obsticals = []
 running = True
 restart = False
 while running and not restart:
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
    
 if not running:
  pygame.quit()
 elif restart :
     start_game()


pygame.init()
start_game()