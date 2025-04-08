import pygame
from cellitaire.environment.ui.ui_element_constants import *
from cellitaire.environment.ui.event_types import *


class RewardSprite(pygame.sprite.Sprite):
    def __init__(self, height, width, x, y):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.height = height
        self.width = width

        self.total_reward = 0

    def draw_reward(self):
        r = pygame.draw.rect(
            self.image, BLACK, pygame.Rect(
                0, 0, self.width, self.height))

        font = pygame.font.Font(None, 24)
        score_text = f'Reward: {str(self.total_reward)}'
        text_surface = font.render(score_text, True, WHITE)
        text_rect = text_surface.get_rect(center=r.center)
        self.image.blit(text_surface, text_rect)

    def hadle_reward_update_event(self, event):
        self.total_reward += event.reward

    def update(self, events):
        for event in events:
            if event.type == REWARD_UPDATED:
                self.hadle_reward_update_event(event)
            elif event.type == RESET:
                self.total_reward = 0

        self.draw_reward()
