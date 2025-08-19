import pygame
from cellitaire.environment.ui.ui_element_constants import *
from cellitaire.environment.ui.event_types import *


class FoundationSprite(pygame.sprite.Sprite):
    def __init__(self, height, width, x, y):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.height = height
        self.width = width

        self.foundation_dict = {
            's': None,
            'h': None,
            'd': None,
            'c': None
        }
        self.total_saved = 0
        self.draw_foundation()

    def draw_background(self):
        pygame.draw.rect(
            self.image,
            FOUNDATION_BACKGROUND_COLOR,
            pygame.Rect(
                0,
                0,
                self.width,
                self.height))

    def draw_card(self, x_offset, card):
        if card is None:
            return
        card_rect = pygame.draw.rect(
            self.image,
            WHITE,
            pygame.Rect(
                x_offset + FOUNDATION_PADDING,
                FOUNDATION_PADDING,
                self.width // 5 - 2 * FOUNDATION_PADDING,
                self.height - 2 * FOUNDATION_PADDING
            )
        )

        font = pygame.font.SysFont('arial', 24)
        card_text = str(card)
        text_surface = font.render(card_text, True, get_card_text_color(card))
        text_rect = text_surface.get_rect(center=card_rect.center)
        self.image.blit(text_surface, text_rect)

    def draw_pile_outline(self, x_offset):
        pygame.draw.rect(
            self.image,
            FOUNDATION_PILE_OUTLINE_COLOR,
            pygame.Rect(
                x_offset,
                0,
                self.width // 5,
                self.height
            ),
            FOUNDATION_PADDING
        )

    def draw_cards(self):
        x_offset = self.width // 5
        step = x_offset
        for suit, card in self.foundation_dict.items():
            self.draw_pile_outline(x_offset)
            self.draw_card(x_offset, card)
            x_offset += step

    def draw_total_saved(self):
        font = pygame.font.Font(None, 24)
        card_text = str(self.total_saved)
        text_surface = font.render(card_text, True, WHITE)
        text_rect = text_surface.get_rect(
            center=(self.width // 10, self.height // 2))
        self.image.blit(text_surface, text_rect)

    def draw_foundation(self):
        self.draw_background()
        self.draw_cards()
        self.draw_total_saved()

    def handle_foundation_update_event(self, event):
        self.foundation_dict = event.foundation_dict
        self.total_saved = event.total_saved

    def update(self, events):
        for event in events:
            if event.type == GU_FOUNDATION_UPDATE:
                self.handle_foundation_update_event(event)

        self.draw_foundation()
