import pygame
from cellitaire.environment.ui.ui_element_constants import *
from cellitaire.environment.ui.event_types import *


class StockpileSprite(pygame.sprite.Sprite):
    def __init__(self, height, width, x, y):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.height = height
        self.width = width

        self.card = None
        self.count = 0

        self.draw_stockpile()

    def draw_stockpile_outline(self):
        pygame.draw.rect(
            self.image,
            YELLOW,
            pygame.Rect(
                0,
                0,
                self.width // 2,
                self.height
            ),
            STOCKPILE_PADDING
        )

    def draw_background(self):
        pygame.draw.rect(
            self.image,
            STOCKPILE_BACKGROUND_COLOR,
            pygame.Rect(
                0,
                0,
                self.width,
                self.height))

    def draw_card(self):
        if self.card is None:
            return

        card_rect = pygame.draw.rect(
            self.image,
            WHITE,
            pygame.Rect(
                STOCKPILE_PADDING,
                STOCKPILE_PADDING,
                (self.width // 2) - 2 * STOCKPILE_PADDING,
                self.height - 2 * STOCKPILE_PADDING
            )
        )

        font = pygame.font.SysFont('arial', 24)
        card_text = str(self.card)
        text_surface = font.render(
            card_text, True, get_card_text_color(
                self.card))
        text_rect = text_surface.get_rect(center=card_rect.center)
        self.image.blit(text_surface, text_rect)

    def draw_count(self):
        font = pygame.font.Font(None, 24)
        card_text = str(self.count)
        text_surface = font.render(card_text, True, WHITE)
        text_rect = text_surface.get_rect(
            center=(3 * self.width // 4, self.height // 2))
        self.image.blit(text_surface, text_rect)

    def draw_stockpile(self):
        self.draw_background()
        self.draw_stockpile_outline()
        self.draw_card()
        self.draw_count()

    def handle_stockpile_update_event(self, event):
        self.card = event.top_card
        self.count = event.count

    def update(self, events):
        for event in events:
            if event.type == GU_STOCKPILE_UPDATE:
                self.handle_stockpile_update_event(event)

        self.draw_stockpile()
