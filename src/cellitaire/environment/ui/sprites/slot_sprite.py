import pygame
from cellitaire.environment.ui.ui_element_constants import *
from cellitaire.environment.ui.event_types import *


class SlotSprite(pygame.sprite.Sprite):
    def __init__(self, height, width, x, y, coordinate):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.height = height
        self.width = width

        self.coordinate = coordinate

        self.card = None
        self.is_lonely = False
        self.is_suffocated = False
        self.is_placeable = False
        self.is_hovered = False

        self.draw_slot()

    def draw_background(self):
        pygame.draw.rect(
            self.image,
            SLOT_BACKGROUND_COLOR,
            pygame.Rect(
                0,
                0,
                self.width,
                self.height))

    def draw_outline(self):
        if not (self.is_lonely or self.is_suffocated or self.is_placeable):
            return

        outline_color = SLOT_LONELY_OR_SUFFOCATED_COLOR
        if self.is_placeable:
            outline_color = SLOT_PLACEABLE_COLOR

        pygame.draw.rect(
            self.image,
            outline_color,
            pygame.Rect(
                0,
                0,
                self.width,
                self.height
            ),
            SLOT_PADDING
        )

    def draw_card(self):
        if self.card is None:
            return
        card_rect = pygame.draw.rect(
            self.image,
            WHITE,
            pygame.Rect(
                SLOT_PADDING,
                SLOT_PADDING,
                self.width - 2 * SLOT_PADDING,
                self.height - 2 * SLOT_PADDING
            )
        )

        font = pygame.font.Font(None, 24)
        card_text = str(self.card)
        text_surface = font.render(
            card_text, True, get_card_text_color(
                self.card))
        text_rect = text_surface.get_rect(center=card_rect.center)
        self.image.blit(text_surface, text_rect)

    def draw_hover_overlay(self):
        if not self.is_hovered:
            return
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 50))
        self.image.blit(overlay, (0, 0))
        self.rect = self.image.get_rect(topleft=self.rect.topleft)

    def draw_slot(self):
        self.draw_background()
        self.draw_card()
        self.draw_outline()
        self.draw_hover_overlay()

    def handle_slot_update_event(self, event):
        self.card = event.card
        self.is_lonely = event.is_lonenly
        self.is_suffocated = event.is_suffocated
        self.is_placeable = event.is_placeable

    def handle_clicked(self):
        if not (self.is_lonely or self.is_suffocated or self.is_placeable):
            return
        pygame.event.post(
            pygame.event.Event(
                SLOT_CLICKED,
                coordinate=self.coordinate))

    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            self.is_hovered = True
        else:
            self.is_hovered = False

        for event in events:
            if event.type == GU_SLOT_UPDATE and event.coordinate == self.coordinate:
                self.handle_slot_update_event(event)
            if self.is_hovered and event.type == pygame.MOUSEBUTTONUP:
                self.handle_clicked()

        self.draw_slot()
