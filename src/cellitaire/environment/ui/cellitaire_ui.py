import pygame
import threading
from cellitaire.environment.ui.ui_element_constants import *
from cellitaire.environment.ui.sprites.foundation_sprite import FoundationSprite
from cellitaire.environment.ui.sprites.reward_sprite import RewardSprite
from cellitaire.environment.ui.sprites.slot_sprite import SlotSprite
from cellitaire.environment.ui.sprites.stockpile_sprite import StockpileSprite
from cellitaire.environment.ui.event_types import *
import queue

FRAME_RATE = 30

class CellitaireUI(threading.Thread):
    def __init__(self, rows=7, cols=12):
        super().__init__()

        pygame.init()

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(StockpileSprite(
            height=STOCKPILE_HEIGHT,
            width=STOCKPILE_WIDTH,
            x=SCREEN_MARGIN,
            y=SCREEN_MARGIN
        ))
    
        self.all_sprites.add(FoundationSprite(
            height=FOUNDATION_HEIGHT,
            width=FOUNDATION_WIDTH,
            x=SCREEN_MARGIN + SLOT_WIDTH * cols - FOUNDATION_WIDTH,
            y=SCREEN_MARGIN
        ))
    
        self.all_sprites.add(RewardSprite(
            height=SLOT_HEIGHT,
            width=SLOT_WIDTH,
            x=SCREEN_MARGIN + SLOT_WIDTH * cols - FOUNDATION_WIDTH - 2 * SLOT_WIDTH,
            y=SCREEN_MARGIN
        ))
    
        for i in range(rows):
          for j in range(cols):
              self.all_sprites.add(
                  SlotSprite(
                      height=SLOT_HEIGHT,
                      width=SLOT_WIDTH, 
                      x=SCREEN_MARGIN + j * SLOT_WIDTH, 
                      y=SCREEN_MARGIN + STOCKPILE_HEIGHT + BOARD_MARGIN + i * SLOT_HEIGHT,
                      coordinate=(i, j)
                  )
              )
    
        self.rows = rows
        self.cols = cols
    
        self.clock = pygame.time.Clock()
        self.running = True
        self.screen = pygame.display.set_mode(SCREEN_DIMS)
        pygame.display.set_caption("Cellitaire RL")
    
        self.event_queue = queue.Queue()

    def add_events(self, events):
        for event in events:
            self.event_queue.put(event)

    def _step(self, events):
        self.all_sprites.update(events)
    
    def _draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FRAME_RATE)

    def run(self):
        while self.running:
            local_events = []
            try:
                while True:
                    local_events.append(self.event_queue.get_nowait())
            except queue.Empty:
                pass
            for event in local_events:
                if event.type == pygame.QUIT:
                    self.running = False
            self._step(local_events)
            self._draw()

    def kill(self):
        self.running = False
        pygame.quit()
        CellitaireUI.instantiated = False