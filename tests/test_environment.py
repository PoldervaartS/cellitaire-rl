from cellitaire.environment.cellitaire_env import CellitaireEnv
from cellitaire.game.game import Game
from cellitaire.environment.rewards.reward import get_stockpile_cards, get_cards_in_foundation

def test_init_doesnt_crash():
  env = CellitaireEnv(None)
  assert True

def test_state_translator():
  rows = 7
  cols = 12
  num_reserved = 6
  env = CellitaireEnv(None, rows=rows, cols=cols, num_reserved=num_reserved)
  env.reset()
  state = env.get_state()

  actual_stockpile_cards = get_stockpile_cards(state, rows, cols, num_reserved)
  actual_cards_in_foundation = get_cards_in_foundation(state, rows, cols)

  assert env.game.foundation.total_cards() == actual_cards_in_foundation
  assert env.game.stockpile.count() == actual_stockpile_cards
