# This is for rendering the entire board


from cellitaire.game.card import Card
from cellitaire.game.game import Game


def test_render_new_game():
    game = Game()
    game.new_game(7,12,6)
    print(game.render())
    assert False

def test_render_card():
    card = Card(3)
    card.render()
    print("\n".join(card.render()))
    assert False