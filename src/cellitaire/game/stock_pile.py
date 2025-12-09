import random

from cellitaire.game.card import Card, UNKNOWN_CARD_ID


class StockPile:
    def __init__(self, cards):
        """
        Initializes the stock pile with a list of cards.

        :param cards: A list of Card objects representing the stock pile.
        """
        self.cards = cards
        self.initialize_known_cards()

    @classmethod
    def create_stock_pile(cls, reserve_n: int):
        """
        Creates a stock pile from a shuffled deck of 52 cards while reserving
        the top n cards for the board.

        :param reserve_n: The number of cards to reserve for the board.
        :return: A tuple (stock_pile, reserved_cards)
                 where stock_pile is an instance of StockPile containing the remaining cards,
                 and reserved_cards is a list of Card objects that were removed from the top.
        """
        # Generate a full deck of 52 cards (ids 1 to 52)
        deck = [Card(i) for i in range(1, 53)]
        random.shuffle(deck)
        # Reserve the top n cards for the board.
        reserved_cards = deck[:reserve_n]
        # The stock pile consists of the remaining cards.
        stock_cards = deck[reserve_n:]
        return cls(stock_cards), reserved_cards

    def top_card(self) -> Card:
        """
        Returns the top card of the stock pile without removing it.

        :return: The Card object at the top, or None if the pile is empty.
        """
        if self.cards:
            return self.cards[0]
        return None

    def draw_top_card(self):
        """
        Removes and returns the top card from the stock pile.

        :return: The Card object that was drawn, or None if the pile is empty.
        """
        if self.cards:
            self.known_cards.pop(0)
            return self.cards.pop(0)
        return None

    def place_card_bottom(self, card):
        """
        Places a card at the bottom of the stock pile.

        :param card: The Card object to be placed.
        """
        self.cards.append(card)
        self.known_cards.append(card)

    def count(self):
        """
        Returns the number of cards currently in the stock pile.

        :return: An integer count of cards.
        """
        return len(self.cards)
    
    def initialize_known_cards(self):
        self.known_cards = [self.cards[0]] + [Card(UNKNOWN_CARD_ID) for card in range(len(self.cards) - 1)]

    def get_stockpile_state(self):
        return [card.card_id for card in self.known_cards] + [Card(0).card_id] * (52 - len(self.known_cards))
    
    def __str__(self):
        return f"StockPile({self.count()} cards)"

    def __repr__(self):
        return self.__str__()

    def render(self):
        return "\n".join(self.top_card().render()) + f" Count: {self.count()}"
