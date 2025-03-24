from game.card import Card  # Assumes your Card class is defined in card.py

class Foundation:
    def __init__(self):
        """
        Initialize the foundation.
        Each suit (from Card.SUITS) maps to the current highest card placed in that suit.
        Initially, all are empty (None).
        """
        # Create a mapping for each suit, initializing with None.
        self.foundation = {suit: None for suit in Card.SUITS}

    def can_place(self, card: Card) -> bool:
        """
        Determines if a card can be placed on the foundation according to the rules.
        
        Rules:
         - If the foundation for a suit is empty, only an Ace ('a') can be placed.
         - Otherwise, the card's rank must be exactly one step higher than the current card.
        
        :param card: The Card object to be placed.
        :return: True if the card can be placed, False otherwise.
        """
        current = self.foundation.get(card.suit)
        rank_order = Card.RANKS  # e.g., ['a', '2', ..., '10', 'j', 'q', 'k']
        
        if current is None:
            # Only an Ace may be placed if the foundation is empty.
            return card.rank == 'a'
        else:
            # The card to be placed must be exactly one rank higher than the current card.
            current_index = rank_order.index(current.rank)
            if current_index + 1 < len(rank_order):
                return card.rank == rank_order[current_index + 1]
            else:
                # The current card is a King; no further card can be added.
                return False

    def add_card(self, card: Card) -> bool:
        """
        Attempts to add a card to the foundation.
        
        :param card: The Card object to add.
        :return: True if the card was successfully added, False otherwise.
        """
        if self.can_place(card):
            self.foundation[card.suit] = card
            return True
        return False

    def get_current_card(self, suit: str):
        """
        Returns the current highest card in the foundation for a given suit.
        
        :param suit: A string representing the suit (e.g., 's', 'h', 'd', 'c').
        :return: The Card object at the top of the foundation for that suit, or None if empty.
        """
        return self.foundation.get(suit)
    
    def total_cards(self) -> int:
        """
        Calculates and returns the total number of cards in the foundation piles.
        For each suit, if a card is present, its position (index in Card.RANKS + 1)
        is taken as the number of cards in that pile.
        
        :return: The total number of cards across all foundation piles.
        """
        total = 0
        for suit in Card.SUITS:
            card = self.foundation.get(suit)
            if card is not None:
                # The count for the suit is the index in Card.RANKS plus one.
                total += Card.RANKS.index(card.rank) + 1
        return total


    def __str__(self):
        """
        Returns a human-readable string representing the foundation status.
        """
        foundation_str = []
        for suit in Card.SUITS:
            card = self.foundation[suit]
            if card is None:
                foundation_str.append(f"{suit.upper()}: Empty")
            else:
                foundation_str.append(f"{suit.upper()}: {card}")
        return " | ".join(foundation_str)

    def __repr__(self):
        return self.__str__()
