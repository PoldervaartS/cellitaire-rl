class Card:
    # Define the order of suits and ranks.
    # We'll use lowercase letters for suits: 's' = spades, 'h' = hearts, 'd' = diamonds, 'c' = clubs.
    SUITS = ['s', 'h', 'd', 'c']
    RANKS = ['a', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k']

    def __init__(self, card_id: int):
        """
        Initialize a card given its unique integer id.
        """
        if not 0 < card_id <= 52:
            raise ValueError("card_id must be between 1 and 52 or 0 for 'blank'")
        self.card_id = card_id
        # Convert the id into a zero-indexed value for calculations.
        card_index = card_id - 1
        suit_index = card_index // 13  # integer division determines suit group
        rank_index = card_index % 13   # remainder gives the rank within the suit
        self.suit = Card.SUITS[suit_index]
        self.rank = Card.RANKS[rank_index]

    @classmethod
    def from_string(cls, card_str: str):
        """
        Create a card from a string representation.
        For example:
          - 'as' represents Ace of spades,
          - '10c' represents 10 of clubs,
          - 'Kd' represents King of diamonds.
        """
        if len(card_str) < 2:
            raise ValueError("Card string is too short.")

        # Assume the last character represents the suit.
        suit = card_str[-1].lower()
        # The rest of the string represents the rank.
        rank = card_str[:-1].lower()

        if suit not in cls.SUITS:
            raise ValueError(f"Invalid suit '{suit}' in card string.")
        # Normalize the ranks in our lookup by lowercasing them.
        normalized_ranks = [r.lower() for r in cls.RANKS]
        if rank not in normalized_ranks:
            raise ValueError(f"Invalid rank '{rank}' in card string.")
        rank_index = normalized_ranks.index(rank)
        suit_index = cls.SUITS.index(suit)
        # Calculate the card id using our ordering.
        card_id = suit_index * 13 + rank_index + 1
        return cls(card_id)

    @classmethod
    def from_id(cls, card_id: int):
        """
        Create a card from its integer id.
        """
        return cls(card_id)
    
    @classmethod
    def suit_to_ascii(cls, suit:str):
        match suit:
            case 's':
                return '♠'
            case 'h':
                return '♥'
            case 'd':
                return '♦'
            case 'c':
                return '♣'

    def __str__(self):
        """
        Return a simple string representation like 'as' or '10c'.
        """
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        """
        More detailed representation for debugging.
        """
        return f"Card({self.card_id}: {self.rank}{self.suit})"

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.card_id == other.card_id
        return False

    def __hash__(self):
        return hash(self.card_id)
    
    def render(self):
        # TODO color for the card. Will need to pass in the status from above.
        card_length = 6
        card_height = 6
        # if it is 0 then it is an empty card
        array_strings_out = []
        for i in range(card_height):
            if i == 0:
                array_strings_out.append(f""" {'_' * card_length} """)
            elif i == card_height//2:
                if self.card_id == 0:
                    str_out = f"""|{' ' * (card_length//2 - 1)}  {' ' * (card_length//2 - 1)}|"""
                # convert the suit to the ASCII
                else:
                    str_out = f"""|{' ' * (card_length//2 - 1)}{self.rank}{Card.suit_to_ascii(self.suit)}{' ' * (card_length//2 - 1)}|"""
                array_strings_out.append(str_out)
            elif i == card_height - 1:
                array_strings_out.append(f"""|{'_' * card_length}|""")
            else:
                array_strings_out.append(f"""|{' ' * card_length}|""")
        return array_strings_out
