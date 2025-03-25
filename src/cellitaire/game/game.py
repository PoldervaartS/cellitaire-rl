from cellitaire.game.board import Board
from cellitaire.game.stock_pile import StockPile
from cellitaire.game.foundation import Foundation

class Game:
    def __init__(self):
        self.board = None
        self.stockpile = None
        self.foundation = None
        self.moves = []  # List to track moves (each move is recorded as a coordinate tuple)

    def new_game(self, rows: int, cols: int, initial_reserved: int):
        """
        Creates a new game with the specified board dimensions and a number of reserved cards.
        - Creates a stockpile from a shuffled deck, reserving the top `initial_reserved` cards.
        - Creates a board using the reserved cards (placed in a centered row).
        - Creates a new foundation.
        
        :param rows: Number of rows for the board (assumed to be odd).
        :param cols: Number of columns for the board.
        :param initial_reserved: Number of cards to reserve for board initialization.
        """
        # Create stockpile and reserve initial cards for the board.
        self.stockpile, reserved_cards = StockPile.create_stock_pile(initial_reserved)
        # Initialize board with reserved cards placed in its center row.
        self.board = Board(rows, cols, initial_cards=reserved_cards)
        # Create a new foundation.
        self.foundation = Foundation()
        # Reset move list.
        self.moves = []

    def make_move(self, coordinate: tuple) -> bool:
        """
        Executes a move at the given coordinate, based on the following logic:
        
          1. Validate that the slot at 'coordinate' can be changed (using board.can_change_slot).
          2. If the slot is occupied:
             a. Remove the card from the board.
             b. If the card can be placed on the foundation (foundation.can_place(card) returns True),
                add it to the foundation.
             c. Otherwise, place it at the bottom of the stockpile.
          3. If the slot is empty and changeable, and if there is a card in the stockpile,
             draw the top card from the stockpile and place it into the slot.
          4. Otherwise, the move is invalid.
        
        If the move is successful, the coordinate is recorded in the move list.
        
        :param coordinate: A tuple (row, col) indicating the target slot.
        :return: True if the move was valid and executed; False otherwise.
        """
        if not self.board.can_change_slot(coordinate):
            return False  # Slot is not eligible for change

        if self.board.has_card_at(coordinate):
            # A card is present; remove it.
            card = self.board.remove_card(coordinate)
            # Try to place the card on the foundation.
            if self.foundation.can_place(card):
                self.foundation.add_card(card)
            else:
                # If not, place the card at the bottom of the stockpile.
                self.stockpile.place_card_bottom(card)
            self.moves.append(coordinate)
            return True
        else:
            # The slot is empty; if there's a card in the stockpile, place it.
            if self.stockpile.top_card() is not None:
                card = self.stockpile.draw_top_card()
                self.board.place_card(coordinate, card)
                self.moves.append(coordinate)
                return True
            else:
                # No card available from the stockpile.
                return False

    def possible_moves_remaining(self) -> bool:
        """
        Determines whether any moves are still possible.
        A move is possible if at least one slot on the board can be changed or if there are
        cards remaining in the stockpile.
        
        :return: True if moves are possible; False otherwise.
        """
        return (len(self.board.get_placeable_coords()) > 0 and self.stockpile.count() > 0) or len(self.board.get_suffocated_or_lonely_coords()) > 0

    def foundation_card_count(self) -> int:
        """
        Retrieves the total number of cards in the foundation by calling the foundation's total_cards method.
        
        :return: Total number of cards in the foundation.
        """
        return self.foundation.total_cards()

    def total_moves_played(self) -> int:
        """
        Returns the total number of moves that have been played.
        
        :return: The number of moves recorded.
        """
        return len(self.moves)

    def is_move_possible_at(self, coordinate: tuple) -> bool:
        """
        Checks if a move at the given coordinate is possible.
        A move is possible if the slot at the coordinate can be changed.
        
        :param coordinate: A tuple (row, col) indicating the target slot.
        :return: True if the move is possible; False otherwise.
        """
        return self.board.can_change_slot(coordinate)

    def get_possible_lonely_suffocated_coords(self) -> list:
        """
        Retrieves a list of coordinates where slots are occupied and the card is either lonely or suffocated.
        
        :return: A list of (row, col) tuples.
        """
        return self.board.get_suffocated_or_lonely_coords()

    def get_possible_placeable_coords(self) -> list:
        """
        Retrieves a list of coordinates where a card can be placed.
        A card can be placed if the slot is empty and marked as placeable.
        
        :return: A list of (row, col) tuples.
        """
        return self.board.get_placeable_coords()

    def __str__(self):
        return (
            f"Game State:\n"
            f"Board:\n{self.board}\n"
            f"StockPile: {self.stockpile}\n"
            f"Foundation: {self.foundation}\n"
            f"Moves Played: {self.total_moves_played()}\n"
            f"Total Cards in Foundation: {self.foundation_card_count()}"
        )
    
    def render(self):
        """Print an Ascii Image of the current game state"""

        # TODO render board, draw lines below foundation/basically whole board. 
        return f"""{self.stockpile.render()}\n{self.foundation.render()}"""
