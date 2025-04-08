from cellitaire.game.slot import Slot


class Board:
    def __init__(self, rows: int, cols: int, initial_cards=None):
        """
        Initializes the board with the given number of rows and columns.
        Each board cell is a Slot object stored in a matrix (list of lists).
        Optionally, an array of initial cards can be provided to be placed
        in a single row at the center of the board.

        :param rows: Number of rows on the board (assumed to be odd).
        :param cols: Number of columns on the board.
        :param initial_cards: Optional list of card objects to initialize the board.
        """
        self.rows = rows
        self.cols = cols
        self.slots = [[Slot() for _ in range(cols)] for _ in range(rows)]

        # Assign coordinates to each slot for easy reference.
        for r in range(rows):
            for c in range(cols):
                self.slots[r][c].coordinate = (r, c)

        if initial_cards is not None:
            # Determine the center row (since rows is odd, this is
            # unambiguous).
            center_row = rows // 2
            num_cards = len(initial_cards)
            # Center the initial cards horizontally.
            start_col = (cols - num_cards) // 2
            placed_coords = []

            # Force place each initial card in the center row.
            for i, card in enumerate(initial_cards):
                coord = (center_row, start_col + i)
                self.slots[center_row][start_col + i].force_place_card(card)
                placed_coords.append(coord)

            # Collect all coordinates that are either occupied or adjacent to
            # an occupied slot.
            all_coords_to_touch = set(placed_coords)
            for coord in placed_coords:
                for neighbor in self.get_neighbors(coord):
                    # Only add empty neighbor slots.
                    if not self.slots[neighbor[0]][neighbor[1]].has_card():
                        all_coords_to_touch.add(neighbor)

            # Touch every slot in the collected set to update its status.
            for coord in all_coords_to_touch:
                self.touch(coord)

    def get_neighbors(self, coordinate: tuple) -> list:
        """
        Given a coordinate (row, col), returns a list of valid neighboring coordinates.
        Neighbors include horizontal, vertical, and diagonal adjacent slots.
        """
        r, c = coordinate
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the slot itself.
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def touch(self, coordinate: tuple):
        """
        Updates the status of the slot at the given coordinate based on the number of neighboring cards.
        Calls the slot's update_status method with the count of occupied neighboring slots.
        """
        neighbors = self.get_neighbors(coordinate)
        count = sum(1 for nr, nc in neighbors if self.slots[nr][nc].has_card())
        self.slots[coordinate[0]][coordinate[1]].update_status(count)

    def place_card(self, coordinate: tuple, card):
        """
        Places a card at the given coordinate.
        Adds the card to the slot (using the slot's add_card method), then touches that slot
        and all surrounding slots to update their status.
        :param coordinate: A tuple (row, col) indicating where to place the card.
        :param card: The card object to place.
        """
        r, c = coordinate
        self.slots[r][c].add_card(card)
        self.touch(coordinate)
        for neighbor in self.get_neighbors(coordinate):
            self.touch(neighbor)

    def remove_card(self, coordinate: tuple):
        """
        Removes the card from the slot at the given coordinate.
        After removal, touches that slot and all surrounding slots.
        :param coordinate: A tuple (row, col) indicating from where to remove the card.
        :return: The removed card.
        """
        r, c = coordinate
        removed = self.slots[r][c].remove_card()
        self.touch(coordinate)
        for neighbor in self.get_neighbors(coordinate):
            self.touch(neighbor)
        return removed

    def get_special_slots(self) -> tuple:
        """
        Retrieves two lists:
          1. Coordinates where a slot is occupied and the card is either lonely or suffocated.
          2. Coordinates where a slot is empty and a card can be placed.
        :return: A tuple (special_coords, placeable_coords)
        """
        special_coords = [slot.coordinate for row in self.slots for slot in row
                          if slot.has_card() and (slot.is_lonely or slot.is_suffocated)]
        placeable_coords = [slot.coordinate for row in self.slots for slot in row
                            if not slot.has_card() and slot.is_placeable]
        return special_coords, placeable_coords

    def get_lonely_coords(self) -> list:
        return [slot.coordinate for row in self.slots for slot in row if slot.has_card(
        ) and slot.is_lonely]

    def get_suffocated_coords(self) -> list:
        return [slot.coordinate for row in self.slots for slot in row if slot.has_card(
        ) and slot.is_suffocated]

    def get_suffocated_or_lonely_coords(self) -> list:
        """
        Retrieves a list of coordinates for slots that are occupied and where the card is either lonely or suffocated.

        :return: A list of (row, col) tuples.
        """
        return [slot.coordinate for row in self.slots for slot in row
                if slot.has_card() and (slot.is_lonely or slot.is_suffocated)]

    def get_placeable_coords(self) -> list:
        """
        Retrieves a list of coordinates where a card can be placed.
        A card can be placed if the slot is empty and is marked as placeable.

        :return: A list of (row, col) tuples.
        """
        return [slot.coordinate for row in self.slots for slot in row
                if not slot.has_card() and slot.is_placeable]

    def any_moves_possible(self) -> bool:
        """
        Indicates whether any moves remain on the board.
        A move is possible if either:
          - There is an empty slot where a card can be placed.
          - There is an occupied slot where the card is lonely or suffocated.
        :return: True if at least one move is possible, False otherwise.
        """
        special_coords, placeable_coords = self.get_special_slots()
        return len(special_coords) > 0 or len(placeable_coords) > 0

    def has_card_at(self, coordinate: tuple) -> bool:
        """
        Checks if the slot at the given coordinate has a card.

        :param coordinate: A tuple (row, col) indicating the slot to check.
        :return: True if the slot contains a card, False otherwise.
        """
        r, c = coordinate
        return self.slots[r][c].has_card()

    def can_change_slot(self, coordinate: tuple) -> bool:
        """
        Determines if the slot at the given coordinate can be changed.
        A slot can be changed if:
          - It is occupied and the card is lonely or suffocated.
          OR
          - It is empty and a card can be placed in it (i.e. is marked as placeable).

        :param coordinate: A tuple (row, col) indicating the slot to check.
        :return: True if the slot can be changed, False otherwise.
        """
        r, c = coordinate
        slot = self.slots[r][c]
        if slot.has_card():
            return slot.is_lonely or slot.is_suffocated
        else:
            return slot.is_placeable

    def __str__(self):
        """
        Returns a simple string representation of the board for debugging.
        """
        s = ""
        for row in self.slots:
            s += " | ".join(str(slot) for slot in row) + "\n"
        return s

    def __repr__(self):
        return self.__str__()
