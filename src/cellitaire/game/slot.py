class Slot:
    def __init__(self, card=None):
        """
        Initialize a Slot.

        :param card: Optional initial card to place in the slot.
        """
        self.card = card
        self.is_lonely = False
        self.is_suffocated = False
        # Indicates if a card can be placed (only applicable when empty)
        self.is_placeable = False

    def update_status(self, num_neighbors: int):
        """
        Updates the slot's status based on the number of neighbors.

        For a filled slot:
          - The card is lonely if it has fewer than 2 neighbors.
          - The card is suffocated if it has more than 3 neighbors.
          - A card cannot be placed in a slot that already has a card.

        For an empty slot:
          - The slot is considered placeable if and only if it has exactly 3 neighbors.
          - Loneliness and suffocation do not apply to empty slots.

        :param num_neighbors: The number of adjacent (horizontal, vertical, diagonal) neighbors.
        """
        if self.has_card():
            # For a filled slot, update loneliness and suffocation
            self.is_lonely = num_neighbors < 2
            self.is_suffocated = num_neighbors > 3
            self.is_placeable = False  # Cannot place a card on a filled slot.
        else:
            # For an empty slot, clear any previous status and set
            # placeability.
            self.clear_status()
            self.is_placeable = (num_neighbors == 3)

    def add_card(self, card):
        """
        Places a card into the slot.
        A card can only be added if the slot is empty and is marked as placeable.

        :param card: The card object to add.
        :raises ValueError: If the slot is already filled or not placeable.
        """
        if self.has_card():
            raise ValueError("Slot already has a card.")
        if not self.is_placeable:
            raise ValueError(
                "Card cannot be placed in this slot due to neighbor constraints.")
        self.card = card

    def force_place_card(self, card):
        """
        Force places a card into the slot without checking placement constraints.
        This is useful during board initialization when you want to set up the board
        regardless of whether the slot is normally placeable.

        After forcing, the slot is considered occupied, and placement restrictions no longer apply.

        :param card: The card object to place.
        """
        self.card = card
        # Clear any previous status flags since the slot is now occupied.
        self.clear_status()
        # Once a card is forced into the slot, mark it as not placeable.
        self.is_placeable = False

    def remove_card(self):
        """
        Removes the card from the slot.
        Also clears the slot's status.

        :return: The removed card, or None if the slot was empty.
        """
        removed = self.card
        self.card = None
        self.clear_status()  # Clear lonely/suffocated flags.
        # Reset placeable flag (to be updated externally based on neighbor
        # count).
        self.is_placeable = False
        return removed

    def has_card(self):
        """
        Checks if there is a card in the slot.

        :return: True if a card is present, False otherwise.
        """
        return self.card is not None

    def clear_status(self):
        """
        Resets the slot's lonely and suffocated status.
        """
        self.is_lonely = False
        self.is_suffocated = False

    def can_place_card(self):
        """
        Determines if a card can be placed in the slot.
        A card can be placed only if the slot is empty and is marked as placeable.

        :return: True if a card can be placed, False otherwise.
        """
        return not self.has_card() and self.is_placeable

    def __str__(self):
        card_str = str(self.card) if self.card is not None else "Empty"
        if self.has_card():
            status = []
            if self.is_lonely:
                status.append("Lonely")
            if self.is_suffocated:
                status.append("Suffocated")
            status_str = ", ".join(status) if status else "Normal"
        else:
            status_str = "Placeable" if self.is_placeable else "Not Placeable"
        return f"Slot({card_str}) - {status_str}"

    def __repr__(self):
        return self.__str__()
