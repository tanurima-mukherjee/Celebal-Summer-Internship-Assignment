class Node:
    # A Node class to represent each node in the list.

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    # A LinkedList class to manage the nodes.

    def __init__(self):
        self.head = None

    def add_to_end(self, data):
        # Add a node to the end of the list.
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            print(f"Added {data} as the head node.")
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            print(f"Added {data} to the end of the list.")

    def print_list(self):
        # Prints all the elements in the list.
        if self.head is None:
            print("The list is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        # Delete the nth node (where n is a 1-based index).
        try:
            if self.head is None:
                raise IndexError("Cannot delete from an empty list.")

            if n <= 0:
                raise IndexError("Index should be a positive integer (1-based index).")

            if n == 1:
                deleted_data = self.head.data
                self.head = self.head.next
                print(f"Deleted node at position {n} with value {deleted_data}.")
                return

            current = self.head
            count = 1
            while current and count < n - 1:
                current = current.next
                count += 1

            if current is None or current.next is None:
                raise IndexError("Index out of range.")

            deleted_data = current.next.data
            current.next = current.next.next
            print(f"Deleted node at position {n} with value {deleted_data}.")

        except IndexError as e:
            print("Error:", e)


# Main program with user input
if __name__ == "__main__":
    ll = LinkedList()

    while True:
        print("\n--- Singly Linked Operations With OOPS---")
        print("1. Add element(s) to end")
        print("2. Print list")
        print("3. Delete nth node")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            user_input = input("Enter values to add (separated by space or comma): ")
            separators = [",", " "]
            for sep in separators:
                if sep in user_input:
                    user_input = user_input.replace(sep, " ")
            values = user_input.strip().split()

            for val in values:
                try:
                    num = int(val)
                    ll.add_to_end(num)
                except ValueError:
                    print(f"Invalid input '{val}'. Skipping.")

        elif choice == '2':
            ll.print_list()

        elif choice == '3':
            try:
                n = int(input("Enter the position to delete (1-based index): "))
                ll.delete_nth_node(n)
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        elif choice == '4':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")
