

class Node:
    
    """
    An object to store a single node of a linked list.
    Models two attributes - data and the link to the next node in the list
    
    Code: 
    python -i data_structures/linked_list.py
    >>> n1 = Node(10)
    >>> n1
    <Node data : 10>
    >>> n2 = Node(20)
    >>> n1.next_node = n2
    >>> n1.next_node
    <Node data : 20>
    
    """
    data = None
    next_node = None
    def __init__(self,data):
        self.data = data

    def __repr__(self):
        return "<Node data : %s>"  % self.data # %s is to say where to substitute the self.data value inside the string
    
class LinkedList:
    """
    Singly linked list
    
    """
    def __init__(self):
        self.head = None
        
    
    def is_empty(self):
        return self.head == None
    
    def size (self):
        """
        Returns the number of nodes in the list
        Takes O(n) time
        Code: 
        
        python -i data_structures/linked_list.py
        >>> l = LinkedList()
        >>> l.add(1)
        >>> l.size()
        1
        >>> l.add(2)
        >>> l.add(2)
        >>> l.size()
        3
        """
        current = self.head
        count = 0
        
        while current != None:
            count += 1
            current = current.next_node
        return count
    
    def add(self,data):
        """
        Adds new node containing data at the head of the list
        Takes O(n) time
        
        Code: 
        python -i data_structures/linked_list.py
        >>> l = LinkedList()
        >>> l.add(90)
        >>> l.add(34)
        >>> l.add(88)
        >>> l
        [Head: 88]-> [34]-> [Tail: 90]
        """
        new_node = Node(data)
        new_node.next_node = self.head
        self.head = new_node
        
    def search (self,key):
        """
        Search for the first node containing data that matches the key
        Returns the node or None if not found
        
        Takes O(n) time
        
        Code:
        python -i data_structures/linked_list.py
        >>> l = LinkedList()
        >>> l.add(10)
        >>> l.add(89)
        >>> l.add(81)
        >>> l.add(15)
        >>> n = l.search(89)
        >>> n
        <Node data : 89>
        """
        current = self.head
        while current:
            if current.data == key:
                return current
            else:
                current = current.next_node
        return None
    
    def insert(self,data,index):
        """
        Inserts a new Node containing data at index position
        
        Insertion takes O(1) time 
        but finding the node at the insertion point takes O(n) time
        
        Therefore overall O(n) time
        
        
        """
        if index == 0:
            self.add(data)
            
        if index > 0 :
            new = Node(data)
            position = index
            current = self.head
            
        while position > 1:
            current = node.next_node
            position -= 1
        
        prev_node = current
        next_node = current.next_node
        
        prev_node.next_node = new
        new.next_node = next_node
        
    def remove(self, key):
        """
        Removes node containing data tht matches the key
        
        Returns the node or None if key doesn't exist
        
        Takes O(n) time
        """
        current = self.head
        previous = None
        found = False
        
        while current and not found:
            if current.data == key and current is self.head:
                found = True
                self.head = current.next_node
            elif current.data == key:
                found = True
                previous.next_node = current.next_node
            else:
                previous = current
                current = current.next_node
                
        return current
       
    def node_at_index(self,index):
        if index == 0:
            return self.head
        else:
            current = self.head
            position = 0
            
            while position < index:
                current = current.next_node
                position += 1
            
            return current
        
            
    def __repr__(self):
        """
        Returns a string representation of the list
        Takes O(n) time
        """
        nodes = []
        current = self.head
        
        
        while current:
            if current is self.head:
                nodes.append("[Head: %s]" % current.data)
            elif current.next_node is None:
                nodes.append("[Tail: %s]" % current.data)
            else:
                nodes.append("[%s]" % current.data)

            current = current.next_node
        return "-> ".join(nodes)