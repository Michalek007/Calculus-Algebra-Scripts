

class TreeNode:
    """ Implements tree node.
        Attributes:
            data: self value
            child: list of child nodes (TreeNode objects)
            parent: parent node (TreeNode object)
            parent_branch_value: value for parent branch
    """
    def __init__(self, data):
        self.data = data
        self.child = []
        self.parent = None
        self.parent_branch_value = None

    def add_child(self, obj, branch_value):
        """ Appends child node and sets branch value.
            obj: TreeNode object
            branch_value -> value for parent branch
        """
        self.child.append(obj)
        obj.parent = self
        obj.parent_branch_value = branch_value

    def __str__(self):
        self_str = f'Branch->{self.parent_branch_value} ' if self.parent_branch_value is not None else ''
        return self_str + f'Data->{self.data}'


class Tree:
    """ Implements tree structure.
        Attributes:
            main_node: top node of the tree
    """
    def __init__(self, main_node: TreeNode):
        self.main_node = main_node

    @staticmethod
    def tab_generator(count: int):
        tab_str = ''
        for _ in range(count):
            tab_str += '\t'
        return tab_str

    @property
    def min_value(self):
        return min(self.get_all_values())

    def get_all_values(self, main_node=None, values=None):
        if values is None:
            values = []
        if main_node is None:
            main_node = self.main_node
        values.append(main_node.data)
        for child in main_node.child:
            self.get_all_values(child, values)
        return values

    def display(self, main_node=None, iteration: int = 0):
        """ Displays all nodes using recursive algorithm pre-order. """
        if main_node is None:
            main_node = self.main_node
        print(self.tab_generator(iteration) + str(main_node))
        for child in main_node.child:
            self.display(child, iteration + 1)

    def to_string(self, main_node, iteration: int = 0):
        main_str = self.tab_generator(iteration) + str(main_node) + '\n'
        for child in main_node.child:
            main_str += self.to_string(child, iteration + 1)
        return main_str

    def __str__(self):
        return self.to_string(self.main_node)


if __name__ == '__main__':
    top_node = TreeNode(data=1)
    child_left = TreeNode(data=2)
    child_right = TreeNode(data=2)
    top_node.add_child(child_left, 2)
    top_node.add_child(child_right, 2)

    child_left_left = TreeNode(data=3)
    child_left_right = TreeNode(data=2.1)

    child_left.add_child(child_left_left, 2.3)
    child_left.add_child(child_left_right, 2.4)

    tree = Tree(main_node=top_node)

    print(tree)

    print(tree.min_value)
    tree.min_value = 10
    # tree.display()
