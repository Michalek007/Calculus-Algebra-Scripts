

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.child = []
        self.parent = None
        self.parent_branch_value = None

    def add_child(self, obj, branch_value):
        self.child.append(obj)
        obj.parent = self
        obj.parent_branch_value = branch_value

    def display(self):
        print(self.data)
        if self.parent_branch_value:
            print(self.parent_branch_value)


class Tree:
    def __init__(self, main_node: TreeNode):
        self.main_node = main_node

    def display(self, main_node):
        main_node.display()
        for child in main_node.child:
            self.display(child)
