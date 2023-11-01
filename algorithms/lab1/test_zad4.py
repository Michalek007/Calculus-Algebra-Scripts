import unittest

from algorithms.lab1.zad4 import TreeNode, Tree


class TestTreeNode(unittest.TestCase):

    def test_add_child(self):
        top_node = TreeNode(data=1)
        child_left = TreeNode(data=2)
        child_right = TreeNode(data=2)
        top_node.add_child(child_left, 2)
        top_node.add_child(child_right, 2)
        self.assertEqual(top_node.child, [child_left, child_right])

    def test_add_child_parent(self):
        top_node = TreeNode(1)
        child_left = TreeNode(2)
        top_node.add_child(child_left, 2)
        self.assertEqual(child_left.parent, top_node)


class TestTree(unittest.TestCase):

    def test_tree_to_string(self):
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

        self.assertEqual(str(tree), "Data->1\n\tBranch->2 Data->2\n\t\tBranch->2.3 Data->3\n\t\tBranch->2.4 Data->2.1\n\tBranch->2 Data->2\n")

    def test_traversal_method(self):
           assert True


if __name__ == '__main__':
    unittest.main()
