import unittest
import networkx as nx

from utils.primitives import Crossword


class GraphPruneTest(unittest.TestCase):
    def test_empty_answers_returns_empty_neighbors(self):
        crossword = Crossword.__new__(Crossword)
        crossword.current_answers = {}
        crossword.G = nx.Graph()
        crossword.G.add_nodes_from(["1_a", "1_d"])
        crossword.clues = {"1_a": object(), "1_d": object()}

        next_search, solved = crossword._graph_prune()

        self.assertEqual(next_search, {"1_a", "1_d"})
        self.assertFalse(solved)


if __name__ == "__main__":
    unittest.main()
