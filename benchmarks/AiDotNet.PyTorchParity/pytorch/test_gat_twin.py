"""The gat row's twin evaluates attention over the graph's edges; it must compute what the dense masked reference does.

Run with: python -m unittest discover -s benchmarks/AiDotNet.PyTorchParity/pytorch -p test_gat_twin.py
"""

import unittest

import torch

import benchmark


class GatTwinTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.nodes = 60
        self.mask = benchmark.cora_like_graph(self.nodes).double()
        self.edges = benchmark.cora_like_edges(self.nodes)

    def layers(self, concat, heads):
        dense = benchmark.DenseGATLayer(11, 5, heads, concat=concat, dropout=0.6).double().eval()
        edge = benchmark.EdgeGATLayer(11, 5, heads, concat=concat, dropout=0.6).double().eval()
        edge.load_state_dict(dense.state_dict())
        with torch.no_grad():
            dense.bias.normal_()
            edge.bias.copy_(dense.bias)
        return dense, edge

    def test_graph_has_the_rows_edges(self):
        rows, columns = self.edges
        self.assertEqual(rows.numel(), self.nodes * 5)  # self-loop plus four ring and chord neighbours per node
        self.assertEqual(int((self.mask == 0).sum()), rows.numel())
        self.assertTrue(torch.all(self.mask[rows, columns] == 0))

    def test_edge_layer_matches_dense_forward_and_gradients(self):
        for concat, heads in ((True, 8), (False, 1)):
            with self.subTest(concat=concat, heads=heads):
                dense, edge = self.layers(concat, heads)
                x = torch.randn(1, self.nodes, 11, dtype=torch.float64)
                expected = dense(x, self.mask)
                actual = edge(x, self.edges)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

                weights = torch.randn_like(expected)
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                for (name, a), b in zip(edge.named_parameters(), dense.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-12, msg=name)

    def test_cora_gat_keeps_the_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in benchmark.CoraGAT().parameters()), 92_373)


if __name__ == "__main__":
    unittest.main()
