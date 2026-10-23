"""Check the Kuhn notebook against direct play and exhaustive best responses."""
import contextlib
import importlib.util
import io
from itertools import permutations, product
import json
from pathlib import Path
import sys
import types
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'content/notebooks/learning_efg/cfr_template.ipynb'


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy is required for the CFR notebook')
class KuhnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = types.ModuleType('cfr_kuhn_notebook')
        sys.modules[cls.code.__name__] = cls.code
        cells = json.loads(NOTEBOOK.read_text())['cells']
        cls.run_source = ''.join(next(c for c in cells if c['id'] == 'kuhn-run')['source'])
        for cell in cells:
            if cell['id'] == 'kuhn-run':
                break
            if cell['cell_type'] == 'code':
                exec(''.join(cell['source']), cls.code.__dict__)
        cls.np = cls.code.np
        cls.cards = cls.code.cards
        cls.processes = [cls.code.p1, cls.code.p2]
        cls.payoff_matrix = cls.code.A
        cls.plans, cls.vectors = [], []
        for player, process in enumerate(cls.processes):
            plans, vectors = [], []
            for choices in product(range(2), repeat=len(process.decision_points)):
                local_strategies = {}
                plan = {}
                for j, choice in zip(process.decision_points, choices):
                    plan[j.metadata['card'], j.metadata['history']] = j.actions[choice]
                    local_strategies[j] = cls.np.eye(2)[choice]
                plans.append(plan)
                vectors.append(process.sequence_strategy(local_strategies))
            cls.plans.append(plans)
            cls.vectors.append(cls.np.array(vectors))

    def direct_payoff(self, cards, plans):
        """Play a hand using chip contributions, independently of the payoff matrix."""
        contributions = [1, 1]
        history = ()
        while True:
            player = len(history) % 2
            action = plans[player][cards[player], history]
            if action == 'fold':
                return -contributions[0] if player == 0 else contributions[1]
            if action in ('bet', 'call'):
                contributions[player] += 1
            history += (action,)
            if action == 'call' or history == ('check', 'check'):
                if self.cards.index(cards[0]) > self.cards.index(cards[1]):
                    return contributions[1]
                return -contributions[0]

    def test_information_sets_and_matrix_match_all_pure_profiles(self):
        self.assertEqual([len(p.decision_points) for p in self.processes], [6, 6])
        self.assertEqual(self.payoff_matrix.shape, (12, 12))
        for player, histories in enumerate([((), ('check', 'bet')), (('check',), ('bet',))]):
            nodes = self.processes[player].decision_points
            self.assertEqual({(j.metadata['card'], j.metadata['history']) for j in nodes},
                             set(product(self.cards, histories)))
        expected = self.np.zeros((64, 64))
        deals = list(permutations(self.cards, 2))
        for i, first in enumerate(self.plans[0]):
            for j, second in enumerate(self.plans[1]):
                expected[i, j] = sum(self.direct_payoff(cards, [first, second]) for cards in deals) / 6
        actual = self.vectors[0] @ self.payoff_matrix @ self.vectors[1].T
        self.np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_best_response_matches_exhaustive_search(self):
        rng = self.np.random.default_rng(7)
        for process, vectors in zip(self.processes, self.vectors):
            for _ in range(10):
                g = rng.normal(size=len(process.sequences))
                self.assertAlmostEqual(self.code.best_response_value(process, g), max(vectors @ g))

    def test_known_equilibrium_value_and_zero_exploitability(self):
        # An equilibrium with Player 1 betting J with probability 1/3 and K always.
        second_action_probabilities = [
            {(): [1/3, 0, 1], ('check', 'bet'): [0, 2/3, 1]},
            {('check',): [1/3, 0, 1], ('bet',): [0, 1/3, 1]},
        ]
        strategies = []
        for player, process in enumerate(self.processes):
            local_strategies = {}
            for j in process.decision_points:
                card = self.cards.index(j.metadata['card'])
                p = second_action_probabilities[player][j.metadata['history']][card]
                local_strategies[j] = self.np.array([1 - p, p])
            strategies.append(process.sequence_strategy(local_strategies))
        x, y = strategies
        A = self.payoff_matrix
        self.assertAlmostEqual(x @ A @ y, -1/18)
        self.assertAlmostEqual(self.code.best_response_value(self.processes[0], A @ y), -1/18)
        self.assertAlmostEqual(-self.code.best_response_value(self.processes[1], -A.T @ x), -1/18)

    def test_shared_tree_operations_preserve_counterfactual_updates(self):
        left, right = self.code.DecisionPoint(['a', 'b']), self.code.DecisionPoint(['a', 'b'])
        left.children = {'a': None, 'b': None}
        right.children = {'a': None, 'b': None}
        observation = self.code.ObservationPoint(['left', 'right'])
        observation.children = {'left': left, 'right': right}
        root = self.code.DecisionPoint(['enter', 'quit'])
        root.children = {'enter': observation, 'quit': None}
        process = self.code.TreeFormDecisionProcess(root)
        cfr = self.code.CFR(process)
        cfr.local_learners[root].regret_sum[:] = [1, 3]
        cfr.local_learners[left].regret_sum[:] = [3, 1]
        x = cfr.NextStrategy()
        self.np.testing.assert_allclose(x, [.25, .75, .1875, .0625, .125, .125])
        for j, strategy in cfr.local_strategies.items():
            self.np.testing.assert_allclose(process.local_strategy(x, j), strategy)
        g = self.np.array([1., 2., 4., -2., -1., 3.])
        values, utilities = process.evaluate(g, lambda j, u: cfr.local_strategies[j] @ u)
        self.assertAlmostEqual(values[root], 2.625)
        self.np.testing.assert_allclose(utilities[root], [4.5, 2])
        self.assertAlmostEqual(self.code.best_response_value(process, g), 8)
        cfr.ObserveUtility(g)
        self.np.testing.assert_allclose(cfr.local_learners[root].regret_sum, [2.875, 2.375])
        self.np.testing.assert_allclose(cfr.local_learners[left].regret_sum, [4.5, -3.5])
        self.np.testing.assert_allclose(cfr.local_learners[right].regret_sum, [-2, 2])

    def test_training_averages_valid_sequence_strategies_and_converges(self):
        self.code.iterations = 5_000
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.run_source, self.code.__dict__)
        x, y, history = self.code.x_average, self.code.y_average, self.code.history
        self.assertEqual(history[0, 0], 1)
        self.assertEqual(history[-1, 0], 5_000)
        self.assertTrue(self.np.isfinite(history).all())
        self.assertTrue((self.np.diff(history[:, 0]) > 0).all())
        for process, strategy in zip(self.processes, [x, y]):
            self.assertTrue((strategy >= 0).all())
            for j in process.decision_points:
                reach = 1 if j.parent is None else strategy[process.sequence_index[j.parent]]
                mass = sum(strategy[process.sequence_index[self.code.Sequence(j, a)]] for a in j.actions)
                self.assertAlmostEqual(mass, reach)
        self.assertTrue((history[:, 2] <= -1/18 + 1e-12).all())
        self.assertTrue((history[:, 3] >= -1/18 - 1e-12).all())
        self.assertTrue((history[:, 2] <= history[:, 1] + 1e-12).all())
        self.assertTrue((history[:, 1] <= history[:, 3] + 1e-12).all())
        self.assertLess(abs(history[-1, 1] + 1/18), 0.001)
        exploitability = (history[:, 3] - history[:, 2]) / 2
        self.assertLess(exploitability[-1], 0.005)
        self.assertLess(exploitability[-1], exploitability[0] / 50)
        self.np.testing.assert_allclose(history[-1, 1], x @ self.payoff_matrix @ y)


if __name__ == '__main__':
    unittest.main()
