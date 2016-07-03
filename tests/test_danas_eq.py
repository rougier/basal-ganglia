import unittest
import numpy as np

import dotdot
from cdana import pydana
import cdana


class DanaEquivalence(unittest.TestCase):
    """Tests aimed at verifying that dana and cdana produce the exact same ouputs."""


    def test_connections_random(self):
        """Test all 1D connections"""

        for pycon, ccon, src_shape, tgt_shape, wgt_shape in [
            (pydana.OneToOne, cdana.OneToOne,  4,  4,  4),
            (pydana.OneToAll, cdana.OneToAll,  4,  4,  4),
            (pydana.AssToMot, cdana.AssToMot, 16,  4,  4),
            (pydana.AssToCog, cdana.AssToCog, 16,  4,  4),
            (pydana.MotToAss, cdana.MotToAss,  4, 16,  4),
            (pydana.CogToAss, cdana.CogToAss,  4, 16,  4),
            (pydana.AllToAll, cdana.AllToAll,  4,  4, 16)]:

            np.random.seed(0)

            for _ in range(1000):
                source  = np.random.rand(src_shape)
                target  = np.random.rand(tgt_shape)
                weights = np.random.rand(wgt_shape)
                gain    = np.random.random()

                source_copy, weights_copy = np.copy(source), np.copy(weights)

                py = pycon(source, np.copy(target), weights, gain)
                c  =  ccon(source, np.copy(target), weights, gain)


                for _ in range(10):
                    py.propagate()
                    c.propagate()
                    self.assertTrue(np.all(source == source_copy))
                    self.assertTrue(np.all(weights == weights_copy))
                    self.assertTrue(np.allclose(c.target, py.target, rtol=1e-05, atol=1e-08))


    @classmethod
    def random_inputs(cls, group):
        group['U']    = np.random.random(group['U'].shape)
        group['Iext'] = np.random.random(group['Iext'].shape)
        group['Isyn'] = np.random.random(group['Isyn'].shape)

    def test_group_random(self):
        total, error = 0, 0

        N, M = 1000, 10
        dt = np.random.uniform(low=-0.1, high=0.1)

        for seed in range(N):
            cg  = cdana.Group(4)
            pyg = pydana.Group(4)

            Us, deltas = [[], []], [[], []]

            for j, g in enumerate([cg, pyg]):
                Us.append([])
                deltas.append([])
                np.random.seed(seed)
                self.random_inputs(g)

                for i in range(M):
                    g.evaluate(dt)
                    deltas[j].append(g.delta)
                    Us[j].append(g['U'])

            for k in range(M):
                total += 1
                if deltas[0][k] != deltas[1][k]:
                    error += 1
                #self.assertEqual(deltas[0][k], deltas[1][k])
                self.assertTrue(np.all(Us[0][k] == Us[1][k]))


if __name__ == '__main__':
    unittest.main()
