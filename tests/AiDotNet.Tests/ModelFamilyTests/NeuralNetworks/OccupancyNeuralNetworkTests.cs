using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class OccupancyNeuralNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [3];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new OccupancyNeuralNetwork<double>();

    /// <summary>
    /// Occupancy detection is a binary task (a point is occupied or it is not),
    /// so the target is a binary {0, 1} label — not the continuous [0, 1) value
    /// the regression-oriented base default emits. With a binary target the
    /// binary-cross-entropy minimum is 0 (vs. ~0.69 for the symmetric output and
    /// only ~1.4% below baseline for a fractional target), so the
    /// memorization-loss invariant has real headroom instead of sitting on a
    /// razor-thin margin that parallel-reduction noise can flip. This is the
    /// semantically correct target type for a binary classifier, supplied via
    /// the base's documented CreateRandomTargetTensor extension point.
    /// </summary>
    protected override Tensor<double> CreateRandomTargetTensor(int[] shape, System.Random rng)
    {
        var target = new Tensor<double>(shape);
        for (int i = 0; i < target.Length; i++)
            target[i] = rng.NextDouble() < 0.5 ? 0.0 : 1.0;
        return target;
    }
}
