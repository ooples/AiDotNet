using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HopfieldNetworkTests : AssociativeMemoryTestBase
{
    // Hopfield: associative memory, input=output=networkSize=128
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [128];

    // Hopfield is autoassociative: Train stores input patterns, ignores target
    protected override bool IsAutoAssociative => true;

    // Theoretical capacity is N/(4*ln(N)) ≈ 128/(4*4.85) ≈ 6.6 patterns
    // Use 3 patterns to stay well within capacity
    protected override int MultiPatternCount => 3;

    // Hopfield uses sign activation producing binary [-1,+1] output
    // which maps to {0, 1} after denormalization — coarse output
    protected override double RecallTolerance => 0.35;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HopfieldNetwork<double>();

    /// <summary>
    /// Computes the Hopfield energy for a given state via CalculateEnergy.
    /// Energy should be lower for stored patterns (attractors) than random states.
    /// </summary>
    protected override double? ComputeEnergy(INeuralNetworkModel<double> network, Tensor<double> state)
    {
        if (network is HopfieldNetwork<double> hopfield)
        {
            // Convert tensor to bipolar vector [-1, +1] matching Hopfield internals
            var vec = new Vector<double>(state.Length);
            for (int i = 0; i < state.Length; i++)
                vec[i] = 2.0 * state[i] - 1.0; // Map [0,1] → [-1,+1]

            return hopfield.CalculateEnergy(vec);
        }
        return null;
    }
}
