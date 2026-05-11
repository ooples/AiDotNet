using System;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class QuantumNeuralNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new QuantumNeuralNetwork<double>();

    // Paper-faithful constant input for quantum-state-encoded networks.
    //
    // <para>
    // QuantumLayer.Forward (the QNN's first trainable layer) L2-normalizes
    // its input to unit length per the Born-rule convention for state
    // amplitudes (|ψ|² is a probability, so ‖ψ‖₂ = 1). That makes the
    // network deliberately SCALE-invariant: a uniformly-constant tensor at
    // any scalar value normalizes to the same uniform unit vector, so
    // CreateConstantTensor(_, 0.1) and CreateConstantTensor(_, 0.9) feed
    // the QuantumLayer IDENTICAL quantum states. The base class's
    // DifferentInputs_ShouldProduceDifferentOutputs / *_AfterTraining
    // invariants — which compare outputs of these two constants —
    // therefore false-fail on a correctly-implemented quantum model.
    // </para>
    // <para>
    // The base CreateConstantTensor's own XML-doc explicitly documents
    // this override pattern ("Virtual so paper-faithful index-based models
    // can translate constant scalars..."). Apply the same idea here: shape
    // the "constant" value with a position-dependent modulation so two
    // different scalars produce two different DIRECTIONS in state space.
    // The modulation amplitude is small (±10 %) so the test still
    // exercises the "is the network input-sensitive" question rather than
    // turning into an unrelated stress test.
    // </para>
    protected override Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<double>(shape);
        int len = tensor.Length;
        for (int i = 0; i < len; i++)
        {
            // CRITICAL: the modulation amplitude must scale with `value` (not
            // multiplicatively wrap it) — multiplicative wrapping produces the
            // SAME direction for two different `value` inputs, defeating the
            // whole point of distinguishing them under L2 normalization. By
            // adding `value` as an additive offset to a position-dependent
            // sinusoid, the relative shape of the tensor (and therefore its
            // post-normalization direction) varies with `value`: at value=0.1
            // the sinusoid term dominates and the direction tracks Sin(...);
            // at value=0.9 the offset dominates and the direction is closer
            // to uniform with a smaller sinusoidal ripple.
            tensor[i] = value + 0.5 * Math.Sin(i * Math.PI / Math.Max(1, len - 1));
        }
        return tensor;
    }

    // Override of ScaledInput_ShouldChangeOutput for quantum-state-encoded
    // models. The base test scales the input by 10× and expects a
    // different output — a magnitude-preserving network always passes this,
    // but a unit-norm-encoded quantum network is invariant to scalar
    // scaling by design (the normalized state ψ/‖ψ‖ is unchanged). Replace
    // "scale by 10×" with "add a position-dependent perturbation"
    // (which DOES change the unit direction); the underlying invariant the
    // base test cares about — "Forward pass actually consumes input
    // values, isn't a constant function" — still holds, just via a quantum-
    // appropriate probe.
    public override async System.Threading.Tasks.Task ScaledInput_ShouldChangeOutput()
    {
        await System.Threading.Tasks.Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();

        var input = CreateRandomTensor(InputShape, rng);
        var perturbedInput = new Tensor<double>(InputShape);
        int len = input.Length;
        for (int i = 0; i < len; i++)
        {
            // Position-dependent additive perturbation: changes the DIRECTION
            // of the input vector (which a unit-norm quantum encoding is
            // sensitive to), not just its magnitude (which it isn't).
            double delta = 0.5 * Math.Sin(i * Math.PI / Math.Max(1, len - 1));
            perturbedInput[i] = input[i] + delta;
        }

        var output1 = network.Predict(input);
        var output2 = network.Predict(perturbedInput);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Quantum network output didn't change when input direction was perturbed. "
            + "Forward pass may ignore input values (note: scalar scaling is a no-op for "
            + "unit-norm-encoded quantum networks; this test perturbs the input DIRECTION instead).");
    }
}
