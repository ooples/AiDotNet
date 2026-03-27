using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for audio neural network models (speech, TTS, generation, enhancement).
/// Inherits all NN invariant tests and adds audio-specific invariants:
/// finite spectral energy, silence handling, variable input lengths, and output validity.
/// </summary>
public abstract class AudioNNModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // AUDIO INVARIANT: Finite Spectral Energy
    // Audio output must have finite L2 energy — exploding values
    // produce deafening noise or crash downstream processing.
    // =====================================================

    [Fact]
    public void FiniteSpectralEnergy()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        double energy = 0;
        for (int i = 0; i < output.Length; i++)
            energy += output[i] * output[i];

        Assert.True(!double.IsNaN(energy) && !double.IsInfinity(energy),
            "Audio output has infinite energy — values are exploding.");
        Assert.True(energy < 1e12,
            $"Audio output energy = {energy:E4} is unreasonably large. Possible numerical instability.");
    }

    // =====================================================
    // AUDIO INVARIANT: Silence In → Near-Silence Out
    // Zero input (silence) should produce near-zero output.
    // A model that produces loud output from silence is broken.
    // =====================================================

    [Fact]
    public void SilenceIn_NearSilenceOut()
    {
        var network = CreateNetwork();
        var silence = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(silence);

        // Compute RMS of output
        double sumSq = 0;
        for (int i = 0; i < output.Length; i++)
            sumSq += output[i] * output[i];
        double rms = Math.Sqrt(sumSq / Math.Max(1, output.Length));

        Assert.True(rms < 1.0,
            $"Silence input produced output with RMS = {rms:F4}. " +
            "Audio model should produce near-silence for zero input.");
    }

    // =====================================================
    // AUDIO INVARIANT: Different Input Lengths Should Not Crash
    // Audio models must handle varying input sizes gracefully.
    // =====================================================

    [Fact]
    public void DifferentInputLengths_ShouldNotCrash()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        // Try input at half the default size
        var halfShape = (int[])InputShape.Clone();
        halfShape[halfShape.Length - 1] = Math.Max(1, halfShape[halfShape.Length - 1] / 2);
        var smallInput = CreateRandomTensor(halfShape, rng);

        var output = network.Predict(smallInput);
        Assert.True(output.Length > 0, "Output should not be empty for smaller input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN for smaller input — model can't handle variable lengths.");
        }
    }

    // =====================================================
    // AUDIO INVARIANT: Output Length Should Be Positive
    // Audio output must contain at least one sample.
    // =====================================================

    [Fact]
    public void OutputLength_ShouldBePositive()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Audio model produced empty output.");
    }
}
