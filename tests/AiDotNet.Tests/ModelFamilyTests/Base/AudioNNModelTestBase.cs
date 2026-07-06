using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for audio neural network models (speech, TTS, generation, enhancement).
/// Inherits all NN invariant tests and adds audio-specific invariants:
/// finite spectral energy, silence handling, variable input lengths, and output validity.
/// </summary>
public abstract class AudioNNModelTestBase<T> : NeuralNetworkModelTestBase<T>
{
    /// <summary>
    /// Audio models normalize away input SCALE (loudness) — a stacked LayerNorm / instance-norm
    /// front end. Two CONSTANT (DC) inputs that differ only in amplitude (0.1 vs 0.9) therefore
    /// collapse to the same normalized representation, so the base DifferentInputs invariants see
    /// "identical output" even for a perfectly healthy model (a scale-only difference is not a
    /// meaningful different input for a scale-invariant model). Emit a value-SEEDED oscillating
    /// signal instead, so distinct <c>value</c>s differ in CONTENT (waveform), not just scale, and
    /// survive normalization — while <c>value == 0</c> stays true silence for the silence invariants
    /// (SilenceIn_NearSilenceOut / SilenceClassification_ShouldNotCrash). Mirrors the documented
    /// index-model / segmentation target overrides in sibling bases.
    /// </summary>
    protected override Tensor<T> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<T>(shape);
        if (value == 0.0) return tensor; // all-zero silence

        // A distinct angular frequency per value → distinct waveform direction (not a scalar
        // multiple), so scale-normalizing front ends can't wash the two inputs together.
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble(System.Math.Sin((i + 1) * (value + 0.5) * 2.0));
        return tensor;
    }

    // =====================================================
    // AUDIO INVARIANT: Finite Spectral Energy
    // Audio output must have finite L2 energy — exploding values
    // produce deafening noise or crash downstream processing.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task FiniteSpectralEnergy()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        double energy = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double o = ConvertToDouble(output[i]);
            energy += o * o;
        }

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

    [Fact(Timeout = 120000)]
    public async Task SilenceIn_NearSilenceOut()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var silence = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(silence);

        // Compute RMS of output
        double sumSq = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double o = ConvertToDouble(output[i]);
            sumSq += o * o;
        }
        double rms = Math.Sqrt(sumSq / Math.Max(1, output.Length));

        Assert.True(rms < 1.0,
            $"Silence input produced output with RMS = {rms:F4}. " +
            "Audio model should produce near-silence for zero input.");
    }

    // =====================================================
    // AUDIO INVARIANT: Different Input Lengths Should Not Crash
    // Audio models must handle varying input sizes gracefully.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentInputLengths_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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
            Assert.False(double.IsNaN(ConvertToDouble(output[i])),
                $"Output[{i}] is NaN for smaller input — model can't handle variable lengths.");
        }
    }

    // =====================================================
    // AUDIO INVARIANT: Output Length Should Be Positive
    // Audio output must contain at least one sample.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputLength_ShouldBePositive()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Audio model produced empty output.");
    }
}

/// <summary>Double-precision default for <see cref="AudioNNModelTestBase{T}"/>.</summary>
public abstract class AudioNNModelTestBase : AudioNNModelTestBase<double> { }
