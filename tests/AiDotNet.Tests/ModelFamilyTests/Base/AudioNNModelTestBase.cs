using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
        // Some audio-family models accept codec/token IDs rather than waveform samples. Their
        // generated model contract is authoritative: an audio-shaped sine wave would be an illegal
        // fractional token tensor. Let the shared domain-aware fixture synthesize distinct legal
        // constants for every non-continuous domain, and reserve the waveform probe for true audio.
        if (InputDomainFor(shape).Kind != LayerInputDomainKind.Continuous)
            return base.CreateConstantTensor(shape, value);

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
        var input = CreateRandomTensor(EffectiveInputShape, rng);

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
    // AUDIO INVARIANT: Silence Handling Matches the Output Domain
    // Waveform-producing models should emit near-silence. Speech recognizers emit
    // categorical scores, where zero RMS is neither meaningful nor attainable for
    // normalized log probabilities; those outputs must instead be finite and normalized.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task SilenceIn_NearSilenceOut()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var silence = CreateConstantTensor(EffectiveInputShape, 0.0);

        var output = network.Predict(silence);

        if (network is ISpeechRecognizer<T>)
        {
            Assert.True(output.Length > 0, "Speech recognizer output should not be empty for silence.");
            for (int i = 0; i < output.Length; i++)
            {
                double value = ConvertToDouble(output[i]);
                Assert.True(double.IsFinite(value),
                    $"Speech recognizer produced non-finite silence output {value} at index {i}.");
            }

            if (network is NeuralNetworkBase<T> neuralNetwork
                && neuralNetwork.DefaultLossFunction is CTCLoss<T> ctc
                && ctc.InputsAreLogProbabilities)
            {
                Assert.True(output.Rank >= 2,
                    $"CTC log probabilities must include time and class axes; got rank {output.Rank}.");
                Assert.Equal(ctc.NumClasses, output.Shape[^1]);

                int classCount = output.Shape[^1];
                int frameCount = output.Length / classCount;
                double tolerance = typeof(T) == typeof(float) ? 1e-5 : 1e-10;
                for (int frame = 0; frame < frameCount; frame++)
                {
                    double probabilityMass = 0.0;
                    int offset = frame * classCount;
                    for (int @class = 0; @class < classCount; @class++)
                        probabilityMass += Math.Exp(ConvertToDouble(output[offset + @class]));

                    Assert.InRange(probabilityMass, 1.0 - tolerance, 1.0 + tolerance);
                }
            }

            return;
        }

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

    /// <summary>
    /// Index of the <see cref="EffectiveInputShape"/> axis that represents the model's
    /// variable-length (time / audio-token) dimension — the axis
    /// <see cref="DifferentInputLengths_ShouldNotCrash"/> varies to simulate a
    /// different clip. Defaults to the last axis, which is correct for the raw-audio
    /// and time-major frontends that make up most of the audio family. Transformer
    /// audio-language models whose input is <c>[batch, tokens, embedDim]</c> (e.g.
    /// Pengi) must override this to the token axis: their final embedding dimension is
    /// fixed by the attention projection weights and cannot vary, so halving the last
    /// axis would feed an invalid embedding width rather than a shorter sequence.
    /// </summary>
    protected virtual int VariableLengthAxis => EffectiveInputShape.Length - 1;

    [Fact(Timeout = 120000)]
    public async Task DifferentInputLengths_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        // Grow the variable-length axis. Shrinking can violate a generated minimum-length
        // contract (for example a separator whose convolution stack needs a full receptive
        // field), while growing still proves the model does not hard-code the fixture length.
        var variedShape = (int[])EffectiveInputShape.Clone();
        int lenAxis = VariableLengthAxis;
        variedShape[lenAxis] = checked(variedShape[lenAxis] * 2);
        var variedInput = CreateRandomInputTensor(variedShape, rng);

        var output = network.Predict(variedInput);
        Assert.True(output.Length > 0, "Output should not be empty for a different input length.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(ConvertToDouble(output[i])),
                $"Output[{i}] is NaN for a different input length — model can't handle variable lengths.");
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
        var input = CreateRandomTensor(EffectiveInputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Audio model produced empty output.");
    }
}

/// <summary>Double-precision default for <see cref="AudioNNModelTestBase{T}"/>.</summary>
public abstract class AudioNNModelTestBase : AudioNNModelTestBase<double> { }
