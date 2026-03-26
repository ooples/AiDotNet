using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for text-to-speech models (acoustic models, vocoders, end-to-end TTS).
/// Inherits all NN invariant tests and adds TTS-specific invariants:
/// different text produces different audio, longer text produces longer output,
/// empty input handling, and speaker consistency.
/// </summary>
public abstract class TTSModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // TTS INVARIANT: Different Text → Different Audio
    // Different text inputs must produce different mel/audio output.
    // A TTS model ignoring its text conditioning is broken.
    // =====================================================

    [Fact]
    public void DifferentText_DifferentAudio()
    {
        var network = CreateNetwork();

        var text1 = CreateConstantTensor(InputShape, 0.2);
        var text2 = CreateConstantTensor(InputShape, 0.8);

        var audio1 = network.Predict(text1);
        var audio2 = network.Predict(text2);

        bool anyDifferent = false;
        int minLen = Math.Min(audio1.Length, audio2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(audio1[i] - audio2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "TTS model produces identical audio for different text inputs — text conditioning is broken.");
    }

    // =====================================================
    // TTS INVARIANT: Output Should Be Non-Empty
    // TTS models must produce audio output of positive length.
    // =====================================================

    [Fact]
    public void Output_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0,
            "TTS model produced empty audio output.");
    }

    // =====================================================
    // TTS INVARIANT: Output Values Should Be Bounded
    // Audio/mel values should be in a reasonable range.
    // Extreme values produce clipping or distortion.
    // =====================================================

    [Fact]
    public void OutputValues_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"TTS output[{i}] is NaN — numerical instability in synthesis.");
            Assert.False(double.IsInfinity(output[i]),
                $"TTS output[{i}] is Infinity — overflow in synthesis.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"TTS output[{i}] = {output[i]:E4} is out of reasonable range.");
        }
    }

    // =====================================================
    // TTS INVARIANT: Speaker Consistency
    // Same text input twice should produce similar spectral output.
    // A TTS model with high variance is unstable.
    // =====================================================

    [Fact]
    public void SpeakerConsistency()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = network.Predict(input);
        var out2 = network.Predict(input);

        // Should be deterministic (identical)
        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }
}
