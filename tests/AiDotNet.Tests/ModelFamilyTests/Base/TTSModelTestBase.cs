using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for text-to-speech models (acoustic models, vocoders, end-to-end TTS).
/// Inherits all NN invariant tests and adds TTS-specific invariants: different input produces
/// different output, output is non-empty and bounded, and repeated synthesis is identical.
/// </summary>
/// <remarks>
/// The inherited invariants drive <c>Predict</c>, which runs the layer stack front to back. A text-to-speech
/// model's public entry point is <see cref="ITtsModel{T}.Synthesize"/>, which splits that stack into an
/// encoder and a decoder around the model's own variance or alignment logic. The two paths share no
/// code past the layers, so the <c>Synthesize</c> invariants below are what catch a broken split: all
/// thirteen classic acoustic models threw on every call to it (#2091) while every <c>Predict</c>
/// invariant passed.
/// </remarks>
public abstract class TTSModelTestBase<T> : NeuralNetworkModelTestBase<T>
{
    /// <summary>
    /// Trains a model whose paper needs supervision a token/mel pair does not carry (FastSpeech 2's forced-alignment
    /// durations) through its typed entry point, with synthetic supervision consistent with the fixture's target:
    /// durations spread evenly so they sum to the target's frame count, a smooth pitch contour in the speaking range,
    /// and the frame energy of the target spectrogram.
    /// </summary>
    protected override void TrainOn(INeuralNetworkModel<T> network, Tensor<T> input, Tensor<T> target)
    {
        if (network is AiDotNet.TextToSpeech.TtsModelBase<T> tts
            && tts.TrainingSupervision != AiDotNet.TextToSpeech.TtsSupervision.None)
        {
            tts.Train(SyntheticSupervision(input, target));
            return;
        }
        base.TrainOn(network, input, target);
    }

    /// <summary>
    /// Measures a model trained through its typed entry point on the objective it was trained on, with the same
    /// synthetic supervision <see cref="TrainOn"/> gives it. Its prediction length follows its own predicted durations,
    /// so comparing a prediction with a fixed-length target would measure nothing.
    /// </summary>
    protected override double MeasureLoss(INeuralNetworkModel<T> network, Tensor<T> input, Tensor<T> output, Tensor<T> target)
    {
        if (network is AiDotNet.TextToSpeech.TtsModelBase<T> tts
            && tts.TrainingSupervision != AiDotNet.TextToSpeech.TtsSupervision.None)
            return ConvertToDouble(tts.EvaluateTrainingObjective(SyntheticSupervision(input, target)));
        return base.MeasureLoss(network, input, output, target);
    }

    private static AiDotNet.TextToSpeech.TtsTrainingSample<T> SyntheticSupervision(Tensor<T> tokens, Tensor<T> target)
    {
        int frames = target.Rank >= 2 ? target.Shape[target.Rank - 2] : target.Length;
        int channels = target.Rank >= 2 ? target.Shape[target.Rank - 1] : 1;
        var mel = new Tensor<T>(new[] { frames, channels }, target.ToVector());
        int tokenCount = tokens.Length;

        var durations = new int[tokenCount];
        for (int i = 0; i < tokenCount; i++) durations[i] = frames / tokenCount + (i < frames % tokenCount ? 1 : 0);

        var pitch = new double[frames];
        var energy = new double[frames];
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int f = 0; f < frames; f++)
        {
            pitch[f] = 150.0 + 50.0 * Math.Sin(2 * Math.PI * f / Math.Max(1, frames));
            double sum = 0;
            for (int c = 0; c < channels; c++)
            {
                double magnitude = Math.Exp(Math.Min(5.0, ops.ToDouble(mel[f, c])));
                sum += magnitude * magnitude;
            }
            energy[f] = Math.Min(Math.Sqrt(sum), 600.0);
        }

        return new AiDotNet.TextToSpeech.TtsTrainingSample<T>
        {
            Tokens = tokens.Rank == 1 ? tokens : new Tensor<T>(new[] { tokenCount }, tokens.ToVector()),
            Mel = mel,
            Durations = durations,
            Pitch = pitch,
            Energy = energy,
        };
    }

    // =====================================================
    // TTS INVARIANT: Different Text → Different Audio
    // Different text inputs must produce different mel/audio output.
    // A TTS model ignoring its text conditioning is broken.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentText_DifferentAudio()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        var text1 = CreateConstantTensor(EffectiveInputShape, 0.2);
        var text2 = CreateConstantTensor(EffectiveInputShape, 0.8);

        var audio1 = network.Predict(text1);
        var audio2 = network.Predict(text2);

        bool anyDifferent = false;
        int minLen = Math.Min(audio1.Length, audio2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ConvertToDouble(audio1[i]) - ConvertToDouble(audio2[i])) > 1e-10)
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

    [Fact(Timeout = 120000)]
    public async Task Output_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(EffectiveInputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0,
            "TTS model produced empty audio output.");
    }

    // =====================================================
    // TTS INVARIANT: Output Values Should Be Bounded
    // Audio/mel values should be in a reasonable range.
    // Extreme values produce clipping or distortion.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputValues_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(EffectiveInputShape, rng);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(ConvertToDouble(output[i])),
                $"TTS output[{i}] is NaN — numerical instability in synthesis.");
            Assert.False(double.IsInfinity(ConvertToDouble(output[i])),
                $"TTS output[{i}] is Infinity — overflow in synthesis.");
            Assert.True(Math.Abs(ConvertToDouble(output[i])) < 1e6,
                $"TTS output[{i}] = {ConvertToDouble(output[i]):E4} is out of reasonable range.");
        }
    }

    // =====================================================
    // TTS INVARIANT: Speaker Consistency
    // Same text input twice should produce similar spectral output.
    // A TTS model with high variance is unstable.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task SpeakerConsistency()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(EffectiveInputShape, rng);

        var out1 = network.Predict(input);
        var out2 = network.Predict(input);

        // Should be deterministic (identical)
        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }

    // =====================================================
    // TTS ENTRY PATH: Synthesize (#2091)
    // The text-in, audio-out call a user makes. Exercised separately because it does not go
    // through Predict.
    // =====================================================

    private const string SynthesisText = "the quick brown fox";

    private ITtsModel<T> RequireTtsModel(object network)
    {
        Skip.IfNot(network is ITtsModel<T>, "This fixture's model does not implement ITtsModel, so it has no Synthesize entry point.");
        return (ITtsModel<T>)network;
    }

    [SkippableFact(Timeout = 120000)]
    public async Task Synthesize_ShouldProduceFiniteNonEmptyOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var tts = RequireTtsModel(CreateNetwork());

        var output = tts.Synthesize(SynthesisText);

        Assert.True(output.Length > 0, "Synthesize produced empty output.");
        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(v) || double.IsInfinity(v), $"Synthesize output[{i}] is {v}.");
        }
    }

    [SkippableFact(Timeout = 120000)]
    public async Task Synthesize_IsDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var tts = RequireTtsModel(CreateNetwork());

        var first = tts.Synthesize(SynthesisText);
        var second = tts.Synthesize(SynthesisText);

        Assert.Equal(first.Shape.ToArray(), second.Shape.ToArray());
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i]);
    }

    [SkippableFact(Timeout = 120000)]
    public async Task Synthesize_DifferentText_DifferentOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var tts = RequireTtsModel(CreateNetwork());

        var a = tts.Synthesize(SynthesisText);
        var b = tts.Synthesize("a completely different sentence");

        bool differs = a.Length != b.Length;
        for (int i = 0; !differs && i < a.Length; i++)
            differs = Math.Abs(ConvertToDouble(a[i]) - ConvertToDouble(b[i])) > 1e-10;
        Assert.True(differs, "Synthesize returned identical output for different text: the text never reaches the output.");
    }
}

/// <summary>Double-precision default for <see cref="TTSModelTestBase{T}"/>.</summary>
public abstract class TTSModelTestBase : TTSModelTestBase<double> { }
