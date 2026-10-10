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
    /// <summary>Creates the model under test.</summary>
    protected abstract INeuralNetworkModel<T> CreateTtsNetwork();

    /// <summary>
    /// Creates the model and, when its paper synthesizes in a given voice (AdaSpeech reads a speaker and a reference
    /// recording, Chen et al. 2021 §3), gives it one: speaker 0, language 0, a smooth reference mel spectrogram and a
    /// one-second reference waveform. Every
    /// inherited invariant then exercises the model's real inference path instead of a voice-less refusal.
    /// </summary>
    protected sealed override INeuralNetworkModel<T> CreateNetwork()
    {
        var network = CreateTtsNetwork();
        if (network is AiDotNet.TextToSpeech.TtsModelBase<T> tts
            && tts.SynthesisVoiceRequirement != AiDotNet.TextToSpeech.TtsSupervision.None)
        {
            int melChannels = tts.MelChannels;
            var reference = new Tensor<T>(new[] { 12, melChannels });
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            for (int f = 0; f < 12; f++)
                for (int c = 0; c < melChannels; c++)
                    reference[f, c] = ops.FromDouble(Math.Sin(0.37 * f + 0.21 * c));
            // A second of a two-tone waveform at the model's rate, for models whose speaker encoder reads audio (YourTTS).
            int samples = Math.Max(1, tts.SampleRate);
            var recording = new Tensor<T>(new[] { samples });
            for (int i = 0; i < samples; i++)
                recording[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180.0 * i / samples) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / samples));
            tts.Voice = new AiDotNet.TextToSpeech.TtsVoice<T>
            {
                SpeakerId = 0,
                Reference = reference,
                ReferenceAudio = recording,
                LanguageId = 0,
            };
        }
        return network;
    }

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
            tts.Train(SyntheticSupervision(input, target, tts));
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
            return ConvertToDouble(tts.EvaluateTrainingObjective(SyntheticSupervision(input, target, tts)));
        return base.MeasureLoss(network, input, output, target);
    }

    /// <inheritdoc />
    /// <remarks>A model measured on its typed objective (<see cref="MeasureLoss"/>) re-estimates its BatchNorm statistics on
    /// that objective's forward pass, over the same synthetic supervision.</remarks>
    protected override void RunRecalibrationPass(AiDotNet.NeuralNetworks.NeuralNetworkBase<T> network, Tensor<T> input, Tensor<T>? target)
    {
        if (target is not null && network is AiDotNet.TextToSpeech.TtsModelBase<T> tts
            && tts.TrainingSupervision != AiDotNet.TextToSpeech.TtsSupervision.None)
        {
            tts.EvaluateTrainingObjective(SyntheticSupervision(input, target, tts));
            return;
        }
        base.RunRecalibrationPass(network, input, target);
    }

    private static AiDotNet.TextToSpeech.TtsTrainingSample<T> SyntheticSupervision(Tensor<T> tokens, Tensor<T> target,
        AiDotNet.TextToSpeech.TtsModelBase<T>? network = null)
    {
        int frames = target.Rank >= 2 ? target.Shape[target.Rank - 2] : target.Length;
        int channels = target.Rank >= 2 ? target.Shape[target.Rank - 1] : 1;
        var mel = new Tensor<T>(new[] { frames, channels }, target.ToVector());
        int tokenCount = tokens.Length;

        var durations = new int[tokenCount];
        for (int i = 0; i < tokenCount; i++) durations[i] = frames / tokenCount + (i < frames % tokenCount ? 1 : 0);

        var pitch = new double[frames];
        var energy = new double[frames];
        // A log-magnitude linear spectrogram for models whose post-net predicts one (Tacotron): smooth in frequency,
        // following the target's frame-to-frame changes through its mean.
        AiDotNet.Tensors.LinearAlgebra.Tensor<T>? linear = null;
        if (network is not null && (network.TrainingSupervision & AiDotNet.TextToSpeech.TtsSupervision.Recording) != 0)
        {
            int bins = network.LinearSpectrogramBins;
            linear = new AiDotNet.Tensors.LinearAlgebra.Tensor<T>(new[] { frames, bins });
            var linOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            for (int f = 0; f < frames; f++)
            {
                double level = 0;
                for (int c = 0; c < channels; c++) level += linOps.ToDouble(mel[f, c]) / channels;
                for (int k = 0; k < bins; k++) linear[f, k] = linOps.FromDouble(level - 2.0 * k / bins);
            }
        }
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

        // Codec tokens for models that generate them (Pheme): one code per codebook per frame, a deterministic function of
        // the target so a memorization task has something consistent to learn.
        Tensor<T>? codecTokens = null;
        Tensor<T>? speakerRecording = null;
        Tensor<T>? promptCodecTokens = null;
        if (network is not null && (network.TrainingSupervision & AiDotNet.TextToSpeech.TtsSupervision.CodecTokens) != 0)
        {
            int codebooks = network.CodecTokenCodebooks, vocabulary = network.CodecTokenVocabulary;
            // A waveform target [samples] holds one codec frame per hop; a frame-shaped target one per row.
            int codecFrames = target.Rank >= 2 ? frames : Math.Max(1, target.Length / Math.Max(1, network.HopSize));
            codecTokens = new Tensor<T>(new[] { codecFrames, codebooks });
            for (int f = 0; f < codecFrames; f++)
            {
                double level = 0;
                for (int c = 0; c < channels; c++) level += Math.Abs(ops.ToDouble(mel[f, c]));
                for (int q = 0; q < codebooks; q++)
                    codecTokens[f, q] = ops.FromDouble(((int)(level * 997) + 31 * q + 7 * f) % vocabulary);
            }
        }
        if (network is not null && (network.TrainingSupervision & AiDotNet.TextToSpeech.TtsSupervision.PromptCodecTokens) != 0)
        {
            // Another "utterance of the same speaker": half as many frames, a different deterministic code pattern.
            int codebooks = network.CodecTokenCodebooks, vocabulary = network.CodecTokenVocabulary;
            // Sized from the codec tokens (a waveform target counts samples, not frames).
            int promptFrames = Math.Max(2, (codecTokens?.Shape[0] ?? frames) / 2);
            promptCodecTokens = new Tensor<T>(new[] { promptFrames, codebooks });
            for (int f = 0; f < promptFrames; f++)
                for (int q = 0; q < codebooks; q++)
                    promptCodecTokens[f, q] = ops.FromDouble((13 * f + 5 * q + 3) % vocabulary);
        }
        if (network is not null && (network.TrainingSupervision & AiDotNet.TextToSpeech.TtsSupervision.ReferenceRecording) != 0)
        {
            // The same one-second two-tone waveform CreateNetwork gives as the voice.
            int samples = Math.Max(1, network.SampleRate);
            speakerRecording = new Tensor<T>(new[] { samples });
            for (int i = 0; i < samples; i++)
                speakerRecording[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180.0 * i / samples) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / samples));
        }

        return new AiDotNet.TextToSpeech.TtsTrainingSample<T>
        {
            Tokens = tokens.Rank == 1 ? tokens : new Tensor<T>(new[] { tokenCount }, tokens.ToVector()),
            Mel = mel,
            Durations = durations,
            Pitch = pitch,
            Energy = energy,
            SpeakerId = 0,
            LanguageId = 0,
            LinearSpectrogram = linear,
            CodecTokens = codecTokens,
            SpeakerReference = speakerRecording,
            PromptCodecTokens = promptCodecTokens,
        };
    }

    // =====================================================
    // TTS INVARIANT: Different Text → Different Audio
    // Different text inputs must produce different mel/audio output.
    // A TTS model ignoring its text conditioning is broken.
    // =====================================================

    [Fact(Timeout = 120000)]
    public virtual async Task DifferentText_DifferentAudio()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        WarmUpForInputSensitivity(network);

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
    public virtual async Task Synthesize_DifferentText_DifferentOutput()
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
