using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// Issue #2313: the surviving CLAP network diverged from Wu et al. (ICASSP 2023) and from the
/// LAION-AI/CLAP reference in ways that are each separately observable.
/// </summary>
public sealed class ClapPaperFidelityTests
{
    private static CLAPModelOptions SmallOptions() => new()
    {
        SampleRate = 48_000,
        NumMelBands = 16,
        StftWindowSize = 256,
        HopLength = 128,
        AudioHiddenDim = 16,
        AudioEncoderLayers = 2,
        AudioEncoderHeads = 2,
        SwinWindowSize = 4,
        AudioPatchSize = 4,
        TextHiddenDim = 16,
        TextEncoderLayers = 2,
        TextEncoderHeads = 2,
        MaxTextLength = 16,
        VocabSize = 64,
        ProjectionDim = 8,
        Seed = 20261012,
    };

    /// <summary>
    /// The reference filterbank spans 50 Hz to 14 kHz (<c>model_configs/HTSAT-tiny.json</c>), not
    /// 0 Hz to Nyquist. A 20 kHz tone is therefore outside every mel band, so adding one to a
    /// signal must not move the embedding — which it does when fmax is the 24 kHz Nyquist rate.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ContentAboveTheReferenceFmaxDoesNotReachTheEmbedding()
    {
        await Task.Yield();

        var model = new CLAPModel<double>(SmallOptions());
        model.SetTrainingMode(false);

        const int samples = 2048;
        var plain = Tone(samples, 440.0, 48_000);
        var withUltrasound = Tone(samples, 440.0, 48_000);
        var ultrasound = Tone(samples, 20_000.0, 48_000);
        for (int i = 0; i < samples; i++)
            withUltrasound[i] += 0.5 * ultrasound[i];

        var baseline = model.EncodeAudio(AsTensor(plain));
        var perturbed = model.EncodeAudio(AsTensor(withUltrasound));

        double worst = 0.0;
        for (int i = 0; i < baseline.Length; i++)
            worst = Math.Max(worst, Math.Abs(baseline[i] - perturbed[i]));

        Assert.True(worst <= 1e-9,
            $"A 20 kHz tone moved the CLAP embedding by {worst:R}, so the mel filterbank still "
                + "reaches above the reference's 14 kHz fmax.");
    }

    /// <summary>
    /// A 1 kHz tone IS inside the filterbank, so the test above cannot be passing because the
    /// encoder ignores its input.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ContentInsideTheFilterbankDoesReachTheEmbedding()
    {
        await Task.Yield();

        var model = new CLAPModel<double>(SmallOptions());
        model.SetTrainingMode(false);

        const int samples = 2048;
        var plain = Tone(samples, 440.0, 48_000);
        var withTone = Tone(samples, 440.0, 48_000);
        var audible = Tone(samples, 1_000.0, 48_000);
        for (int i = 0; i < samples; i++)
            withTone[i] += 0.5 * audible[i];

        var baseline = model.EncodeAudio(AsTensor(plain));
        var perturbed = model.EncodeAudio(AsTensor(withTone));

        double worst = 0.0;
        for (int i = 0; i < baseline.Length; i++)
            worst = Math.Max(worst, Math.Abs(baseline[i] - perturbed[i]));

        Assert.True(worst > 1e-6,
            $"A 1 kHz tone moved the embedding by only {worst:R}; the control arm is dead, so the "
                + "fmax test above proves nothing.");
    }

    /// <summary>
    /// HTSAT normalizes the mel image over its frequency bins before patching
    /// (<c>htsat.py</c>: <c>self.bn0 = nn.BatchNorm2d(mel_bins)</c>). Without it the first
    /// projection sees raw decibels, whose large negative mean saturates it.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task TheAudioTowerNormalizesTheMelBinsBeforePatching()
    {
        await Task.Yield();

        var audio = AudioTower();

        Assert.NotEmpty(audio);
        var first = Assert.IsType<BatchNormalizationLayer<double>>(audio[0]);
        Assert.Equal(BatchNormDataLayout.ChannelsLast, first.Layout);
        Assert.IsType<PatchEmbeddingLayer<double>>(audio[1]);
    }

    /// <summary>
    /// Both reference projections are <c>Linear -> ReLU -> Linear</c>
    /// (<c>model.py</c>: <c>audio_projection</c> / <c>text_projection</c>), not one linear map.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task BothTowersProjectThroughTheReferenceMlp()
    {
        await Task.Yield();

        AssertMlpTail(AudioTower(), "audio");
        AssertMlpTail(TextTower(), "text");
    }

    /// <summary>
    /// RoBERTa learns its position embeddings rather than taking Vaswani's sinusoid, which
    /// contributed no parameters at all to the text tower.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task TheTextTowerLearnsItsPositions()
    {
        await Task.Yield();

        var text = TextTower();

        Assert.Contains(text, layer => layer is LearnedPositionalEmbeddingLayer<double>);
        Assert.DoesNotContain(text, layer => layer is PositionalEncodingLayer<double>);
    }

    private static List<ILayer<double>> AudioTower()
    {
        var options = SmallOptions();
        return LayerHelper<double>.CreateDefaultCLAPAudioEncoderLayers(
            audioHiddenDim: options.AudioHiddenDim,
            audioEncoderLayers: options.AudioEncoderLayers,
            audioEncoderHeads: options.AudioEncoderHeads,
            swinWindowSize: options.SwinWindowSize,
            projectionDim: options.ProjectionDim,
            audioPatchSize: options.AudioPatchSize,
            melBins: options.NumMelBands).ToList();
    }

    private static List<ILayer<double>> TextTower()
    {
        var options = SmallOptions();
        return LayerHelper<double>.CreateDefaultCLAPTextEncoderLayers(
            vocabSize: options.VocabSize,
            maxTextLength: options.MaxTextLength,
            textHiddenDim: options.TextHiddenDim,
            textEncoderLayers: options.TextEncoderLayers,
            textEncoderHeads: options.TextEncoderHeads,
            projectionDim: options.ProjectionDim).ToList();
    }

    private static void AssertMlpTail(List<ILayer<double>> layers, string tower)
    {
        Assert.True(layers.Count >= 2, $"The {tower} tower has too few layers to carry a projection.");
        var last = layers[layers.Count - 1] as DenseLayer<double>;
        var penultimate = layers[layers.Count - 2] as DenseLayer<double>;

        Assert.True(last is not null,
            $"The {tower} tower's last layer is {layers[layers.Count - 1].GetType().Name}, not a projection.");
        Assert.True(penultimate is not null,
            $"The {tower} tower projects through one layer only; the reference uses "
                + "Linear -> ReLU -> Linear.");
        Assert.Contains(ActivationFunction.ReLU, penultimate.GetActivationTypes());
    }

    private static double[] Tone(int samples, double frequency, int sampleRate)
    {
        var buffer = new double[samples];
        for (int i = 0; i < samples; i++)
            buffer[i] = Math.Sin(2.0 * Math.PI * frequency * i / sampleRate);
        return buffer;
    }

    private static Tensor<double> AsTensor(double[] audio)
    {
        var tensor = new Tensor<double>([1, audio.Length]);
        for (int i = 0; i < audio.Length; i++)
            tensor[i] = audio[i];
        return tensor;
    }
}
