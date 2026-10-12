using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
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
    private static NeuralNetworkArchitecture<double> ClapArchitecture() =>
        new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2048,
            outputSize: 8);

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
    /// 0 Hz to Nyquist. A 20 kHz tone therefore falls outside every mel band and may reach the
    /// encoder only as window leakage, while a 1 kHz tone sits inside one. Both are measured on
    /// the SAME model, so the comparison does not depend on the weights either one happens to get.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ContentAboveTheReferenceFmaxBarelyReachesTheEmbedding()
    {
        await Task.Yield();

        var model = new CLAPModel<double>(ClapArchitecture(), SmallOptions());
        model.SetTrainingMode(false);

        double outOfBand = SensitivityToAddedTone(model, 20_000.0);
        double inBand = SensitivityToAddedTone(model, 1_000.0);

        Assert.True(inBand > 1e-6,
            $"A 1 kHz tone moved the embedding by only {inBand:R}; the control arm is dead, so the "
                + "comparison below proves nothing.");
        Assert.True(outOfBand < inBand * 0.25,
            $"A 20 kHz tone moved the embedding by {outOfBand:R} against the 1 kHz tone's "
                + $"{inBand:R} (ratio {outOfBand / inBand:F3}). With fmax at the reference's "
                + "14 kHz, out-of-band content should only reach the encoder as window leakage.");
    }

    private static double SensitivityToAddedTone(CLAPModel<double> model, double frequency)
    {
        const int samples = 2048;
        var plain = Tone(samples, 440.0, 48_000);
        var mixed = Tone(samples, 440.0, 48_000);
        var added = Tone(samples, frequency, 48_000);
        for (int i = 0; i < samples; i++)
            mixed[i] += 0.5 * added[i];

        var baseline = model.EncodeAudio(AsTensor(plain));
        var perturbed = model.EncodeAudio(AsTensor(mixed));

        double worst = 0.0;
        for (int i = 0; i < baseline.Length; i++)
            worst = Math.Max(worst, Math.Abs(baseline[i] - perturbed[i]));
        return worst;
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
