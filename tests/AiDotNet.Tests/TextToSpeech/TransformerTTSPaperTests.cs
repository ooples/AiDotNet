using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Transformer TTS trains and synthesizes as its paper specifies (Li et al. 2019): Tacotron 2's pre-nets with linear
/// projections, scaled positional encodings with trainable weights, a Transformer encoder and causally masked decoder,
/// mel and stop linears, and the post-net; mel MSE before and after the post-net plus weighted stop-token BCE.
/// </summary>
/// <remarks>
/// Before this change Transformer TTS had none of these components: its synthesis derived a mel length from a fixed
/// formula on encoder values, and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class TransformerTTSPaperTests
{
    private const int MelBins = 8;

    private static TransformerTTS<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins),
        new TransformerTTSOptions
        {
            EncoderDim = 16, HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, FeedForwardDim = 32,
            EncoderPrenetLayers = 1, EncoderPrenetChannels = 16, DecoderPrenetSizes = new[] { 16, 16 }, PrenetDropout = 0.0,
            PostnetLayers = 2, PostnetDim = 16, PostnetDropout = 0.0, DropoutRate = 0.0, MelChannels = MelBins,
            MaxDecoderSteps = 7,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static Tensor<double> Mel(int frames)
    {
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return mel;
    }

    [Fact(Timeout = 60000)]
    public async Task DecoderSelfAttention_IsCausal()
    {
        await Task.Yield();
        // Changing a later target frame must not change any earlier output.
        var block = new PostNormTransformerDecoderBlock<double>(hiddenSize: 8, numHeads: 2, ffnDim: 16, dropoutRate: 0.0);
        block.SetTrainingMode(false);
        var memory = new Tensor<double>(new[] { 4, 8 });
        for (int i = 0; i < memory.Length; i++) memory[i] = Math.Cos(0.3 * i);
        var target = new Tensor<double>(new[] { 5, 8 });
        for (int i = 0; i < target.Length; i++) target[i] = Math.Sin(0.7 * i);
        var first = block.Forward(target, memory);
        var changed = new Tensor<double>(target._shape, target.ToVector());
        for (int c = 0; c < 8; c++) changed[4, c] += 3.0;
        var second = block.Forward(changed, memory);
        for (int t = 0; t < 4; t++)
            for (int c = 0; c < 8; c++) Assert.Equal(first[t, c], second[t, c], 10);
        Assert.True(Enumerable.Range(0, 8).Any(c => Math.Abs(first[4, c] - second[4, c]) > 1e-6));
    }

    [Fact(Timeout = 120000)]
    public async Task TeacherForcedTraining_ReducesTheObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = new TtsTrainingSample<double> { Tokens = Tokens(5), Mel = Mel(9) };
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample.Tokens, sample.Mel!);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_StopsAtTheStopTokenOrTheStepLimit()
    {
        await Task.Yield();
        var mel = CreateModel().Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        Assert.InRange(mel.Shape[0], 1, 7);
        for (int i = 0; i < mel.Length; i++) Assert.True(double.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }
}
