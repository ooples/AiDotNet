using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Generative;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Behavioural checks of the decoder-only KOSMOS rebuild, each of which a plausible wiring mistake would fail:
/// <list type="bullet">
/// <item>Causal masking: a later token cannot change earlier logits.</item>
/// <item>The image embeddings reach the language model.</item>
/// <item>Caption training lowers next-token cross-entropy.</item>
/// <item>KOSMOS-1's xPos makes attention depend on relative position.</item>
/// <item>Generation terminates.</item>
/// </list>
/// </summary>
public class KosmosBehaviourTests
{
    private static NeuralNetworkArchitecture<double> Architecture() => new(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
        inputHeight: 28, inputWidth: 28, inputDepth: 3, outputSize: 64);

    private static T Small<T>(T options) where T : KosmosOptions
    {
        options.ImageSize = 28; options.PatchSize = 14; options.VisionDim = 16; options.VisionHeads = 2;
        options.DecoderDim = 16; options.NumHeads = 2; options.NumVisionLayers = 1; options.NumDecoderLayers = 2;
        options.DecoderFeedForwardDim = 32; options.NumImageTokens = 4; options.VocabSize = 40;
        options.MaxSequenceLength = 12; options.MaxGenerationLength = 5;
        return options;
    }

    private static Tensor<double> Image(int seed)
    {
        var rng = new Random(seed);
        var image = new Tensor<double>(new[] { 3, 28, 28 });
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble();
        return image;
    }

    [Fact]
    public void Predict_ReturnsLogitsForTheImageBlock()
    {
        using var model = new KOSMOS2<double>(Architecture(), Small(new KOSMOS2Options()));
        using var image = Image(1);
        using var logits = model.Predict(image);
        // <s> <image> [4 image slots] </image>
        Assert.Equal(new[] { 4 + 3, 40 }, logits.Shape.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoder_IsCausal(bool kosmos1)
    {
        using var model = kosmos1
            ? (IDisposable)new KOSMOS1<double>(Architecture(), Small(new KOSMOS1Options { ResamplerDepth = 1 }))
            : new KOSMOS2<double>(Architecture(), Small(new KOSMOS2Options()));
        using var image = Image(2);
        var a = Predict(model, image, new[] { 5, 6, 7, 8 });
        var b = Predict(model, image, new[] { 5, 6, 7, 30 });
        int header = 4 + 3;
        for (int row = 0; row < header + 3; row++)
            for (int v = 0; v < 40; v++)
                Assert.True(Math.Abs(a[row, v] - b[row, v]) < 1e-9, $"row {row} changed when only the last token changed: not causal");
        double last = Enumerable.Range(0, 40).Sum(v => Math.Abs(a[header + 3, v] - b[header + 3, v]));
        Assert.True(last > 1e-9, "The changed token did not affect its own row: the decoder ignores its input.");
    }

    [Fact]
    public void ImageEmbeddings_ReachTheLanguageModel()
    {
        using var model = new KOSMOS2<double>(Architecture(), Small(new KOSMOS2Options()));
        using var first = Image(3);
        using var second = Image(4);
        var a = model.PredictTokens(first, new[] { 5, 6 });
        var b = model.PredictTokens(second, new[] { 5, 6 });
        int lastRow = a.Shape[0] - 1;
        double diff = Enumerable.Range(0, 40).Sum(v => Math.Abs(a[lastRow, v] - b[lastRow, v]));
        Assert.True(diff > 1e-8, "Two different images gave the same caption logits: the image slots are disconnected.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CaptionTraining_LowersCrossEntropy(bool kosmos1)
    {
        var model = kosmos1
            ? new VisionLanguageModelBaseHandle(new KOSMOS1<double>(Architecture(), Small(new KOSMOS1Options { ResamplerDepth = 1 })))
            : new VisionLanguageModelBaseHandle(new KOSMOS2<double>(Architecture(), Small(new KOSMOS2Options())));
        using var _ = model;
        using var image = Image(5);
        int[] caption = { 9, 14, 3, 21, 2 };
        model.TrainCaption(image, caption);
        double first = model.LastLoss;
        for (int step = 0; step < 9; step++) model.TrainCaption(image, caption);
        double last = model.LastLoss;
        Assert.False(double.IsNaN(last) || double.IsInfinity(last));
        Assert.True(last < first, $"10 caption steps did not lower the cross-entropy: {first} -> {last}");
    }

    [Fact]
    public void XPos_MakesAttentionDependOnRelativePosition()
    {
        // Without any position signal, attention is permutation-invariant over its keys: row 2 of [a, b, c] and
        // of [b, a, c] has the same query (c) and the same key set, so the logits would be equal. xPos rotates
        // q and k by position, so the two orders must differ. (Identical tokens cannot test this: every value
        // is the same vector, so every attention weighting gives the same row.)
        using var decoder = new MagnetoDecoderLayer<double>(20, 8, 1, 2, 16, 32, KosmosPositionEncoding.XPos);
        using var abc = Ids(3, 11, 17);
        using var bac = Ids(11, 3, 17);
        using var first = decoder.Forward(abc);
        using var second = decoder.Forward(bac);
        double diff = Enumerable.Range(0, 20).Sum(v => Math.Abs(first[2, v] - second[2, v]));
        Assert.True(diff > 1e-8, "Reordering the context left the last row unchanged: xPos is not applied.");
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var t = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) t[i] = ids[i];
        return t;
    }
    [Fact]
    public void GenerateFromImage_StopsWithinTheLimit()
    {
        using var model = new KOSMOS2<double>(Architecture(), Small(new KOSMOS2Options()));
        using var image = Image(6);
        using var tokens = model.GenerateFromImage(image);
        Assert.InRange(tokens.Length, 1, 5);
        for (int i = 0; i < tokens.Length; i++) Assert.InRange(tokens[i], 0, 39);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void The_decoder_is_counted_and_checkpointed(bool kosmos1)
    {
        // The decoder holds the token embeddings and every MAGNETO block. Outside the model's parameter walk it was
        // missing from ParameterCount and from every checkpoint, so a reloaded model got a fresh decoder.
        IDisposable Build() => kosmos1
            ? new KOSMOS1<double>(Architecture(), Small(new KOSMOS1Options { ResamplerDepth = 1 }))
            : new KOSMOS2<double>(Architecture(), Small(new KOSMOS2Options()));
        using var trained = Build();
        using var image = Image(6);
        int[] caption = { 9, 14, 3, 21, 2 };
        var handle = trained is KOSMOS1<double> t1 ? new VisionLanguageModelBaseHandle(t1) : new VisionLanguageModelBaseHandle((KOSMOS2<double>)trained);
        for (int step = 0; step < 3; step++) handle.TrainCaption(image, caption);
        var network = (NeuralNetworkBase<double>)trained;
        // 40 x 16 token embeddings alone exceed what the vision encoder and resampler of this size hold together.
        Assert.True(network.ParameterCount > 40 * 16, $"ParameterCount {network.ParameterCount} leaves out the decoder");

        using var reloaded = Build();
        ((NeuralNetworkBase<double>)reloaded).Deserialize(network.Serialize());
        var expected = Predict(trained, image, new[] { 9, 14 });
        var actual = Predict(reloaded, image, new[] { 9, 14 });
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-9, $"logit {i} differs after a checkpoint round trip: the decoder was not saved");
    }
    private static Tensor<double> Predict(IDisposable model, Tensor<double> image, int[] prompt) => model switch
    {
        KOSMOS1<double> k1 => k1.PredictTokens(image, prompt),
        KOSMOS2<double> k2 => k2.PredictTokens(image, prompt),
        _ => throw new InvalidOperationException()
    };

    /// <summary>Uniform access to the two model types' training API.</summary>
    private sealed class VisionLanguageModelBaseHandle : IDisposable
    {
        private readonly KOSMOS1<double>? _k1;
        private readonly KOSMOS2<double>? _k2;
        public VisionLanguageModelBaseHandle(KOSMOS1<double> model) => _k1 = model;
        public VisionLanguageModelBaseHandle(KOSMOS2<double> model) => _k2 = model;

        public void TrainCaption(Tensor<double> image, int[] caption)
        {
            if (_k1 is not null) _k1.TrainCaption(image, caption);
            else _k2?.TrainCaption(image, caption);
        }

        public double LastLoss => _k1 is not null ? _k1.GetLastLoss() : _k2 is not null ? _k2.GetLastLoss() : double.NaN;

        public void Dispose()
        {
            _k1?.Dispose();
            _k2?.Dispose();
        }
    }
}
