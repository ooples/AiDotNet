using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ComputerVision;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// One GLaMM training sample against the official computation assembled from its parts' references
/// (tools/reference-data/glamm_reference.py): the tokenized conversation and its labels, the multimodal sequence with a
/// region token, the [SEG] masks from SAM, each loss term and the total, before and after one SGD step.
/// </summary>
public class GlammReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "glamm_reference.json";
        string relative = Path.Combine("VisionLanguage", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    /// <summary>The fixture's weights: generated from each name (the same rule as glamm_reference.py synthetic()).</summary>
    internal static double[] Synthetic(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var u = EastReferenceTests.Uniforms(name, count);
        if (name.EndsWith(".bias", StringComparison.Ordinal) || name.EndsWith("_bias", StringComparison.Ordinal))
            return u.Select(v => 0.1 * (2 * v - 1)).ToArray();
        if (shape.Length == 1) return u.Select(v => 0.5 + v).ToArray();
        double bound = Math.Sqrt(6.0 / (count / shape[0]));
        return u.Select(v => bound * (2 * v - 1)).ToArray();
    }

    internal static GlammConfiguration SmallConfiguration(int vocabulary) => new()
    {
        Clip = new ClipVisionConfiguration { ImageSize = 28, PatchSize = 7, Hidden = 64, Layers = 10, Heads = 4, Intermediate = 128 },
        Llm = new LlamaConfiguration { VocabSize = vocabulary, Hidden = 32, Layers = 2, Heads = 4, KeyValueHeads = 4, Intermediate = 64, LoraRank = 4, LoraAlpha = 8, LoraDropout = 0 },
        Region = new GlammRegionConfiguration { Embed = 64, Grid = 4, RegionWidth = 32, OutDims = 32 },
        Sam = new SamConfiguration
        {
            ImageSize = 64, PatchSize = 16, EncoderDim = 32, EncoderDepth = 2, EncoderHeads = 2, WindowSize = 2, GlobalAttention = new[] { 1 },
            OutChannels = 16, MaskInChannels = 8, DecoderDepth = 2, DecoderHeads = 2, DecoderMlp = 32, IouHidden = 16,
        },
        OutDim = 16,
    };

    private static Tensor<double> Uniform(string name, int[] shape, Func<double, double> map)
        => new(shape, new Vector<double>(EastReferenceTests.Uniforms(name, shape.Aggregate(1, (a, b) => a * b)).Select(map).ToArray()));

    [Fact(Timeout = 600000)]
    public async Task TrainingSample_MatchesTheOfficialComputation_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Fixture();
        var tokenizer = new GlammTokenizer(new SentencePieceModel(GlammTokenizerReferenceTests.TestModel()));
        string conversation = k.GetProperty("conversation").GetString()!;
        var ids = tokenizer.EncodeWithImage(conversation);
        Assert.Equal(k.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray(), ids.ToArray());
        var labels = tokenizer.MaskInstructions(conversation, ids);
        Assert.Equal(k.GetProperty("labels").EnumerateArray().Select(v => v.GetInt32()).ToArray(), labels);

        var layers = new List<LayerBase<double>>();
        var net = new GlammNetwork<double>(engine, layers, SmallConfiguration(tokenizer.VocabularySize));
        net.LoadTorchWeights(Synthetic);
        var globalImage = Uniform("global_image", new[] { 1, 3, 28, 28 }, v => 2 * v - 1);
        var samImage = Uniform("sam_image", new[] { 1, 3, 64, 64 }, v => 2 * v - 1);
        var gtMasks = Uniform("gt_masks", new[] { 2, 24, 32 }, v => v > 0.5 ? 1.0 : 0.0);
        var boxes = k.GetProperty("boxes").EnumerateArray().Select(b => b.EnumerateArray().Select(v => v.GetDouble()).ToArray()).ToArray();

        (Tensor<double> Ce, Tensor<double> Bce, Tensor<double> Dice, Tensor<double> Total, Tensor<double> Masks) Run()
        {
            var (features, levels) = net.EncodeGlobal(globalImage);
            var regions = net.Region.Forward(levels, new[] { boxes });
            var imageFeatures = engine.Reshape(features, new[] { features.Shape[1], features.Shape[2] });
            var (embeddings, _, fullLabels) = net.Multimodal(ids, labels, imageFeatures, regions, tokenizer.BboxId);
            var (hidden, logits) = net.Llm.Forward(embeddings);
            var labels2D = new int[1, fullLabels!.Length];
            for (int i = 0; i < fullLabels.Length; i++) labels2D[0, i] = fullLabels[i];
            var ce = net.Llm.CausalLmLoss(logits, labels2D);
            var seg = net.SegEmbeddings(hidden, ids, tokenizer.SegId, imageFeatures.Shape[0])!;
            var samEmbedding = net.Sam.EncodeImage(samImage);
            var masks = net.DecodeMasks(seg, samEmbedding, 48, 64, 24, 32);
            var (bce, dice) = net.MaskLosses(masks, gtMasks);
            var total = engine.TensorAdd(ce, engine.TensorAdd(engine.TensorMultiplyScalar(bce, 2.0), engine.TensorMultiplyScalar(dice, 0.5)));
            return (ce, bce, dice, total, masks);
        }

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        (Tensor<double> Ce, Tensor<double> Bce, Tensor<double> Dice, Tensor<double> Total, Tensor<double> Masks) result;
        using (var tape = new GradientTape<double>())
        {
            result = Run();
            grads = tape.ComputeGradients(result.Total, sources: parameters);
        }
        void Close(double expected, double actual, string what)
            => Assert.True(Math.Abs(expected - actual) < 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: {expected} vs {actual}");
        var expectedMasks = k.GetProperty("mask_logits").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        Assert.Equal(expectedMasks.Length, result.Masks.Length);
        for (int i = 0; i < expectedMasks.Length; i++) Close(expectedMasks[i], result.Masks[i], $"mask logit {i}");
        Close(k.GetProperty("ce").GetDouble(), result.Ce[0], "ce");
        Close(k.GetProperty("bce").GetDouble(), 2.0 * result.Bce[0], "2 bce");
        Close(k.GetProperty("dice").GetDouble(), 0.5 * result.Dice[0], "0.5 dice");
        Close(k.GetProperty("loss").GetDouble(), result.Total[0], "total");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.001 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Run().Total[0];
        Close(k.GetProperty("loss_after_sgd").GetDouble(), after, "loss after SGD");
    }
}
