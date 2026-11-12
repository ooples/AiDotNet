using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using AiDotNet.Video.Segmentation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// SAM 2's public surface: the released sizes, the paper's optimizer, prompted tracking at the frames' size, and training.
/// The network itself is checked against the official code in <c>Sam2VideoReferenceTests</c>.
/// </summary>
public class SAM2ModelTests
{
    private static NeuralNetworkArchitecture<double> Architecture(int height, int width) => new(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.BinaryClassification,
        inputHeight: height, inputWidth: width, inputDepth: 3, outputSize: 1);

    private static SAM2Options Small() => new()
    {
        ModelSize = SAM2ModelSize.Tiny, ImageSize = 32, HieraEmbeddingDimension = 8, HieraStageDepths = new[] { 1, 1, 1, 1 },
        HieraInitialHeadCount = 1, HieraWindowSizes = new[] { 4, 4, 2, 1 }, HieraGlobalAttentionBlockIndexes = new[] { 2 },
        ModelDimension = 32, MemoryDimension = 16, DecoderHeadCount = 2, MaskDecoderMlpDimension = 32, MemoryAttentionLayerCount = 1,
        MemoryAttentionFeedForwardDimension = 32, MemoryFuserDimension = 32, MemoryBankSize = 3, DropPath = 0.0, Seed = 3,
    };

    private static List<Tensor<double>> Video(int frames, int height, int width)
    {
        var video = new List<Tensor<double>>();
        for (int t = 0; t < frames; t++)
        {
            var frame = new Tensor<double>(new[] { 3, height, width });
            for (int i = 0; i < frame.Length; i++) frame[i] = 0.5 + 0.5 * Math.Sin(0.37 * i + t);
            video.Add(frame);
        }
        return video;
    }

    [Fact]
    public void ReleasedSizes_MatchTheOfficialConfigs()
    {
        // facebookresearch/sam2 configs (sam2.1_hiera_{t,s,b+,l}.yaml and Hiera's defaults for B+), and the paper's drop path.
        var tiny = Sam2Configuration.Hiera(0);
        Assert.Equal(new[] { 96, 192, 384, 768 }, tiny.EmbedDims);
        Assert.Equal(new[] { 1, 2, 4, 8 }, tiny.HeadsPerStage);
        Assert.Equal(new[] { 1, 2, 7, 2 }, tiny.BlocksPerStage);
        Assert.Equal(new[] { 5, 7, 9 }, tiny.GlobalAttentionBlocks);
        Assert.Equal(7, tiny.BackgroundPositionSize);
        Assert.Equal(0.1, tiny.DropPath);
        var small = Sam2Configuration.Hiera(1);
        Assert.Equal(new[] { 1, 2, 11, 2 }, small.BlocksPerStage);
        Assert.Equal(new[] { 7, 10, 13 }, small.GlobalAttentionBlocks);
        var basePlus = Sam2Configuration.Hiera(2);
        Assert.Equal(new[] { 112, 224, 448, 896 }, basePlus.EmbedDims);
        Assert.Equal(new[] { 2, 4, 8, 16 }, basePlus.HeadsPerStage);
        Assert.Equal(new[] { 2, 3, 16, 3 }, basePlus.BlocksPerStage);
        Assert.Equal(new[] { 12, 16, 20 }, basePlus.GlobalAttentionBlocks);
        Assert.Equal(new[] { 8, 4, 14, 7 }, basePlus.WindowSizes);
        Assert.Equal(14, basePlus.BackgroundPositionSize);
        Assert.Equal(0.2, basePlus.DropPath);
        var large = Sam2Configuration.Hiera(3);
        Assert.Equal(new[] { 144, 288, 576, 1152 }, large.EmbedDims);
        Assert.Equal(new[] { 2, 6, 36, 4 }, large.BlocksPerStage);
        Assert.Equal(new[] { 23, 33, 43 }, large.GlobalAttentionBlocks);
        Assert.Equal(new[] { 8, 4, 16, 8 }, large.WindowSizes);
        Assert.Equal(0.3, large.DropPath);
        Assert.Equal(new[] { 0.8, 0.8, 0.9, 0.925 }, Enumerable.Range(0, 4).Select(Sam2Recipe.PaperLayerDecay).ToArray());
        Assert.False(new Sam2MemoryConfiguration().TemporalPointerEncoding);
        Assert.True(Sam2MemoryConfiguration.Sam21.TemporalPointerEncoding && Sam2MemoryConfiguration.Sam21.OcclusionEmbedding);
    }

    [Fact]
    public void Optimizer_FollowsThePaperRecipe()
    {
        var layers = new List<LayerBase<double>>();
        var config = Sam2VideoReferenceTests_Configuration();
        var net = new Sam2Network<double>(AiDotNetEngine.Current, layers, config);
        net.InitializeLikeReference(new Random(1));
        var recipe = new Sam2Recipe { LayerDecay = 0.8 };
        var (parameters, optimizer) = Sam2Optimizer.Create(net, layers, recipe);
        var options = (AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>)optimizer.GetOptions();
        var rates = options.LearningRateMask!;
        var decays = options.WeightDecayMask!;
        Assert.Equal(parameters.Sum(p => p.Length), rates.Length);
        int Offset(Tensor<double> tensor)
        {
            int offset = 0;
            foreach (var p in parameters)
            {
                if (ReferenceEquals(p, tensor)) return offset;
                offset += p.Length;
            }
            throw new InvalidOperationException("Not a trained parameter.");
        }
        Tensor<double> Weight(LayerBase<double> layer) => AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new ILayer<double>[] { layer }).First(p => p.Rank == 2);
        Tensor<double> Bias(LayerBase<double> layer) => AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new ILayer<double>[] { layer }).First(p => p.Rank == 1);
        int blocks = net.Blocks.Count;
        double encoder = recipe.ImageEncoderLearningRate / recipe.LearningRate;
        Assert.Equal(encoder * Math.Pow(0.8, blocks + 1), rates[Offset(Weight(net.PatchProjection))], 12);
        Assert.Equal(encoder * Math.Pow(0.8, blocks), rates[Offset(Weight(net.Blocks[0].Qkv))], 12);
        Assert.Equal(encoder * 0.8, rates[Offset(Weight(net.Blocks[^1].Mlp2))], 12);
        Assert.Equal(encoder, rates[Offset(Weight(net.BackgroundPosition))], 12);
        Assert.Equal(encoder, rates[Offset(Weight(net.NeckConvs[0]))], 12);
        Assert.Equal(1.0, rates[Offset(Weight(net.HighRes0))], 12);
        Assert.Equal(1.0, rates[Offset(Weight(net.MemoryProjection!))], 12);
        Assert.Equal(1.0, decays[Offset(Weight(net.Blocks[0].Qkv))]);
        Assert.Equal(0.0, decays[Offset(Bias(net.Blocks[0].Qkv))]);
        Assert.Equal(0.0, decays[Offset(AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new ILayer<double>[] { net.Blocks[0].Norm1 })[0])]);
        Assert.Equal(0.0, decays[Offset(Weight(net.Fuser[0].DepthwiseBias))]);
        Assert.Equal(1.0, decays[Offset(Weight(net.Fuser[0].DepthwiseWeight))]);
        Assert.Equal(0.1, options.WeightDecay);
        Assert.Equal(0.1, options.MaxGradientNorm);
    }

    private static Sam2Configuration Sam2VideoReferenceTests_Configuration()
        => AiDotNet.Tests.ComputerVision.Sam2VideoReferenceTests.SmallConfiguration(false);

    [Fact(Timeout = 600000)]
    public async Task PromptedTracking_GivesProbabilitiesAtTheFramesSize()
    {
        await Task.Yield();
        using var model = new SAM2<double>(Architecture(40, 48), options: Small());
        var frames = Video(3, 40, 48);
        var mask = new Tensor<double>(new[] { 40, 48 });
        for (int y = 10; y < 25; y++) for (int x = 12; x < 30; x++) mask[y, x] = 1.0;
        var results = new[]
        {
            model.TrackObject(frames, new float[,] { { 20, 15 } }, new[] { 1 }),
            model.TrackObjectWithBox(frames, new float[] { 10, 8, 30, 26 }),
            model.TrackObjectWithMask(frames, mask),
            model.InteractiveVideoSegmentation(frames, new Dictionary<int, (float[,], int[])>
            {
                [1] = (new float[,] { { 20, 15 }, { 40, 30 } }, new[] { 1, 0 }),
            }),
        };
        foreach (var masks in results)
        {
            Assert.Equal(3, masks.Count);
            foreach (var m in masks)
            {
                Assert.Equal(new[] { 40, 48 }, m.Shape.ToArray());
                Assert.All(m.ToArray(), v => Assert.InRange(v, 0.0, 1.0));
            }
        }
        // A mask prompt is taken as the frame's output: probabilities sigmoid(±10) on and off the mask.
        var prompted = results[2][0];
        Assert.True(prompted[15, 20] > 0.99 && prompted[35, 5] < 0.01, $"{prompted[15, 20]} {prompted[35, 5]}");
        var image = model.SegmentWithPoints(frames[0], new float[,] { { 20, 15 } }, new[] { 1 });
        Assert.Equal(new[] { 40, 48 }, image.Shape.ToArray());
        double occlusion = model.GetOcclusionScore(frames[0], new float[,] { { 20, 15 } }, new[] { 1 });
        Assert.InRange(occlusion, 0.0, 1.0);
    }

    [Fact(Timeout = 600000)]
    public async Task Training_LowersTheObjective()
    {
        await Task.Yield();
        var options = Small();
        options.LearningRate = 1e-3;
        options.ImageEncoderLearningRate = 1e-3;
        options.WarmupSteps = 0;
        options.TrainingSteps = 1000;
        options.CorrectionPoints = 1;
        using var model = new SAM2<double>(Architecture(32, 32), options: options);
        var frames = Video(2, 32, 32);
        var video = new Tensor<double>(new[] { 2, 3, 32, 32 }, new Vector<double>(frames.SelectMany(f => f.ToArray()).ToArray()));
        var masks = new Tensor<double>(new[] { 2, 32, 32 });
        for (int t = 0; t < 2; t++) for (int y = 8; y < 20; y++) for (int x = 10 + t; x < 24 + t; x++) masks[t, y, x] = 1.0;
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(video, masks);
        for (int i = 0; i < 15; i++) model.Train(video, masks);
        double after = objective.EvaluateTrainingObjective(video, masks);
        Assert.True(after < before, $"The objective did not fall: {before} -> {after}");
    }
}
