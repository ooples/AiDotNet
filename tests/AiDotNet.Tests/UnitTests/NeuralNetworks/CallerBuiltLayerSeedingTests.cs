using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A caller who builds the layer stack (<c>Architecture.Layers</c>) constructs every layer before the model exists,
/// so the model's seed cannot reach a layer that draws its weights in its constructor. MambaBlock and RWKV7Block
/// did exactly that; they now draw on first use, and the model seeds caller-built layers at construction, so equal
/// model seeds reproduce the stack's initial weights.
///
/// <para>
/// The same shape reaches any layer that fixes its STREAM in its constructor, even one whose weights are
/// allocated lazily: T5RelativeBiasAttentionLayer keeps Q/K/V/O as [0, 0] placeholders but draws its bias table
/// up front, so the stream the projections later continue was opened before any seed existed.
/// </para>
/// </summary>
public class CallerBuiltLayerSeedingTests
{
    private const int Sequence = 4;
    private const int Width = 8;

    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class SuppliedStack : NeuralNetworkBase<double>
    {
        public SuppliedStack(NeuralNetworkArchitecture<double> architecture)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            InitializeLayers();
        }

        protected override void InitializeLayers() => Layers.AddRange(Architecture.Layers ?? new List<ILayer<double>>());

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy() => new SuppliedStack(Architecture);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(SuppliedStack),
            Description = "Test double that runs exactly the layers its caller supplied.",
        };
    }

    private static double[] InitialWeights(int seed)
    {
        // A fresh stack per model, built by the caller before any model or seed exists.
        var layers = new List<ILayer<double>>
        {
            new MambaBlock<double>(Sequence, modelDimension: Width, stateDimension: 4),
            new RWKV7Block<double>(Sequence, modelDimension: Width, numHeads: 2),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: Sequence, inputWidth: Width, outputSize: Width, layers: layers)
        {
            RandomSeed = seed,
        };

        var model = new SuppliedStack(architecture);
        model.MaterializeParameters();
        return model.GetParameters().ToArray();
    }

    private static double[] InitialT5Weights(int seed)
    {
        // Built by the caller before any model or seed exists, exactly as the SSM stack above is.
        var layers = new List<ILayer<double>>
        {
            new T5RelativeBiasAttentionLayer<double>(hiddenSize: Width, numHeads: 2),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: 1, inputWidth: Width, outputSize: Width, layers: layers)
        {
            RandomSeed = seed,
        };

        var model = new SuppliedStack(architecture);
        model.MaterializeParameters();
        return model.GetParameters().ToArray();
    }

    private static double[] T5LayerWeights(int? constructorSeed, params int[] assignedSeeds)
    {
        var layer = new T5RelativeBiasAttentionLayer<double>(hiddenSize: Width, numHeads: 2, seed: constructorSeed);
        foreach (var s in assignedSeeds)
        {
            layer.RandomSeed = s;
        }

        // Reading the parameters allocates Q/K/V/O, which is where the stream the seed restarted is consumed.
        return layer.GetParameters().ToArray();
    }

    [Fact]
    public void AnExplicitConstructorSeed_IsNotOverriddenByAnAssignedSeed()
    {
        // The caller chose the constructor seed, so the same one must win whatever is wired on afterwards: both
        // assigned seeds give exactly the constructor-seeded weights, and a different constructor seed does not.
        var baseline = T5LayerWeights(7);
        Assert.NotEmpty(baseline);
        Assert.Equal(baseline, T5LayerWeights(7, 2290));
        Assert.Equal(baseline, T5LayerWeights(7, 2291));
        Assert.NotEqual(baseline, T5LayerWeights(8));
    }

    [Fact]
    public void ASeedChangedBeforeAllocation_IsTheOneTheProjectionsUse()
    {
        // Assigning 2290 and then 2291 before anything is allocated has to land on 2291, not quietly keep 2290.
        Assert.Equal(T5LayerWeights(null, 2291), T5LayerWeights(null, 2290, 2291));
        Assert.NotEqual(T5LayerWeights(null, 2290), T5LayerWeights(null, 2290, 2291));
    }

    [Fact]
    public void ASharedBiasTable_StillSeedsTheProjections()
    {
        // The shared table is not the layer's to redraw, but Q/K/V/O draw from the same stream, so an assigned
        // seed still has to reach them: two layers sharing a table at one seed must agree.
        static double[] WithSharedTable(int assigned)
        {
            var owner = new T5RelativeBiasAttentionLayer<double>(hiddenSize: Width, numHeads: 2, seed: 11);
            var layer = new T5RelativeBiasAttentionLayer<double>(
                hiddenSize: Width, numHeads: 2, sharedRelativeBiasTable: owner.GetRelativeBiasTable())
            {
                RandomSeed = assigned,
            };
            return layer.GetParameters().ToArray();
        }

        Assert.Equal(WithSharedTable(2290), WithSharedTable(2290));
        Assert.NotEqual(WithSharedTable(2290), WithSharedTable(2291));
    }

    [Fact]
    public void EqualModelSeeds_GiveEqualInitialWeights_ForACallerBuiltT5Stack()
    {
        var first = InitialT5Weights(2290);

        Assert.NotEmpty(first);
        Assert.Equal(first, InitialT5Weights(2290));
    }

    [Fact]
    public void DifferentModelSeeds_GiveDifferentInitialWeights_ForACallerBuiltT5Stack()
    {
        Assert.NotEqual(InitialT5Weights(2290), InitialT5Weights(2291));
    }

    // A layer built by a caller, then handed a seed the way a model wires one on. Reading the parameters
    // forces whatever allocation the layer defers.
    private static double[] DrawnAtAssignedSeed(Func<LayerBase<double>> make, int assigned)
    {
        var layer = make();
        layer.RandomSeed = assigned;
        return layer.GetParameters().ToArray();
    }

    private static void AssertSeedReachesTheWeights(Func<LayerBase<double>> make)
    {
        var first = DrawnAtAssignedSeed(make, 2290);
        Assert.NotEmpty(first);
        Assert.Equal(first, DrawnAtAssignedSeed(make, 2290));
        Assert.NotEqual(first, DrawnAtAssignedSeed(make, 2291));
    }

    [Fact]
    public void AnAssignedSeed_ReachesACallerBuiltFlorence2Projector()
        => AssertSeedReachesTheWeights(() => new Florence2ProjectorLayer<double>(
            visionDim: Width, projectionDim: Width, grid: 2));

    [Fact]
    public void AnAssignedSeed_ReachesACallerBuiltClipVisionTransformer()
        => AssertSeedReachesTheWeights(() => new ClipVisionTransformerLayer<double>(
            hidden: Width, numLayers: 1, numHeads: 2, patchSize: 4, imageSize: 8));

    [Fact]
    public void AnAssignedSeed_ReachesACallerBuiltKosmosImageResampler()
        => AssertSeedReachesTheWeights(() => new KosmosImageResamplerLayer<double>(
            visionDim: Width, dim: Width, numLatents: 2, numHeads: 2, depth: 1, perceiver: true));

    [Fact]
    public void AnAssignedSeed_ReachesACallerBuiltT5FeedForward()
        => AssertSeedReachesTheWeights(() => new T5FeedForwardLayer<double>(Width, Width * 2));

    [Fact]
    public void AnAssignedSeed_ReachesACallerBuiltUdopTransformer()
        => AssertSeedReachesTheWeights(() => new UdopTransformerLayer<double>(
            vocabSize: 16, dim: Width, numHeads: 2, keyValueDim: 4, ffDim: Width * 2,
            numEncoderLayers: 1, numDecoderLayers: 1, imageSize: 8, patchSize: 4,
            numBuckets: 8, maxDistance: 16, maxDistance2D: 16, max2DPositions: 8));

    private static Florence2ProjectorLayer<double> Projector()
        => new Florence2ProjectorLayer<double>(visionDim: Width, projectionDim: Width, grid: 2);

    [Fact]
    public void ASeedAssignedAfterAnEdit_KeepsTheEditedWeights()
    {
        // The redraw only replaces weights construction drew: once a caller has set them, a seed must not undo that.
        var layer = Projector();
        var edited = layer.GetParameters().ToArray().Select(v => v + 0.5).ToArray();
        layer.SetParameters(new Vector<double>(edited));

        layer.RandomSeed = 2290;

        Assert.Equal(edited, layer.GetParameters().ToArray());
    }

    [Fact]
    public void ASecondSeed_RedrawsWeightsTheFirstSeedDrew()
    {
        // The first redraw refreshes the construction baseline, so a second seed sees untouched weights and redraws
        // them; a stale baseline would refuse and leave the first seed's weights in place.
        var layer = Projector();
        layer.RandomSeed = 2290;
        layer.RandomSeed = 2291;

        Assert.Equal(DrawnAtAssignedSeed(Projector, 2291), layer.GetParameters().ToArray());
        Assert.NotEqual(DrawnAtAssignedSeed(Projector, 2290), layer.GetParameters().ToArray());
    }

    public static TheoryData<string> ComposedLayers => new() { "Florence2Projector", "ClipVisionTransformer", "KosmosImageResampler", "UdopTransformer" };

    private static LayerBase<double> Composed(string kind) => kind switch
    {
        "Florence2Projector" => Projector(),
        "ClipVisionTransformer" => new ClipVisionTransformerLayer<double>(
            hidden: Width, numLayers: 1, numHeads: 2, patchSize: 4, imageSize: 8),
        "KosmosImageResampler" => new KosmosImageResamplerLayer<double>(
            visionDim: Width, dim: Width, numLatents: 2, numHeads: 2, depth: 1, perceiver: true),
        "UdopTransformer" => new UdopTransformerLayer<double>(
            vocabSize: 16, dim: Width, numHeads: 2, keyValueDim: 4, ffDim: Width * 2,
            numEncoderLayers: 1, numDecoderLayers: 1, imageSize: 8, patchSize: 4,
            numBuckets: 8, maxDistance: 16, maxDistance2D: 16, max2DPositions: 8),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    private static int?[] ChildSeeds(LayerBase<double> layer)
        => layer.GetSubLayers().OfType<LayerBase<double>>().Select(child => child.RandomSeed).ToArray();

    [Theory]
    [MemberData(nameof(ComposedLayers))]
    public void AnAssignedSeed_ReachesTheChildren_WhenTheParentKeepsEditedWeights(string kind)
    {
        // An edit makes the parent refuse to redraw its own tables, which used to return before the children were
        // seeded: they kept their construction seeds. They must get exactly the seeds an unedited layer gives them.
        var edited = Composed(kind);
        var shifted = edited.GetParameters().ToArray().Select(v => v + 0.5).ToArray();
        edited.SetParameters(new Vector<double>(shifted));
        edited.RandomSeed = 2290;

        var fresh = Composed(kind);
        fresh.RandomSeed = 2290;

        var expected = ChildSeeds(fresh);
        Assert.NotEmpty(expected);
        Assert.All(expected, seed => Assert.NotNull(seed));
        Assert.Equal(expected, ChildSeeds(edited));
        Assert.Equal(shifted, edited.GetParameters().ToArray());
    }

    [Theory]
    [MemberData(nameof(ComposedLayers))]
    public void AChildSeedTheCallerChose_SurvivesTheParentsSeed_AndTheOthersKeepTheirs(string kind)
    {
        // A seed set on a child is the caller's choice and outranks the one its parent hands down. The parent still
        // draws a seed for that child, so every other child gets exactly what it would have got.
        var reference = Composed(kind);
        reference.RandomSeed = 2290;
        var expected = ChildSeeds(reference);
        Assert.True(expected.Length > 1, $"{kind} needs two children for this check.");

        var layer = Composed(kind);
        var chosen = layer.GetSubLayers().OfType<LayerBase<double>>().First();
        chosen.RandomSeed = 7;
        layer.RandomSeed = 2290;

        var actual = ChildSeeds(layer);
        Assert.Equal(7, actual[0]);
        Assert.Equal(expected.Skip(1), actual.Skip(1));
    }

    [Theory]
    [MemberData(nameof(ComposedLayers))]
    public void AParentsSecondSeed_ReplacesTheSeedsItsFirstHandedDown(string kind)
    {
        // An inherited seed is the parent's, not a choice of the caller's, so the parent's next seed replaces it.
        var layer = Composed(kind);
        layer.RandomSeed = 2290;
        layer.RandomSeed = 2291;

        var fresh = Composed(kind);
        fresh.RandomSeed = 2291;
        Assert.Equal(ChildSeeds(fresh), ChildSeeds(layer));
    }

    [Fact]
    public void EqualModelSeeds_GiveEqualInitialWeights_ForACallerBuiltSsmStack()
    {
        var first = InitialWeights(2290);

        Assert.NotEmpty(first);
        Assert.Equal(first, InitialWeights(2290));
    }

    [Fact]
    public void DifferentModelSeeds_GiveDifferentInitialWeights_ForACallerBuiltSsmStack()
    {
        Assert.NotEqual(InitialWeights(2290), InitialWeights(2291));
    }
}
