using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.PointCloud.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Proves the construction-state categories that used to be pinned to constructor defaults.
/// </summary>
public sealed class LayerConstructionStateCloneTests
{
    [Fact]
    public void Composite_clone_preserves_every_activation_slot_independently()
    {
        var original = new ReconstructionLayer<double>(
            inputDimension: 4,
            hidden1Dimension: 3,
            hidden2Dimension: 2,
            outputDimension: 4,
            hiddenActivation: new TanhActivation<double>(),
            outputActivation: new IdentityActivation<double>());

        var clone = (ReconstructionLayer<double>)original.Clone(CloneOptions.Architecture);
        var originalChildren = original.GetSubLayers().Cast<LayerBase<double>>().ToArray();
        var cloneChildren = clone.GetSubLayers().Cast<LayerBase<double>>().ToArray();

        Assert.IsType<TanhActivation<double>>(cloneChildren[0].ScalarActivation);
        Assert.IsType<TanhActivation<double>>(cloneChildren[1].ScalarActivation);
        Assert.IsType<IdentityActivation<double>>(cloneChildren[2].ScalarActivation);
        Assert.NotSame(originalChildren[0].ScalarActivation, cloneChildren[0].ScalarActivation);
        Assert.NotSame(originalChildren[2].ScalarActivation, cloneChildren[2].ScalarActivation);
    }

    [Fact]
    public void Json_configuration_is_equal_but_not_aliased()
    {
        var original = new FlashAttentionLayer<double>(
            sequenceLength: 4,
            embeddingDimension: 8,
            headCount: 2,
            config: new FlashAttentionConfig
            {
                BlockSizeQ = 3,
                BlockSizeKV = 2,
                UseCausalMask = true,
                DropoutProbability = 0.125f,
                Precision = FlashAttentionPrecision.Mixed,
            });

        var clone = (FlashAttentionLayer<double>)original.Clone(CloneOptions.Architecture);

        Assert.NotSame(original.Config, clone.Config);
        Assert.Equal(3, clone.Config.BlockSizeQ);
        Assert.Equal(2, clone.Config.BlockSizeKV);
        Assert.True(clone.Config.UseCausalMask);
        Assert.Equal(0.125f, clone.Config.DropoutProbability);
        Assert.Equal(FlashAttentionPrecision.Mixed, clone.Config.Precision);

        clone.Config.BlockSizeQ = 7;
        Assert.Equal(3, original.Config.BlockSizeQ);
    }

    [Fact]
    public void Enum_arrays_and_resolved_nullable_arrays_round_trip_non_defaults()
    {
        var pna = new PrincipalNeighbourhoodAggregationLayer<double>(
            inputFeatures: 3,
            outputFeatures: 2,
            aggregators: new[] { PNAAggregator.Max, PNAAggregator.StdDev },
            scalers: new[] { PNAScaler.Attenuation });
        var pnaClone = (PrincipalNeighbourhoodAggregationLayer<double>)pna.Clone(CloneOptions.Architecture);

        Assert.Equal(
            new[] { PNAAggregator.Max, PNAAggregator.StdDev },
            Field<PNAAggregator[]>(pnaClone, "_aggregators"));
        Assert.Equal(
            new[] { PNAScaler.Attenuation },
            Field<PNAScaler[]>(pnaClone, "_scalers"));

        var tnet = new TNetLayer<double>(
            transformDim: 2,
            numFeatures: 3,
            mlpChannels: new[] { 5, 7 },
            fcChannels: new[] { 11 });
        var tnetClone = (TNetLayer<double>)tnet.Clone(CloneOptions.Architecture);

        Assert.Equal(new[] { 5, 7 }, Field<int[]>(tnetClone, "_mlpChannels"));
        Assert.Equal(new[] { 11 }, Field<int[]>(tnetClone, "_fcChannels"));
        Assert.NotSame(Field<int[]>(tnet, "_mlpChannels"), Field<int[]>(tnetClone, "_mlpChannels"));
    }

    [Fact]
    public void Non_default_derived_topology_settings_survive_architecture_clone()
    {
        var denseBlock = new DenseBlock<double>(numLayers: 1, growthRate: 2, bnMomentum: 0.37);
        var denseClone = (DenseBlock<double>)denseBlock.Clone(CloneOptions.Architecture);
        Assert.Equal(0.37, Field<double>(denseClone, "_bnMomentum"), precision: 12);

        var cls = new PrependCLSTokenLayer<double>(embedDim: 4, initScale: 0.17, seed: 3);
        var clsClone = (PrependCLSTokenLayer<double>)cls.Clone(CloneOptions.Architecture);
        Assert.Equal(0.17, Field<double>(clsClone, "_initScale"), precision: 12);

        var rrdb = new RRDBNetGenerator<double>(
            inputChannels: 1,
            outputChannels: 2,
            numFeatures: 4,
            growthChannels: 3,
            numRRDBBlocks: 1,
            scale: 2,
            residualScale: 0.31);
        var rrdbClone = (RRDBNetGenerator<double>)rrdb.Clone(CloneOptions.Architecture);
        Assert.Equal(3, Field<int>(rrdbClone, "_growthChannels"));
        Assert.Equal(0.31, Field<double>(rrdbClone, "_residualScale"), precision: 12);

        var vgg = new VGGishAudioEmbedding<double>(
            conv1Filters: 2,
            conv2Filters: 3,
            conv3Filters: 4,
            conv4Filters: 5,
            fullyConnectedWidth: 7,
            embeddingSize: 6);
        var vggClone = (VGGishAudioEmbedding<double>)vgg.Clone(CloneOptions.Architecture);
        Assert.Equal(2, Field<int>(vggClone, "_conv1Filters"));
        Assert.Equal(3, Field<int>(vggClone, "_conv2Filters"));
        Assert.Equal(4, Field<int>(vggClone, "_conv3Filters"));
        Assert.Equal(5, Field<int>(vggClone, "_conv4Filters"));
        Assert.Equal(7, vggClone.FullyConnectedWidth);
        Assert.Equal(6, vggClone.EmbeddingSize);
    }

    [Fact]
    public void Live_tensor_construction_state_is_cloned_without_changing_ownership()
    {
        var sharedBias = new Tensor<double>(new[] { 4, 2 });
        for (int i = 0; i < sharedBias.Length; i++) sharedBias[i] = i + 0.5;
        var original = new T5RelativeBiasAttentionLayer<double>(
            hiddenSize: 8,
            numHeads: 2,
            numBuckets: 4,
            sharedRelativeBiasTable: sharedBias);

        var clone = (T5RelativeBiasAttentionLayer<double>)original.Clone(CloneOptions.Full);

        Assert.False(original.OwnsRelativeBiasTable);
        Assert.False(clone.OwnsRelativeBiasTable);
        Assert.NotSame(original.GetRelativeBiasTable(), clone.GetRelativeBiasTable());
        Assert.Equal(original.GetRelativeBiasTable().ToArray(), clone.GetRelativeBiasTable().ToArray());

        clone.GetRelativeBiasTable()[0] = 99;
        Assert.Equal(0.5, original.GetRelativeBiasTable()[0], precision: 12);
    }

    [Fact]
    public void Live_child_lists_clone_elements_and_persistent_state_without_aliasing()
    {
        var sharedFc = new FullyConnectedLayer<double>(
            4, (IActivationFunction<double>)new IdentityActivation<double>());
        var sharedBn = new GhostBatchNormalization<double>(4, virtualBatchSize: 2, momentum: 0.13);
        var original = new FeatureTransformerLayer<double>(
            inputDim: 4,
            outputDim: 2,
            sharedLayers: new List<FullyConnectedLayer<double>> { sharedFc },
            sharedBNLayers: new List<GhostBatchNormalization<double>> { sharedBn },
            numSharedLayers: 1,
            numStepSpecificLayers: 0,
            virtualBatchSize: 2,
            momentum: 0.13);

        var input = new Tensor<double>(new[] { 2, 4 });
        input.Fill(1.0);
        _ = original.Forward(input);

        var clone = (FeatureTransformerLayer<double>)original.Clone(CloneOptions.Full);
        var originalFc = Field<List<FullyConnectedLayer<double>>>(original, "_sharedFCLayers");
        var cloneFc = Field<List<FullyConnectedLayer<double>>>(clone, "_sharedFCLayers");
        var originalBn = Field<List<GhostBatchNormalization<double>>>(original, "_sharedBNLayers");
        var cloneBn = Field<List<GhostBatchNormalization<double>>>(clone, "_sharedBNLayers");

        Assert.NotSame(originalFc, cloneFc);
        Assert.NotSame(originalFc[0], cloneFc[0]);
        Assert.NotSame(originalBn, cloneBn);
        Assert.NotSame(originalBn[0], cloneBn[0]);

        var cloneParameters = cloneFc[0].GetParameters();
        cloneParameters[0] += 10;
        cloneFc[0].UpdateParameters(cloneParameters);
        Assert.NotEqual(originalFc[0].GetParameters()[0], cloneFc[0].GetParameters()[0]);
    }

    [Fact]
    public void Swin_block_with_rectangular_derived_state_clones_after_forward()
    {
        var original = new SwinTransformerBlockLayer<double>(
            dim: 16,
            numHeads: 2,
            windowSize: 4);
        var input = new Tensor<double>(new[] { 1, 16, 16 });
        input.Fill(0.25);
        _ = original.Forward(input);

        var clone = (SwinTransformerBlockLayer<double>)original.Clone(CloneOptions.Full);

        Assert.Equal(original.ParameterCount, clone.ParameterCount);
        Assert.NotSame(
            Field<int[,]>(original, "_relativePositionIndex"),
            Field<int[,]>(clone, "_relativePositionIndex"));
        Assert.Equal(original.GetParameters().ToArray(), clone.GetParameters().ToArray());
    }

    private static TValue Field<TValue>(object instance, string name)
    {
        for (Type? type = instance.GetType(); type is not null; type = type.BaseType)
        {
            if (type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
                    ?.GetValue(instance) is TValue value)
                return value;
        }

        throw new InvalidOperationException($"{instance.GetType().Name}.{name} was not found.");
    }
}
