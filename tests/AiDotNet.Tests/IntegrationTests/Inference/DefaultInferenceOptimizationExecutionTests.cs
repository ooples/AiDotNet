using System.Linq;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Issue #1632's remaining checklist item: <c>DefaultInferenceOptimizationEngagementTests</c> pins
/// which optimizations the builder RESOLVES, and states in its own summary that it "does not assert
/// that each component executes inside a forward pass". These tests close that gap for the
/// components the default path claims, by asserting what the optimizer actually did to the model
/// rather than what it was configured to do.
/// </summary>
/// <remarks>
/// <para>
/// The observable is the optimized model's own layer list. An attention block that was rewritten
/// for inference is no longer a <see cref="MultiHeadAttentionLayer{T}"/>: it is a
/// <see cref="CachedMultiHeadAttention{T}"/> or a <see cref="PagedCachedMultiHeadAttention{T}"/>,
/// which is what carries the KV cache and the paged kernel. A folded BatchNorm is gone from the
/// list entirely. Both are structural consequences no configuration flag can fake.
/// </para>
/// </remarks>
public sealed class DefaultInferenceOptimizationExecutionTests
{
    /// <summary>
    /// With the default config, an attention model comes back rewritten onto the cached attention
    /// that holds the KV cache — so the KV-cache and paged-attention components are reached, not
    /// merely switched on.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task TheDefaultConfigRewritesAttentionOntoTheCachedPath()
    {
        await Task.Yield();

        var model = BuildAttentionModel();
        Assert.Contains(model.Layers, layer => layer is MultiHeadAttentionLayer<double>);

        var optimizer = new InferenceOptimizer<double>(InferenceOptimizationConfig.Default);
        var (optimized, applied) = optimizer.OptimizeForInference(model);

        Assert.True(applied,
            "The default config reported no optimization applied to a model with an attention "
                + "layer, so nothing in the inference stack engaged.");
        Assert.Contains(optimized.Layers, layer =>
            layer is CachedMultiHeadAttention<double> or PagedCachedMultiHeadAttention<double>);
    }

    /// <summary>
    /// Control arm: with every flag off the same model must come back untouched, otherwise the
    /// assertion above would pass on an optimizer that rewrites regardless of configuration.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task OptingOutLeavesTheAttentionLayerAlone()
    {
        await Task.Yield();

        var model = BuildAttentionModel();
        var optimizer = new InferenceOptimizer<double>(new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnablePagedKVCache = false,
            EnableFlashAttention = false,
            EnableLayerFusion = false,
            EnableBatching = false,
            EnableWeightOnlyQuantization = false,
        });

        var layersBefore = model.Layers.ToArray();

        var (optimized, applied) = optimizer.OptimizeForInference(model);

        Assert.False(applied, "An opted-out config still reported optimizations as applied.");

        // With every rewrite flag off the optimizer must hand back the model it was given, not a
        // clone: cloning costs a full parameter copy for nothing. Asserting the instance and the
        // layer sequence also catches a layer added or removed, which the type checks below cannot.
        Assert.Same(model, optimized);
        Assert.Equal(layersBefore, optimized.Layers.ToArray());
        Assert.Contains(optimized.Layers, layer => layer is MultiHeadAttentionLayer<double>);
        Assert.DoesNotContain(optimized.Layers, layer =>
            layer is CachedMultiHeadAttention<double> or PagedCachedMultiHeadAttention<double>);
    }

    /// <summary>
    /// The optimizer must not mutate the model it was handed — the default is on for every model
    /// now, so a rewrite in place would change training behaviour for everyone.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task TheOriginalModelIsLeftUntouched()
    {
        await Task.Yield();

        var model = BuildAttentionModel();
        var optimizer = new InferenceOptimizer<double>(InferenceOptimizationConfig.Default);
        optimizer.OptimizeForInference(model);

        Assert.Contains(model.Layers, layer => layer is MultiHeadAttentionLayer<double>);
        Assert.DoesNotContain(model.Layers, layer =>
            layer is CachedMultiHeadAttention<double> or PagedCachedMultiHeadAttention<double>);
    }

    private static NeuralNetwork<double> BuildAttentionModel()
    {
        const int embedding = 16;
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputSize: embedding,
            outputSize: embedding,
            layers:
            [
                new MultiHeadAttentionLayer<double>(
                    headCount: 4,
                    headDimension: embedding / 4,
                    activationFunction: (AiDotNet.Interfaces.IActivationFunction<double>)
                        new AiDotNet.ActivationFunctions.IdentityActivation<double>()),
                new DenseLayer<double>(
                    outputSize: embedding,
                    activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<double>()),
            ]);

        return new NeuralNetwork<double>(architecture);
    }
}
