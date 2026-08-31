using System;
using System.Linq;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Correctness contract for the copy-on-write <see cref="NeuralNetworkBase{T}.Clone"/> lever (#1624):
/// a clone must be <b>observationally identical</b> to its source yet <b>independent under mutation</b>.
/// The COW fast path shares weight storage via <c>Tensor&lt;T&gt;.CloneShared()</c> (O(1)-until-write),
/// so the load-bearing guarantee is that the first in-place write to either model privatizes that tensor
/// — never silently corrupting the other. (If the share never engages, the eager fallback satisfies the
/// same contract, so this test pins the behavior regardless of which path runs.)
/// </summary>
public class CopyOnWriteCloneTests
{
    private static FeedForwardNeuralNetwork<double> BuildModel()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        return new FeedForwardNeuralNetwork<double>(arch);
    }

    private static NeuralNetwork<double> BuildBufferedModel()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 4,
            layers: new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<double>>
            {
                new BatchNormalizationLayer<double>(numFeatures: 4)
            });
        return new NeuralNetwork<double>(architecture);
    }

    private static Tensor<double> Input() =>
        new(new Vector<double>(new[] { 0.1, -0.2, 0.3, -0.4 }), new[] { 1, 4 });

    /// <summary>A cloned batch-norm model must predict exactly what its original predicts.</summary>
    /// <remarks>
    /// SupportsCopyOnWriteLayerGraph refuses copy-on-write for any model containing a
    /// BatchNormalizationLayer, on the grounds that CpuEngine's TensorDivide reads its operands
    /// through the mutable DataVector surface, which copy-on-write tensors reject. That guard's own
    /// comment says to remove it once TensorDivide reads through a read-only span -- but nothing
    /// verified either the hazard or its absence, so the guard could neither be trusted nor retired.
    ///
    /// This is that verification, and it is written to fail loudly if copy-on-write ever corrupts a
    /// batch-norm clone: identical inputs must give identical outputs, and mutating one model must
    /// not move the other. It is deliberately independent of whether the guard is currently on --
    /// with the guard it exercises the eager path, without it the shared path.
    /// </remarks>
    [Fact]
    public void BatchNormModel_ClonePredictsIdentically_AndStaysIndependent()
    {
        using var source = BuildBufferedModel();
        var input = Input();

        var beforeClone = source.Predict(input);

        var clone = source.DeepCopy() as NeuralNetworkBase<double>;
        Assert.NotNull(clone);

        var original = source.Predict(input);
        var copied = clone!.Predict(input);

        Assert.Equal(original.Length, copied.Length);
        for (var i = 0; i < original.Length; i++)
        {
            Assert.True(
                Math.Abs(original[i] - copied[i]) < 1e-12,
                $"clone diverged at {i}: {original[i]} vs {copied[i]}");
        }

        // The source must also be unchanged by having been cloned.
        for (var i = 0; i < beforeClone.Length; i++)
        {
            Assert.True(
                Math.Abs(beforeClone[i] - original[i]) < 1e-12,
                $"cloning moved the source at {i}: {beforeClone[i]} vs {original[i]}");
        }

        // Writing through the clone must not reach the source -- the aliasing hazard the guard
        // exists to prevent would show up exactly here.
        var mutated = clone.GetParameters();
        if (mutated.Length > 0)
        {
            for (var i = 0; i < mutated.Length; i++) mutated[i] = mutated[i] + 1.0;
            clone.UpdateParameters(mutated);

            var sourceAfter = source.Predict(input);
            for (var i = 0; i < original.Length; i++)
            {
                Assert.True(
                    Math.Abs(sourceAfter[i] - original[i]) < 1e-12,
                    $"writing through the clone moved the source at {i}");
            }
        }
    }

    [Fact]
    public void MaterializedPersistentBuffers_AreIncludedInCopyOnWriteCoverage()
    {
        using var source = BuildBufferedModel();
        using var destination = BuildBufferedModel();
        var input = Input();
        _ = source.Predict(input);
        _ = destination.ParameterCount;

        var sourceState = source.GetParameters().Clone();
        for (int i = 0; i < sourceState.Length; i++)
            sourceState[i] = 0.125 + i * 0.03125;
        source.SetParameters(sourceState);

        bool shared = CopyOnWriteCloneHelper.TryShareTrainableParameters<double>(
            source,
            destination,
            out CopyOnWriteShareStatus status,
            out string mismatch);

        Assert.True(shared, $"Status={status}; mismatch={mismatch}");
        Assert.Equal(CopyOnWriteShareStatus.Shared, status);
        var destinationState = destination.GetParameters();
        Assert.Equal(sourceState.Length, destinationState.Length);
        for (int i = 0; i < sourceState.Length; i++)
            Assert.Equal(sourceState[i], destinationState[i]);
    }

    [Fact]
    public void RejectedCandidate_WithBorrowedArchitectureLayers_DoesNotDisposeSource()
    {
        using var sharedLayer = new DenseLayer<double>(2);
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2,
            layers: new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<double>>
            {
                sharedLayer
            });
        using var source = new NeuralNetwork<double>(architecture);
        var input = Input();
        var expected = source.Predict(input);

        using var clone = (NeuralNetwork<double>)source.Clone();

        // The first COW candidate is intentionally rejected because CreateNewInstance receives
        // the architecture's same layer objects. Cleaning up that rejected candidate must observe
        // the ownership boundary and leave the source layer executable.
        var sourceAfterClone = source.Predict(input);
        Assert.Equal(expected.Length, sourceAfterClone.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], sourceAfterClone[i], 10);

        var clonePrediction = clone.Predict(input);
        Assert.Equal(expected.Length, clonePrediction.Length);
    }

    [Fact]
    public void Clone_IsObservationallyIdentical_AndIndependentUnderMutation()
    {
        var model = BuildModel();
        var input = Input();
        _ = model.Predict(input); // materialize lazy weights so Clone sees the full parameter set

        var sourceParamsBefore = model.GetParameters().Clone();
        var sourcePredBefore = model.Predict(input);

        var clone = (FeedForwardNeuralNetwork<double>)model.Clone();

        // 1) The clone is observationally identical to the source right after cloning.
        var clonePred = clone.Predict(input);
        Assert.Equal(sourcePredBefore.Length, clonePred.Length);
        for (int i = 0; i < sourcePredBefore.Length; i++)
            Assert.Equal(sourcePredBefore[i], clonePred[i], 10);

        // 2) Mutate the clone's weights. Copy-on-write must privatize them so the SOURCE is untouched.
        var mutated = clone.GetParameters().Clone();
        for (int i = 0; i < mutated.Length; i++) mutated[i] += 0.5;
        clone.SetParameters(mutated);

        var sourceParamsAfter = model.GetParameters();
        double maxDrift = 0;
        for (int i = 0; i < sourceParamsBefore.Length; i++)
            maxDrift = Math.Max(maxDrift, Math.Abs(sourceParamsAfter[i] - sourceParamsBefore[i]));
        Assert.True(maxDrift < 1e-12,
            $"Mutating the clone changed the source by {maxDrift:E3} — copy-on-write failed to privatize " +
            "the shared weight storage (the clone and source are aliasing the same tensors).");

        // 3) The source's prediction is unchanged; the clone's actually changed (mutation took effect).
        var sourcePredAfter = model.Predict(input);
        for (int i = 0; i < sourcePredBefore.Length; i++)
            Assert.Equal(sourcePredBefore[i], sourcePredAfter[i], 10);

        var clonePredAfter = clone.Predict(input);
        double cloneDelta = 0;
        for (int i = 0; i < clonePredAfter.Length; i++)
            cloneDelta = Math.Max(cloneDelta, Math.Abs(clonePredAfter[i] - sourcePredBefore[i]));
        Assert.True(cloneDelta > 1e-6, "Mutating the clone's parameters did not change its prediction.");
    }

    [Fact]
    public void NestedTransformerGraph_IsIdenticalAndIndependentAfterCopyOnWriteShare()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 16);
        var options = new XTTSv2CloneOptions
        {
            VocabSize = 32,
            TextEncoderDim = 8,
            LLMDim = 8,
            NumEncoderLayers = 0,
            NumLLMLayers = 1,
            NumHeads = 2,
            NumCodebooks = 1,
            CodebookSize = 16,
            SpeakerEmbeddingDim = 8,
            DropoutRate = 0,
        };
        using var source = new XTTSv2Clone<float>(architecture, options);
        using var destination = new XTTSv2Clone<float>(architecture, new XTTSv2CloneOptions(options));
        var input = new Tensor<float>(
            new Vector<float>(new[] { 1f, 2f, 3f, 4f }),
            new[] { 4 });

        _ = source.Predict(input);
        _ = destination.Predict(input);
        var sourceParamsBefore = source.GetParameters().Clone();

        Assert.True(
            CopyOnWriteCloneHelper.TryShareTrainableParameters<float>(source, destination),
            "The registered XTTS transformer module graph was not eligible for copy-on-write sharing.");

        var sourcePrediction = source.Predict(input);
        var sharedPrediction = destination.Predict(input);
        Assert.Equal(sourcePrediction.Length, sharedPrediction.Length);
        for (int i = 0; i < sourcePrediction.Length; i++)
            Assert.Equal(sourcePrediction[i], sharedPrediction[i], 5);

        var mutated = destination.GetParameters().Clone();
        for (int i = 0; i < mutated.Length; i++)
            mutated[i] += 0.25f;
        destination.SetParameters(mutated);

        var sourceParamsAfter = source.GetParameters();
        float maxSourceDrift = 0;
        for (int i = 0; i < sourceParamsBefore.Length; i++)
            maxSourceDrift = Math.Max(maxSourceDrift, Math.Abs(sourceParamsAfter[i] - sourceParamsBefore[i]));
        Assert.True(maxSourceDrift < 1e-6f,
            $"Mutating a nested-transformer destination changed the source by {maxSourceDrift:E3}.");

        var mutatedPrediction = destination.Predict(input);
        float destinationDelta = 0;
        for (int i = 0; i < mutatedPrediction.Length; i++)
            destinationDelta = Math.Max(
                destinationDelta,
                Math.Abs(mutatedPrediction[i] - sourcePrediction[i]));
        Assert.True(destinationDelta > 1e-5f,
            "Mutating the copy-on-write transformer graph did not change the destination prediction.");
    }

    [Fact]
    public async Task FreshDense_PreservesReboundWeightsOnFirstForward()
    {
        await Task.Yield();

        using var source = new DenseLayer<float>(outputSize: 3);
        using var destination = new DenseLayer<float>(outputSize: 3);
        var input = new Tensor<float>(
            new Vector<float>(new[] { 0.25f, -0.5f, 0.75f, 1.25f }),
            new[] { 1, 4 });

        var expected = source.Forward(input);
        var sourceParameters = source.GetTrainableParameters();
        var shared = new Tensor<float>[sourceParameters.Count];
        for (int i = 0; i < sourceParameters.Count; i++)
            shared[i] = (Tensor<float>)sourceParameters[i].CloneShared();

        // The destination has resolved no input shape yet. Rebinding trained
        // tensors must materialize it without reinitializing those tensors.
        destination.SetTrainableParameters(shared);
        var actual = destination.Forward(input);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 6);
    }

    [Fact]
    public async Task MaterializedDiT_CanShareIntoShapeResolvedLazyDestination()
    {
        await Task.Yield();

        using var source = new DiTNoisePredictor<float>(
            inputChannels: 2,
            hiddenSize: 8,
            numLayers: 1,
            numHeads: 2,
            patchSize: 1,
            contextDim: 8,
            latentSpatialSize: 2,
            seed: 17);
        using var destination = new DiTNoisePredictor<float>(
            inputChannels: 2,
            hiddenSize: 8,
            numLayers: 1,
            numHeads: 2,
            patchSize: 1,
            contextDim: 8,
            latentSpatialSize: 2,
            seed: 91);
        using var input = new Tensor<float>(new[] { 1, 2, 2, 2 });

        var expected = source.PredictNoise(input, timestep: 3, conditioning: null);
        long destinationCount = destination.ParameterCount; // resolves structure, not lazy weight storage
        var countDiagnostics = string.Join("; ",
            CopyOnWriteCloneHelper.CollectTrainableLayers<float>(destination)
                .OfType<LayerBase<float>>()
                .Select(layer =>
                    $"{layer.GetType().Name}: count={layer.ParameterCount}, " +
                    $"tensors=[{string.Join(",", layer.GetTrainableParametersWithoutMaterialization().Select(t => t.Length))}], " +
                    $"layout=[{string.Join(",", layer.GetParameterLayout().Select(slot => $"{slot.Readiness}:{slot.ParameterCount}"))}]"));
        Assert.True(destinationCount > 0, countDiagnostics);

        bool sharedParameters = CopyOnWriteCloneHelper.TryShareTrainableParameters<float>(
            source,
            destination,
            out CopyOnWriteShareStatus shareStatus,
            out string shareMismatch);
        var sourceLayers = CopyOnWriteCloneHelper.CollectTrainableLayers<float>(source);
        var destinationLayers = CopyOnWriteCloneHelper.CollectTrainableLayers<float>(destination);
        var shareDiagnostics = string.Join("; ", sourceLayers.Zip(destinationLayers, (sourceLayer, destinationLayer) =>
        {
            var sourceValues = ((LayerBase<float>)sourceLayer).GetOwnTrainableParameterValueTensors();
            var destinationBase = (LayerBase<float>)destinationLayer;
            return $"{sourceLayer.GetType().Name}: source=[{string.Join(",", sourceValues.Select(t => string.Join("x", t.Shape.ToArray())))}], " +
                   $"destination=[{string.Join(",", destinationBase.GetTrainableParametersWithoutMaterialization().Select(t => string.Join("x", t.Shape.ToArray())))}], " +
                   $"canAdopt={destinationBase.CanAdoptTrainableParametersWithoutMaterialization(sourceValues)}";
        }));
        Assert.True(
            sharedParameters,
            $"Generated shape declarations should authorize COW binding into lazy DiT placeholders. " +
            $"Status={shareStatus}; mismatch={shareMismatch}. {shareDiagnostics}");

        for (int layerIndex = 0; layerIndex < sourceLayers.Count; layerIndex++)
        {
            var sourceValues = ((LayerBase<float>)sourceLayers[layerIndex]).GetOwnTrainableParameterValueTensors();
            var destinationValues = ((LayerBase<float>)destinationLayers[layerIndex]).GetOwnTrainableParameterValueTensors();
            Assert.Equal(sourceValues.Count, destinationValues.Count);
            for (int parameterIndex = 0; parameterIndex < sourceValues.Count; parameterIndex++)
            {
                Assert.Equal(sourceValues[parameterIndex].Shape.ToArray(), destinationValues[parameterIndex].Shape.ToArray());
                Assert.Equal(sourceValues[parameterIndex].AsSpan().ToArray(), destinationValues[parameterIndex].AsSpan().ToArray());
            }
        }

        var sourceAfterShare = source.PredictNoise(input, timestep: 3, conditioning: null);
        Assert.Equal(expected.Length, sourceAfterShare.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], sourceAfterShare[i], 5);

        var actual = destination.PredictNoise(input, timestep: 3, conditioning: null);
        for (int layerIndex = 0; layerIndex < sourceLayers.Count; layerIndex++)
        {
            var sourceValues = ((LayerBase<float>)sourceLayers[layerIndex]).GetOwnTrainableParameterValueTensors();
            var destinationValues = ((LayerBase<float>)destinationLayers[layerIndex]).GetOwnTrainableParameterValueTensors();
            for (int parameterIndex = 0; parameterIndex < sourceValues.Count; parameterIndex++)
            {
                Assert.True(
                    sourceValues[parameterIndex].AsSpan().SequenceEqual(destinationValues[parameterIndex].AsSpan()),
                    $"{sourceLayers[layerIndex].GetType().Name} parameter {parameterIndex} changed during destination first-forward initialization.");
            }
        }
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public async Task FreshDense_RejectsIncompatibleReboundWeightsOnFirstForward()
    {
        await Task.Yield();

        using var destination = new DenseLayer<float>(outputSize: 3);
        using var weights = new Tensor<float>(new[] { 5, 3 });
        using var biases = new Tensor<float>(new[] { 3 });
        using var input = new Tensor<float>(new[] { 1, 4 });

        destination.SetTrainableParameters(new[] { weights, biases });
        var ex = Assert.Throws<InvalidOperationException>(() => destination.Forward(input));

        Assert.Contains("Expected weights [4, 3]", ex.Message);
        Assert.Contains("received weights [5, 3]", ex.Message);
    }

    [Fact]
    public void FreshEmbedding_PreservesReboundWeightsOnFirstForward()
    {
        using var source = new EmbeddingLayer<float>(32, 8);
        using var destination = new EmbeddingLayer<float>(32, 8);
        var input = new Tensor<float>(
            new Vector<float>(new[] { 1f, 7f, 11f, 19f }),
            new[] { 4 });

        var expected = source.Forward(input);
        var sourceParameters = source.GetTrainableParameters();
        var shared = new Tensor<float>[sourceParameters.Count];
        for (int i = 0; i < sourceParameters.Count; i++)
            shared[i] = (Tensor<float>)sourceParameters[i].CloneShared();

        // The destination is intentionally still fresh. This is the state a
        // graph-safe DeepCopy destination is in when its trained tensors are rebound.
        destination.SetTrainableParameters(shared);
        var actual = destination.Forward(input);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 6);
    }

    [Fact]
    public void DeferredGroupedQueryAttention_PreservesReboundWeightsOnFirstForward()
    {
        using var source = new GroupedQueryAttentionLayer<float>(
            sequenceLength: 4,
            embeddingDimension: 8,
            numHeads: 2,
            numKVHeads: 1,
            deferAllocation: true);
        using var destination = new GroupedQueryAttentionLayer<float>(
            sequenceLength: 4,
            embeddingDimension: 8,
            numHeads: 2,
            numKVHeads: 1,
            deferAllocation: true);
        var values = new float[32];
        for (int i = 0; i < values.Length; i++)
            values[i] = (i - 16) / 32f;
        var input = new Tensor<float>(new Vector<float>(values), new[] { 4, 8 });

        var expected = source.Forward(input);
        var sourceParameters = source.GetTrainableParameters();
        var shared = new Tensor<float>[sourceParameters.Count];
        for (int i = 0; i < sourceParameters.Count; i++)
            shared[i] = (Tensor<float>)sourceParameters[i].CloneShared();

        // Keep the destination deferred until after rebinding, which is the
        // exact state produced by a graph-safe clone of a decoder stack.
        destination.SetTrainableParameters(shared);
        var actual = destination.Forward(input);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 5);
    }
}
