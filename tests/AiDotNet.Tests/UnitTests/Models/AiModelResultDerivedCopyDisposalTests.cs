using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Preprocessing;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// The model copies an <see cref="AiModelResult{T,TInput,TOutput}"/> makes for itself are released with it.
///
/// <para>With inference optimizations configured, the first <c>Predict</c> clones the model and rewrites the
/// clone for inference, and every inference sequence clones it again for its own KV cache. Neither copy was
/// ever disposed: <c>AiModelResult.Dispose</c> released only the wrapped <c>Model</c>, and
/// <c>InferenceSequence.Dispose</c> only cleared its cache, so each copy's layers -- pooled weight buffers,
/// compiled plans -- lived until the garbage collector ran.</para>
/// </summary>
[Collection(AiDotNet.Tests.TestInfrastructure.DiagnosticsEnvironmentCollection.Name)]
public partial class AiModelResultDerivedCopyDisposalTests
{
    private const int SequenceLength = 1;
    private const int EmbeddingDimension = 8;
    private const int HeadCount = 2;
    private const int FlatSize = SequenceLength * EmbeddingDimension;

    [Fact]
    public void Dispose_releases_the_inference_optimized_copy_and_the_model()
    {
        var model = CreateAttentionModel();
        var result = CreateResult(model, new InferenceOptimizationConfig
        {
            EnableFlashAttention = true,
            EnableKVCache = false,
            EnablePagedKVCache = false,
        });

        _ = result.Predict(Token());
        // Premise: Predict really did build a separate optimized copy.
        var copy = Assert.IsAssignableFrom<NeuralNetworkBase<float>>(
            ReadField(result, "_inferenceOptimizedNeuralModel"));
        Assert.NotSame(model, copy);

        result.Dispose();

        AssertAllLayersReleased(copy, "the inference-optimized copy");
        AssertAllLayersReleased(model, "the wrapped model");
    }

    [Fact]
    public void Disposing_a_sequence_releases_the_copy_it_made_but_not_the_model()
    {
        var model = CreateAttentionModel();
        var result = CreateResult(model, new InferenceOptimizationConfig
        {
            EnableFlashAttention = false,
            EnableKVCache = true,
            EnablePagedKVCache = false,
            AttentionMasking = AttentionMaskingMode.Auto,
        });

        // Baseline taken before the session exists, so the comparison isolates the effect of
        // releasing the sequence's copy rather than anything session setup did.
        var baseline = result.Predict(Token());

        var session = result.BeginInferenceSession();
        var sequence = session.CreateSequence();
        _ = sequence.Predict(Token());
        var copy = Assert.IsAssignableFrom<NeuralNetworkBase<float>>(
            ReadField(sequence, "_sequenceOptimizedNeuralModel"));
        Assert.NotSame(model, copy);

        sequence.Dispose();
        session.Dispose();

        AssertAllLayersReleased(copy, "the sequence's copy");

        // The sequence only borrowed the model; the result still owns it. It is not enough that
        // Predict returns SOMETHING -- releasing the copy must not have perturbed the borrowed
        // model, so the same input must still produce the same output.
        AssertSamePrediction(baseline, result.Predict(Token()));
        result.Dispose();
    }

    [Fact]
    public void Dispose_releases_deep_ensemble_members_and_the_model_once()
    {
        // The builder trains the extra ensemble members for this result and hands them over; nothing else holds them.
        var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(inputFeatures: 4, outputSize: 2));
        var member = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(inputFeatures: 4, outputSize: 2));
        var result = new AiModelResult<float, Tensor<float>, Tensor<float>> { Model = model };
        result.SetDeepEnsembleModels(new List<IFullModel<float, Tensor<float>, Tensor<float>>> { model, member });

        result.Dispose();

        AssertAllLayersReleased(member, "a deep-ensemble member");
        AssertAllLayersReleased(model, "the wrapped model");
    }

    private static AiModelResult<float, Tensor<float>, Tensor<float>> CreateMetaLearningResult(
        NeuralNetwork<float> baseModel, NeuralNetwork<float> best)
    {
        var metaLearner = new AiDotNet.MetaLearning.Algorithms.MAMLAlgorithm<float, Tensor<float>, Tensor<float>>(
            new AiDotNet.MetaLearning.Options.MAMLOptions<float, Tensor<float>, Tensor<float>>(baseModel)
            {
                LossFunction = new AiDotNet.LossFunctions.MeanSquaredErrorLoss<float>(),
            });
        return new AiModelResult<float, Tensor<float>, Tensor<float>>(new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            MetaLearner = metaLearner,
            MetaTrainingResult = new MetaTrainingResult<float>(
                new Vector<float>(new[] { 1.0f }), new Vector<float>(new[] { 0.0f }), TimeSpan.Zero),
            OptimizationResult = new OptimizationResult<float, Tensor<float>, Tensor<float>> { BestSolution = best },
        });
    }

    [Fact]
    public void Meta_learning_result_releases_the_optimizers_separate_best_solution()
    {
        // On the meta-learning path Model is the meta-learner's BaseModel, while the optimizer's BestSolution is a
        // different instance that only this result holds -- previously neither Dispose path reached it.
        var baseModel = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(inputFeatures: 4, outputSize: 2));
        var best = new CountingNetwork();
        var result = CreateMetaLearningResult(baseModel, best);

        // Premise: this really is the case where the two differ.
        Assert.NotSame(best, result.Model);

        result.Dispose();
        result.Dispose();

        Assert.Equal(1, best.DisposeCalls);
        AssertAllLayersReleased(best, "the optimizer's best solution");
    }

    [Fact]
    public void A_best_solution_that_is_also_an_ensemble_member_is_released_once()
    {
        var baseModel = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(inputFeatures: 4, outputSize: 2));
        var best = new CountingNetwork();
        var result = CreateMetaLearningResult(baseModel, best);

        // The builder puts the best solution first in a deep ensemble, so both disposal paths can reach it.
        result.SetDeepEnsembleModels(new List<IFullModel<float, Tensor<float>, Tensor<float>>> { best });
        result.Dispose();

        Assert.Equal(1, best.DisposeCalls);
    }

    [Fact]
    public void A_sequence_releases_both_clones_when_multi_lora_task_selection_fails()
    {
        // Two clones are live during sequence setup: the Multi-LoRA copy, and the one the inference
        // optimizer makes. When SelectTask throws, setup falls back to the source model, so the
        // optimizer's clone is the one that ends up unused -- and releasing the two through a single
        // if/else-if chain meant only ever releasing one of them, leaving the Multi-LoRA copy's
        // layers holding their pooled weight buffers until the garbage collector ran.
        var model = CreateCloneTrackingAttentionModel();
        var result = CreateResult(model, new InferenceOptimizationConfig
        {
            EnableFlashAttention = false,
            EnableKVCache = true,
            EnablePagedKVCache = false,
            // Not Auto: the sequence path rewrites Auto to Causal, and causal masking is what lets
            // the KV-cache rewrite apply. Nothing must apply here, so the optimizer's clone is spare.
            AttentionMasking = AttentionMaskingMode.Disabled,
        });


        using var session = result.BeginInferenceSession();
        var sequence = session.CreateSequence("a-task-this-model-never-registered");
        _ = sequence.Predict(Token());

        // Premise: selection failed after its clone was taken, the optimizer cloned again, and
        // neither clone was kept.
        Assert.Equal(2, model.Clones.Count);
        Assert.Null(ReadField(sequence, "_sequenceOptimizedNeuralModel"));

        foreach (var clone in model.Clones)
        {
            AssertAllLayersReleased(clone, "a clone the failed sequence setup left behind");
        }

        sequence.Dispose();
        result.Dispose();
    }


    /// <summary>A model that records every clone taken from it, so a leaked clone can be named.</summary>
    private sealed class CloneTrackingNetwork : NeuralNetwork<float>
    {
        public CloneTrackingNetwork(NeuralNetworkArchitecture<float> architecture) : base(architecture)
        {
        }

        public List<NeuralNetworkBase<float>> Clones { get; } = new();

        public override IFullModel<float, Tensor<float>, Tensor<float>> Clone()
        {
            var clone = base.Clone();
            if (clone is NeuralNetworkBase<float> network)
            {
                Clones.Add(network);
            }

            return clone;
        }

        // The clone plan is generated for library types, so rebuild by hand. The architecture
        // carries explicit layers and models take those BY REFERENCE, so the copy must get its own.
        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new NeuralNetwork<float>(Architecture.CloneForModelConstruction());
    }

    private static CloneTrackingNetwork CreateCloneTrackingAttentionModel()
    {
        // MultiLoRAAdapter needs its base layer's input dimension, and DenseLayer only learns that
        // when a network resolves it. Build the plain model first and wrap its resolved dense layer.
        // Take an owned copy of the dense layer before the temporary network goes away.
        // MultiLoRAAdapter retains its base layer BY REFERENCE, so handing it a layer that the
        // temporary network still owns forces a choice between two leaks: dispose the network and
        // the adapter's base layer dies with it, or skip the dispose and every other layer in the
        // temporary network stays alive until garbage collection.
        var resolved = (NeuralNetworkBase<float>)CreateAttentionModel();
        ILayer<float> dense;
        try
        {
            dense = ((LayerBase<float>)resolved.Layers[resolved.Layers.Count - 1]).Clone();
        }
        finally
        {
            resolved.Dispose();
        }

        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(FlatSize),
            new ReshapeLayer<float>(new[] { SequenceLength, EmbeddingDimension }),
            new MultiHeadAttentionLayer<float>(HeadCount, EmbeddingDimension / HeadCount,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new FlattenLayer<float>(),
            new AiDotNet.LoRA.Adapters.MultiLoRAAdapter<float>(
                dense,
                defaultTaskName: "base",
                defaultRank: 2),
        };

        var model = new CloneTrackingNetwork(new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: FlatSize,
            outputSize: FlatSize,
            layers: layers));

        _ = model.Predict(new Tensor<float>(new[] { 1, FlatSize }));
        return model;
    }

    private sealed partial class CountingNetwork : NeuralNetwork<float>
    {
        public CountingNetwork() : base(new NeuralNetworkArchitecture<float>(inputFeatures: 4, outputSize: 2))
        {
        }

        public int DisposeCalls { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }
    }

    /// <summary>
    /// Asserts two predictions are bit-identical.
    /// </summary>
    /// <remarks>
    /// Exact equality rather than a tolerance is deliberate: this compares the same model against
    /// the same input with no training in between, so the only thing that can move a value is the
    /// disposal under test perturbing the borrowed model. A tolerance would hide exactly the
    /// defect the assertion exists to catch.
    /// </remarks>
    private static void AssertSamePrediction(Tensor<float> baseline, Tensor<float> actual)
    {
        Assert.Equal(baseline.Length, actual.Length);
        for (int i = 0; i < baseline.Length; i++)
        {
            Assert.True(
                baseline[i].Equals(actual[i]),
                $"Releasing the sequence's copy changed the borrowed model's output at index {i}: "
                + $"was {baseline[i]}, now {actual[i]}.");
        }
    }

    private static void AssertAllLayersReleased(NeuralNetworkBase<float> network, string what)
    {
        Assert.NotEmpty(network.Layers);
        foreach (var layer in network.Layers)
        {
            Assert.False(
                DisposeOnceGuard.TryDispose(Assert.IsAssignableFrom<IDisposable>(layer)),
                $"A {layer.GetType().Name} of {what} was still live after disposal.");
        }
    }

    private static object? ReadField(object owner, string name)
        => owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(owner);

    private static AiModelResult<float, Tensor<float>, Tensor<float>> CreateResult(
        NeuralNetworkBase<float> model, InferenceOptimizationConfig config)
        => new(new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = new OptimizationResult<float, Tensor<float>, Tensor<float>> { BestSolution = model },
            PreprocessingInfo = new PreprocessingInfo<float, Tensor<float>, Tensor<float>>(),
            InferenceOptimizationConfig = config,
        });

    private static NeuralNetworkBase<float> CreateAttentionModel()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(FlatSize),
            new ReshapeLayer<float>(new[] { SequenceLength, EmbeddingDimension }),
            new MultiHeadAttentionLayer<float>(HeadCount, EmbeddingDimension / HeadCount,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new FlattenLayer<float>(),
            new DenseLayer<float>(FlatSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
        };

        var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: FlatSize,
            outputSize: FlatSize,
            layers: layers));

        // Materialize the lazy attention weights on the source before anything clones it.
        _ = model.Predict(new Tensor<float>(new[] { 1, FlatSize }));
        return model;
    }

    private static Tensor<float> Token()
    {
        var t = new Tensor<float>(new[] { 1, FlatSize });
        for (var i = 0; i < t.Length; i++) t[i] = 1.0f + i * 0.01f;
        return t;
    }
}
