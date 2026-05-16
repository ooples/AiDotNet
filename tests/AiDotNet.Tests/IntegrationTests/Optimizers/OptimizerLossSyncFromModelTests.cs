#nullable disable
using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression coverage for AiDotNet#1334:
/// <c>GradientBasedOptimizerOptions.LossFunction</c> defaults to
/// <see cref="MeanSquaredErrorLoss{T}"/>. Before the fix, when callers configured
/// a non-MSE loss on their model (e.g. <see cref="CategoricalCrossEntropyLoss{T}"/>
/// on a classification Transformer) and handed the model to an optimizer without
/// also setting the optimizer's loss, training silently used MSE. With one-hot
/// vocab targets <c>[batch, vocab]</c> the resulting shape mismatch crashed
/// <c>TensorSubtract</c> inside the optimizer's gradient path.
///
/// The fix routes the auto-sync through
/// <c>GradientBasedOptimizerBase.OnModelChanged</c> — when the model is (re-)set
/// and the caller did NOT explicitly configure the optimizer's
/// <c>LossFunction</c>, the optimizer adopts the model's
/// <c>DefaultLossFunction</c>. An explicit set on the optimizer always wins.
/// </summary>
public class OptimizerLossSyncFromModelTests
{
    private static FeedForwardNeuralNetwork<float> BuildToyNetwork(ILossFunction<float> loss)
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(2, (IActivationFunction<float>)new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputSize: 4,
            outputSize: 2,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch, lossFunction: loss);
    }

    [Fact]
    public void OptionsDefault_LossFunction_AdoptedFromModelOnSetModel()
    {
        // Build a model whose loss is CCE.
        var modelLoss = new CategoricalCrossEntropyLoss<float>();
        var model = BuildToyNetwork(modelLoss);

        // Optimizer options where the caller did NOT set LossFunction.
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
        };
        Assert.False(options.LossFunctionExplicitlySet,
            "Sanity: default-constructed options must report LossFunctionExplicitlySet=false.");

        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
        optimizer.SetModel(model);

        // The optimizer's runtime gradient path now uses CCE — this is the
        // load-bearing assertion (it's what training reads).
        Assert.IsType<CategoricalCrossEntropyLoss<float>>(optimizer.CurrentLossFunction);
        // And the options object is mirrored so debug/serialization paths see
        // the synced value too.
        Assert.IsType<CategoricalCrossEntropyLoss<float>>(options.LossFunction);
    }

    [Fact]
    public void OptionsExplicitMse_RespectedEvenWhenModelLossIsCce()
    {
        // Caller deliberately chooses MSE on the optimizer even though the model's
        // default loss is CCE. The auto-sync MUST NOT overwrite the caller's choice.
        var model = BuildToyNetwork(new CategoricalCrossEntropyLoss<float>());

        var explicitOptimizerLoss = new MeanSquaredErrorLoss<float>();
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            LossFunction = explicitOptimizerLoss,
        };
        Assert.True(options.LossFunctionExplicitlySet);

        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
        optimizer.SetModel(model);

        // Caller's instance preserved exactly on both surfaces.
        Assert.Same(explicitOptimizerLoss, optimizer.CurrentLossFunction);
        Assert.Same(explicitOptimizerLoss, options.LossFunction);
    }

    [Fact]
    public void OptionsExplicitCce_NotReplacedByModelMse()
    {
        // Inverse of the first test: model has MSE (the optimizer-default), but
        // the caller explicitly wants CCE on the optimizer. The auto-sync must
        // not silently replace it with the model's MSE.
        var model = BuildToyNetwork(new MeanSquaredErrorLoss<float>());

        var explicitOptimizerLoss = new CategoricalCrossEntropyLoss<float>();
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            LossFunction = explicitOptimizerLoss,
        };

        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
        optimizer.SetModel(model);

        Assert.Same(explicitOptimizerLoss, optimizer.CurrentLossFunction);
        Assert.Same(explicitOptimizerLoss, options.LossFunction);
    }

    [Fact]
    public void SetModel_Twice_StillSyncsLossWhenNotExplicit()
    {
        // Plumbing detail: the auto-sync fires on every SetModel call, not just
        // the first. So re-setting the model also re-syncs the loss.
        var firstLoss = new MeanSquaredErrorLoss<float>();
        var firstModel = BuildToyNetwork(firstLoss);

        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
        optimizer.SetModel(firstModel);
        Assert.IsType<MeanSquaredErrorLoss<float>>(optimizer.CurrentLossFunction);

        var secondLoss = new CategoricalCrossEntropyLoss<float>();
        var secondModel = BuildToyNetwork(secondLoss);
        optimizer.SetModel(secondModel);
        Assert.IsType<CategoricalCrossEntropyLoss<float>>(optimizer.CurrentLossFunction);
    }
}
