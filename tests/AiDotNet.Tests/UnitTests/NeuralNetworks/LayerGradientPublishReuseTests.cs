using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// LayerBase reuses one gradient buffer per layer across training steps. A vector a caller already read must never
/// be refilled by a later step, and each step must still publish that step's own gradients.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class LayerGradientPublishReuseTests
{
    [Fact]
    public void GradientsReadAfterAStep_AreNotOverwrittenByTheNextStep()
    {
        var model = Build();
        var dense = (LayerBase<float>)model.Layers[0];

        model.Train(Input(1), Target(0));
        var first = dense.GetParameterGradients();
        Assert.True(first.Length > 0, "no gradients were published, so this test would prove nothing");
        int nonZero = 0; foreach (var g in first.ToArray()) if (g != 0f) nonZero++;
        Assert.True(nonZero > 0, "the published gradients are all zero; GetParameterGradients is not reading what the step published");
        var firstSnapshot = first.ToArray();

        model.Train(Input(2), Target(1));
        var second = dense.GetParameterGradients();

        Assert.Equal(firstSnapshot, first.ToArray());
        Assert.NotSame(first, second);
        bool differs = false;
        for (int i = 0; i < firstSnapshot.Length && !differs; i++) differs = firstSnapshot[i] != second[i];
        Assert.True(differs, "step 2 published the same gradients as step 1; the surface did not update. step 1: "
            + string.Join(",", System.Linq.Enumerable.Take(firstSnapshot, 6)) + " step 2: "
            + string.Join(",", System.Linq.Enumerable.Take(second.ToArray(), 6)));
    }

    [Fact]
    public void UnreadGradients_AreRepublishedEachStep_FromThatStepAlone()
    {
        var reused = Build();
        var fresh = Build();
        fresh.SetParameters(reused.GetParameters());

        // Step 1 is never read on `reused`, so step 2 refills its buffer; `fresh` publishes step 2 only.
        reused.Train(Input(3), Target(2));
        reused.SetParameters(fresh.GetParameters());
        reused.ResetBaseTrainOptimizerState();
        reused.Train(Input(4), Target(0));
        fresh.Train(Input(4), Target(0));

        var expected = ((LayerBase<float>)fresh.Layers[0]).GetParameterGradients().ToArray();
        var actual = ((LayerBase<float>)reused.Layers[0]).GetParameterGradients().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f,
                $"gradient {i}: {actual[i]} vs {expected[i]}; a reused buffer kept step-1 values");
    }

    private static FeedForwardNeuralNetwork<float> Build()
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new DenseLayer<float>(10, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(3, activationFunction: (IActivationFunction<float>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 5,
            outputSize: 3,
            layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(architecture,
            lossFunction: new CrossEntropyWithLogitsLoss<float>());
        model.SetTrainingMode(true);
        return model;
    }

    private static Tensor<float> Input(int seed)
    {
        var rng = new Random(seed);
        var input = new Tensor<float>(new[] { 6, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        return input;
    }

    private static Tensor<float> Target(int shift)
    {
        var target = new Tensor<float>(new[] { 6, 3 });
        for (int row = 0; row < 6; row++) target[row * 3 + (row + shift) % 3] = 1f;
        return target;
    }
}
