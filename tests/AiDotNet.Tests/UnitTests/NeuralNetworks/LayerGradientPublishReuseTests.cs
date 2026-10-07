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

    // A fused step's layer surfaces borrow the plan's gradient buffers and copy on first read. A model read only
    // after its last step must report exactly what a model read after every step reports for that same step: the
    // borrowed buffers must hold that step's gradients, not an earlier step's and not a half-overwritten mix.
    [Fact]
    public void FusedGradientsReadOnlyAfterTheLastStep_EqualThoseOfAModelReadEveryStep()
    {
        var readEveryStep = Build();
        var readAtEnd = Build();
        readAtEnd.SetParameters(readEveryStep.GetParameters());

        // The fused-step counter is per model (it reads the state of the model that stepped last).
        long fusedEveryStep = 0, fusedAtEnd = 0;
        var lastRead = new float[readEveryStep.Layers.Count][];
        for (int step = 0; step < 4; step++)
        {
            readEveryStep.Train(Input(10 + step), Target(step));
            fusedEveryStep = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
            readAtEnd.Train(Input(10 + step), Target(step));
            fusedAtEnd = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
            for (int l = 0; l < readEveryStep.Layers.Count; l++)
                lastRead[l] = ((LayerBase<float>)readEveryStep.Layers[l]).GetParameterGradients().ToArray();
        }
        Assert.True(fusedEveryStep >= 4 && fusedAtEnd >= 4,
            $"the fused compiled path ran {fusedEveryStep} and {fusedAtEnd} of 4 steps, so the borrowed publication was never exercised");

        for (int l = 0; l < readAtEnd.Layers.Count; l++)
        {
            var actual = ((LayerBase<float>)readAtEnd.Layers[l]).GetParameterGradients().ToArray();
            Assert.True(actual.Length == lastRead[l].Length && actual.Length > 0,
                $"layer {l}: {actual.Length} gradients published, expected {lastRead[l].Length}");
            Assert.Equal(lastRead[l], actual);
        }
        Assert.Equal(readEveryStep.GetParameterGradients().ToArray(), readAtEnd.GetParameterGradients().ToArray());
    }

    // The lease contract on its own: a read while the lease is live copies the borrowed buffers (later writes to
    // them do not reach the copy), and a publication still unread when the lease is revoked reports no gradient
    // rather than whatever the buffers hold by then.
    [Fact]
    public void BorrowedPublication_CopiesOnReadWhileLive_AndIsWithdrawnOnRevoke()
    {
        var model = Build();
        var dense = (LayerBase<float>)model.Layers[0];
        var map = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>(
            AiDotNet.Helpers.TensorReferenceComparer<Tensor<float>>.Instance);
        int expectedCount = 0;
        foreach (var parameter in dense.GetTrainableParameters())
        {
            var gradient = new Tensor<float>(parameter.Shape.ToArray());
            for (int i = 0; i < gradient.Length; i++) gradient[i] = 0.25f * (expectedCount + i + 1);
            map[parameter] = gradient;
            expectedCount += gradient.Length;
        }
        Assert.True(expectedCount > 0, "the dense layer exposed no trainable tensors, so this test would prove nothing");

        var lease = new LayerBase<float>.BorrowedGradientLease();
        Assert.Equal(expectedCount, dense.ScatterParameterGradients(map, lease));
        var read = dense.GetParameterGradients().ToArray();
        Assert.Equal(expectedCount, read.Length);
        foreach (var gradient in map.Values)
            for (int i = 0; i < gradient.Length; i++) gradient[i] = -1f;
        Assert.Equal(read, dense.GetParameterGradients().ToArray());

        var unread = new LayerBase<float>.BorrowedGradientLease();
        Assert.Equal(expectedCount, dense.ScatterParameterGradients(map, unread));
        unread.Revoke();
        Assert.Empty(dense.GetParameterGradients().ToArray());
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
