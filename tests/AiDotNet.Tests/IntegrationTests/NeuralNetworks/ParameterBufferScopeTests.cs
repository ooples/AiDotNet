using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Tests that ParameterBuffer view replacement is scoped to the training step
/// and does not permanently replace layer tensor references (fixes #1084).
/// </summary>
public class ParameterBufferScopeTests
{
    private static FeedForwardNeuralNetwork<double> CreateSimpleNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        return new FeedForwardNeuralNetwork<double>(arch);
    }

    [Fact]
    public void Train_DoesNotPermanentlyReplaceLayerTensors()
    {
        var network = CreateSimpleNetwork();

        // Capture original tensor references before training
        var originalParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                    originalParams.Add(p);
            }
        }

        // Train one step
        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;
        network.Train(input, target);

        // Collect post-training parameters
        var afterParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                    afterParams.Add(p);
            }
        }

        // Same count
        Assert.Equal(originalParams.Count, afterParams.Count);

        // Same tensor references (not buffer views)
        for (int i = 0; i < originalParams.Count; i++)
        {
            Assert.True(ReferenceEquals(afterParams[i], originalParams[i]),
                $"Parameter {i} was permanently replaced with a buffer view after training");
        }
    }

    [Fact]
    public void Train_UpdatesParameterValues()
    {
        var network = CreateSimpleNetwork();

        // Snapshot ALL parameter data before training
        var beforeSnapshot = new Dictionary<int, double[]>();
        int tensorIdx = 0;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    var data = new double[p.Length];
                    p.AsSpan().CopyTo(data);
                    beforeSnapshot[tensorIdx++] = data;
                }
            }
        }

        Assert.True(beforeSnapshot.Count > 0, "Network should have trainable parameters");

        // Train with non-trivial loss
        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

        for (int step = 0; step < 10; step++)
            network.Train(input, target);

        // Compare ALL parameter data after training
        tensorIdx = 0;
        bool anyChanged = false;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    var before = beforeSnapshot[tensorIdx++];
                    for (int i = 0; i < p.Length; i++)
                    {
                        if (Math.Abs(before[i] - p.GetFlat(i)) > 1e-12)
                        {
                            anyChanged = true;
                            break;
                        }
                    }
                    if (anyChanged) break;
                }
                if (anyChanged) break;
            }
        }

        Assert.True(anyChanged, "Training 10 steps with non-trivial loss should update parameter values");
    }

    [Fact]
    public void Train_MultipleConsecutiveSteps_AlwaysRestoreOriginalReferences()
    {
        // Regression: each training step must restore original tensor refs so that
        // references captured before training remain valid across multiple steps.
        var network = CreateSimpleNetwork();

        var originalParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    originalParams.Add(p);
        }

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

        // Run several steps and verify references after each one
        for (int step = 0; step < 5; step++)
        {
            network.Train(input, target);

            int idx = 0;
            foreach (var layer in network.Layers)
            {
                if (layer is ITrainableLayer<double> trainable)
                {
                    foreach (var p in trainable.GetTrainableParameters())
                    {
                        Assert.True(ReferenceEquals(p, originalParams[idx]),
                            $"Step {step}: parameter {idx} was permanently replaced with a buffer view.");
                        idx++;
                    }
                }
            }
        }
    }

    [Fact]
    public void Train_ThenPredict_ReturnsFiniteOutput()
    {
        // After training (which now wraps in try/finally), Predict() must still work correctly.
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 0.5; input[0, 2] = -1.0; input[0, 3] = 2.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 5.0; target[0, 1] = -5.0;

        for (int step = 0; step < 3; step++)
            network.Train(input, target);

        var prediction = network.Predict(new Tensor<double>([4]));

        Assert.NotNull(prediction);
        for (int i = 0; i < prediction.Length; i++)
            Assert.False(double.IsNaN(prediction.GetFlat(i)) || double.IsInfinity(prediction.GetFlat(i)),
                $"Prediction element {i} is not finite after training.");
    }

    [Fact]
    public void Train_LastLossIsPopulatedAfterStep()
    {
        // TrainWithTape now sets LastLoss inside the try block. Verify it's accessible after Train().
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

        network.Train(input, target);

        double loss = Convert.ToDouble(network.GetLastLoss());
        Assert.False(double.IsNaN(loss), "GetLastLoss() must return a valid value after training.");
        Assert.True(loss >= 0.0, "Loss must be non-negative.");
    }

    [Fact]
    public void Train_LossDecreasesOverExtendedTraining()
    {
        // End-to-end regression: the try/finally refactor must not break gradient flow.
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 5.0; target[0, 1] = -5.0;

        // Warm-up to get initial loss
        network.Train(input, target);
        double firstLoss = Convert.ToDouble(network.GetLastLoss());

        // Train for many more steps
        for (int step = 0; step < 50; step++)
            network.Train(input, target);

        double finalLoss = Convert.ToDouble(network.GetLastLoss());

        Assert.True(finalLoss < firstLoss,
            $"Loss should decrease over extended training. Initial: {firstLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void Train_ParameterValuesAreCopiedBackToOriginalTensors()
    {
        // After RestoreOriginalParameters, the original tensor objects must reflect the updated
        // values that were written into the buffer views during the optimizer step.
        var network = CreateSimpleNetwork();

        // Snapshot initial values
        var initialValues = new List<double>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    for (int i = 0; i < p.Length; i++)
                        initialValues.Add(p.GetFlat(i));
        }

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 2.0; input[0, 1] = -3.0; input[0, 2] = 1.0; input[0, 3] = 0.5;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 100.0; target[0, 1] = -100.0;

        for (int step = 0; step < 5; step++)
            network.Train(input, target);

        // Read values back through the original tensor references
        var updatedValues = new List<double>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    for (int i = 0; i < p.Length; i++)
                        updatedValues.Add(p.GetFlat(i));
        }

        bool anyDiffers = false;
        for (int i = 0; i < initialValues.Count; i++)
        {
            if (Math.Abs(initialValues[i] - updatedValues[i]) > 1e-12)
            {
                anyDiffers = true;
                break;
            }
        }

        Assert.True(anyDiffers,
            "Original tensor objects must carry updated values after training — RestoreOriginalParameters must copy data back.");
    }
}