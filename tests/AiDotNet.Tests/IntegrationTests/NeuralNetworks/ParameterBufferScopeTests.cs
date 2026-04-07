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

    /// <summary>
    /// Regression test: each individual training step must leave tensor references intact,
    /// not only the first.  Previously <c>RestoreOriginalParameters</c> was missing and views
    /// leaked out permanently after step 1, so steps 2+ saw views instead of originals.
    /// </summary>
    [Fact]
    public void Train_ConsecutiveSteps_PreservesParameterReferencesOnEveryStep()
    {
        var network = CreateSimpleNetwork();

        // Capture original references once, before any training
        var originalParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    originalParams.Add(p);
        }

        Assert.True(originalParams.Count > 0);

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 5.0; target[0, 1] = -5.0;

        // Execute 5 consecutive training steps and verify after each one
        for (int step = 1; step <= 5; step++)
        {
            network.Train(input, target);

            int idx = 0;
            foreach (var layer in network.Layers)
            {
                if (layer is ITrainableLayer<double> trainable)
                {
                    foreach (var p in trainable.GetTrainableParameters())
                    {
                        Assert.True(ReferenceEquals(originalParams[idx], p),
                            $"Step {step}: parameter {idx} is no longer the original tensor reference.");
                        idx++;
                    }
                }
            }
        }
    }

    /// <summary>
    /// The number of trainable parameters must not change across training steps —
    /// buffer view injection or restoration must not alter the parameter list.
    /// </summary>
    [Fact]
    public void Train_ParameterCount_UnchangedAfterMultipleSteps()
    {
        var network = CreateSimpleNetwork();

        // Count parameters before training
        int countBefore = 0;
        foreach (var layer in network.Layers)
            if (layer is ITrainableLayer<double> t)
                countBefore += t.GetTrainableParameters().Count;

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 1.0; target[0, 1] = -1.0;

        for (int step = 0; step < 5; step++)
            network.Train(input, target);

        // Count parameters after training
        int countAfter = 0;
        foreach (var layer in network.Layers)
            if (layer is ITrainableLayer<double> t)
                countAfter += t.GetTrainableParameters().Count;

        Assert.Equal(countBefore, countAfter);
    }

    /// <summary>
    /// After training, the network must still be fully functional for inference
    /// (i.e. the try/finally restore did not corrupt layer state).
    /// </summary>
    [Fact]
    public void Train_NetworkRemainsUsableForInferenceAfterTraining()
    {
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 0.5; input[0, 1] = 0.5; input[0, 2] = 0.5; input[0, 3] = 0.5;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 1.0; target[0, 1] = 1.0;

        // Train several steps
        for (int step = 0; step < 5; step++)
            network.Train(input, target);

        // After training, Predict must not throw and must return a valid tensor
        var exception = Record.Exception(() =>
        {
            var prediction = network.Predict(input);
            Assert.NotNull(prediction);
            Assert.Equal(2, prediction.Length);
        });

        Assert.Null(exception);
    }

    /// <summary>
    /// Snapshot parameter values after step N and verify they persist unchanged until
    /// step N+1 re-trains with the same input/target (i.e. each step writes its own
    /// updates through the original tensors, not through stale view references).
    /// </summary>
    [Fact]
    public void Train_EachStepWritesUpdatesBackToOriginalTensors()
    {
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

        // Run 5 training steps and capture snapshots after each step
        var snapshots = new List<double[]>();
        for (int step = 0; step < 5; step++)
        {
            network.Train(input, target);

            var snapshot = new List<double>();
            foreach (var layer in network.Layers)
                if (layer is ITrainableLayer<double> trainable)
                    foreach (var p in trainable.GetTrainableParameters())
                        for (int i = 0; i < p.Length; i++)
                            snapshot.Add(p.GetFlat(i));

            snapshots.Add(snapshot.ToArray());
        }

        // At least one of the snapshots must differ from the previous one —
        // parameters must actually update across steps.
        bool foundDifference = false;
        for (int s = 1; s < snapshots.Count; s++)
        {
            for (int i = 0; i < snapshots[s].Length; i++)
            {
                if (Math.Abs(snapshots[s][i] - snapshots[s - 1][i]) > 1e-12)
                {
                    foundDifference = true;
                    break;
                }
            }
            if (foundDifference) break;
        }

        Assert.True(foundDifference,
            "Consecutive training steps must produce different parameter values " +
            "— updates are not propagating through the original tensors.");
    }

    /// <summary>
    /// Boundary case: a single-step training should restore references even when
    /// the network has only one trainable layer (edge-case for the recursive
    /// SaveOriginalParameters/RestoreOriginalParameters walk).
    /// </summary>
    [Fact]
    public void Train_SingleLayer_ParameterReferencePreserved()
    {
        // Build a minimal network with a single dense layer
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);
        var network = new FeedForwardNeuralNetwork<double>(arch);

        var originalParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    originalParams.Add(p);

        var input = new Tensor<double>([1, 2]);
        input[0, 0] = 1.0; input[0, 1] = 1.0;
        var target = new Tensor<double>([1, 1]);
        target[0, 0] = 0.5;

        network.Train(input, target);

        int idx = 0;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    Assert.True(ReferenceEquals(originalParams[idx], p),
                        $"Single-layer network: parameter {idx} reference was not restored.");
                    idx++;
                }
            }
        }
    }
}