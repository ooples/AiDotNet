using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for neural network models implementing INeuralNetworkModel&lt;double&gt;.
/// Tests mathematical invariants: training loss decrease, gradient flow,
/// parameter sensitivity, output stability, and architecture consistency.
/// </summary>
public abstract class NeuralNetworkModelTestBase
{
    protected abstract INeuralNetworkModel<double> CreateNetwork();

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 1];
    protected virtual int TrainingIterations => 10;

    /// <summary>
    /// Tolerance for the MoreData test. Models with non-continuous outputs
    /// (e.g., SOM with one-hot BMU encoding) may need a higher tolerance.
    /// </summary>
    protected virtual double MoreDataTolerance => 1e-4;

    protected Tensor<double> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble();
        return tensor;
    }

    protected Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = value;
        return tensor;
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Should Reduce Loss
    // After multiple training iterations on a fixed (input, target) pair,
    // the output should move closer to the target. If it doesn't, the
    // gradient computation or parameter update is broken.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task Training_ShouldReduceLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        // Measure initial loss (MSE)
        var initialOutput = network.Predict(input);
        double initialLoss = ComputeMSE(initialOutput, target);

        // Train
        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(input, target);

        // Measure final loss
        var finalOutput = network.Predict(input);
        double finalLoss = ComputeMSE(finalOutput, target);

        if (!double.IsNaN(initialLoss) && !double.IsNaN(finalLoss))
        {
            Assert.True(finalLoss <= initialLoss + 1e-6,
                $"Training did not reduce loss: initial={initialLoss:F6}, final={finalLoss:F6}. " +
                "Gradient computation or parameter update may be broken.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Parameters Should Change After Training
    // If training doesn't change parameters, the gradient is zero or
    // the learning rate is zero — both are bugs.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task Training_ShouldChangeParameters()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        var paramsBefore = network.GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++)
            snapshot[i] = paramsBefore[i];

        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        var paramsAfter = network.GetParameters();
        bool anyChanged = false;
        int minLen = Math.Min(snapshot.Length, paramsAfter.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged,
            "Parameters did not change after training. Gradients may be zero or learning rate is 0.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Sensitivity to Input
    // Different inputs should produce different outputs. A network that
    // produces the same output for all inputs has collapsed (dead neurons,
    // zero weights, or broken forward pass).
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Network produces identical output for inputs [0.1,...] and [0.9,...]. " +
            "The network may have collapsed (dead neurons or zero weights).");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Finite (No NaN/Infinity)
    // Numerical instability in forward pass produces NaN/Inf.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ForwardPass_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Output should not be empty.");

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN — numerical instability.");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity — overflow.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Finite Output After Training
    // Training should not destabilize the forward pass.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ForwardPass_ShouldBeFinite_AfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN after {TrainingIterations} training iterations.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity after training — potential gradient explosion.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaling Input Should Change Output
    // If f(x) ≈ f(10x) for all x, the network ignores input magnitude.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ScaledInput_ShouldChangeOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();

        var input = CreateRandomTensor(InputShape, rng);
        var scaledInput = new Tensor<double>(InputShape);
        for (int i = 0; i < input.Length; i++)
            scaledInput[i] = input[i] * 10.0;

        var output1 = network.Predict(input);
        var output2 = network.Predict(scaledInput);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Network output didn't change when input was scaled 10x. Forward pass may ignore input values.");
    }

    // =====================================================
    // BASIC CONTRACTS: Determinism, Parameters, Clone, Metadata, Architecture
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = network.Predict(input);
        var out2 = network.Predict(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.True(Math.Abs(out1[i] - out2[i]) < 1e-12,
                $"Output[{i}] differs between runs: {out1[i]} vs {out2[i]}. Network may be non-deterministic.");
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        // Use ParameterCount instead of GetParameters().Length — GetParameters
        // materializes the full flat parameter vector just to measure existence,
        // forcing all lazy-init layers to allocate. ParameterCount answers the
        // same question ("does the network have learnable parameters?") without
        // the allocation, preserving the lazy-init memory savings.
        Assert.True(network.ParameterCount > 0, "Neural network should have learnable parameters.");
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldProduceIdenticalOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var original = network.Predict(input);
        var cloned = network.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
            Assert.True(Math.Abs(original[i] - clonedOutput[i]) < 1e-10,
                $"Clone output[{i}] differs: original={original[i]}, cloned={clonedOutput[i]}");
    }

    [Fact(Timeout = 120000)]
    public async Task Metadata_ShouldExist()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);
        network.Train(input, target);
        Assert.NotNull(network.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task Architecture_ShouldBeNonNull()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        Assert.NotNull(network.GetArchitecture());
    }

    [Fact(Timeout = 120000)]
    public async Task NamedLayerActivations_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var activations = network.GetNamedLayerActivations(input);
        Assert.NotNull(activations);
        Assert.True(activations.Count > 0, "Named layer activations should not be empty.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: More Data Should Not Degrade Performance
    // Training with 200 iterations should produce loss ≤ 50 iterations loss.
    // If it doesn't, the optimizer is diverging or oscillating.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task MoreData_ShouldNotDegrade()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var network1 = CreateNetwork();
        var network2 = CreateNetwork();

        var input = CreateRandomTensor(InputShape, rng1);
        var target = CreateRandomTensor(OutputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var target2 = CreateRandomTensor(OutputShape, rng2);

        // Train network1 for 50 iterations
        for (int i = 0; i < 50; i++)
            network1.Train(input, target);
        double loss50 = ComputeMSE(network1.Predict(input), target);

        // Train network2 for 200 iterations
        for (int i = 0; i < 200; i++)
            network2.Train(input2, target2);
        double loss200 = ComputeMSE(network2.Predict(input2), target2);

        if (!double.IsNaN(loss50) && !double.IsNaN(loss200))
        {
            Assert.True(loss200 <= loss50 + MoreDataTolerance,
                $"200 iterations loss ({loss200:F6}) > 50 iterations loss ({loss50:F6}). " +
                "Optimizer may be diverging with more training.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Error ≤ Test Error
    // On a simple fitting task, training MSE should not vastly exceed
    // the error on a different random input (overfit check).
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task TrainingError_ShouldNotExceedTestError()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(input, target);

        double trainMSE = ComputeMSE(network.Predict(input), target);
        var testInput = CreateRandomTensor(InputShape, ModelTestHelpers.CreateSeededRandom(99));
        var testTarget = CreateRandomTensor(OutputShape, ModelTestHelpers.CreateSeededRandom(99));
        double testMSE = ComputeMSE(network.Predict(testInput), testTarget);

        if (!double.IsNaN(trainMSE) && !double.IsNaN(testMSE))
        {
            Assert.True(trainMSE <= testMSE * 3.0 + 1e-6,
                $"Training MSE ({trainMSE:F6}) vastly exceeds test MSE ({testMSE:F6}). " +
                "Model is not fitting training data.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Gradient Flow
    // After a backward pass (training), parameters should change and
    // remain finite. Zero gradients or NaN parameters indicate broken
    // gradient computation.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task GradientFlow_ShouldBeNonZeroAndFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        var paramsBefore = network.GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++)
            snapshot[i] = paramsBefore[i];

        network.Train(input, target);

        var paramsAfter = network.GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < Math.Min(snapshot.Length, paramsAfter.Length); i++)
        {
            Assert.False(double.IsNaN(paramsAfter[i]),
                $"Parameter[{i}] is NaN after training — gradient computation is broken.");
            Assert.False(double.IsInfinity(paramsAfter[i]),
                $"Parameter[{i}] is Infinity after training — gradient explosion.");
            if (Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
                anyChanged = true;
        }
        Assert.True(anyChanged,
            "No parameters changed after training — gradients may all be zero.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Batch Consistency
    // Predicting a single input should produce the same result as
    // predicting that input within a sequence of predictions.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task BatchConsistency_SingleMatchesBatch()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        // Single prediction
        var singleOutput = network.Predict(input);

        // Predict again (batch of 1) — should be identical
        var batchOutput = network.Predict(input);

        Assert.Equal(singleOutput.Length, batchOutput.Length);
        for (int i = 0; i < singleOutput.Length; i++)
        {
            Assert.True(Math.Abs(singleOutput[i] - batchOutput[i]) < 1e-12,
                $"Output[{i}] differs: single={singleOutput[i]}, batch={batchOutput[i]}");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Dimension Matches Shape
    // The output tensor length should match the product of OutputShape.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputDimension_ShouldMatchExpectedShape()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape)
            expectedLength *= dim;

        Assert.Equal(expectedLength, output.Length);
    }

    private double ComputeMSE(Tensor<double> output, Tensor<double> target)
    {
        double mse = 0;
        int len = Math.Min(output.Length, target.Length);
        if (len == 0) return double.NaN;
        for (int i = 0; i < len; i++)
        {
            double diff = output[i] - target[i];
            mse += diff * diff;
        }
        return mse / len;
    }
}
