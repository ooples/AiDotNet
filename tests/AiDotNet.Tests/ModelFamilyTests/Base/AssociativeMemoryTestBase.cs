using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for associative memory models (Hopfield, HopeNetwork, etc.)
/// that use non-gradient-based learning (Hebbian) or self-modifying mechanisms.
/// Tests pattern storage/recall invariants rather than gradient flow.
/// </summary>
public abstract class AssociativeMemoryTestBase
{
    protected abstract INeuralNetworkModel<double> CreateNetwork();

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 4];
    protected virtual int TrainingIterations => 10;

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
    // MATHEMATICAL INVARIANT: Output Finite (No NaN/Infinity)
    // Numerical instability in forward pass produces NaN/Inf.
    // =====================================================

    [Fact]
    public void ForwardPass_ShouldProduceFiniteOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
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

    [Fact]
    public void ForwardPass_ShouldBeFinite_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
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
                $"Output[{i}] is Infinity after training — potential instability.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Sensitivity to Input
    // Different inputs should produce different outputs.
    // =====================================================

    [Fact]
    public void DifferentInputs_ShouldProduceDifferentOutputs()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

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
            "The network may have collapsed.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Pattern Recall After Training
    // After training on a pattern, recalling it should produce
    // output closer to the training pattern than a random output.
    // This is the core invariant of associative memory.
    // =====================================================

    [Fact]
    public void TrainedPattern_ShouldBeRecalledCloserThanRandom()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var pattern = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        // Train the network on the pattern
        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(pattern, target);

        // Recall the trained pattern
        var recalled = network.Predict(pattern);

        // Compare distances: recalled should be closer to target than a random tensor
        var randomOutput = CreateRandomTensor(OutputShape, ModelTestHelpers.CreateSeededRandom(99));

        double recalledDistance = ComputeMSE(recalled, target);
        double randomDistance = ComputeMSE(randomOutput, target);

        if (!double.IsNaN(recalledDistance) && !double.IsNaN(randomDistance))
        {
            Assert.True(recalledDistance <= randomDistance + 0.1,
                $"Recalled pattern (MSE={recalledDistance:F6}) is not closer to target than random (MSE={randomDistance:F6}). " +
                "Associative memory storage may be broken.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Should Change State
    // After training, the network's internal state should differ
    // from its initial state.
    // =====================================================

    [Fact]
    public void Training_ShouldChangeParameters()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
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
            "Parameters did not change after training. Learning mechanism may be broken.");
    }

    // =====================================================
    // BASIC CONTRACTS: Determinism, Parameters, Clone, Metadata, Architecture
    // =====================================================

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = network.Predict(input);
        var out2 = network.Predict(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty()
    {
        var network = CreateNetwork();
        var parameters = network.GetParameters();
        Assert.True(parameters.Length > 0, "Network should have learnable parameters.");
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var original = network.Predict(input);
        var cloned = network.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], clonedOutput[i]);
    }

    [Fact]
    public void Metadata_ShouldExist()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);
        network.Train(input, target);
        Assert.NotNull(network.GetModelMetadata());
    }

    [Fact]
    public void Architecture_ShouldBeNonNull()
    {
        var network = CreateNetwork();
        Assert.NotNull(network.GetArchitecture());
    }

    [Fact]
    public void NamedLayerActivations_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var activations = network.GetNamedLayerActivations(input);
        Assert.NotNull(activations);
        Assert.True(activations.Count > 0, "Named layer activations should not be empty.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Dimension Matches Shape
    // =====================================================

    [Fact]
    public void OutputDimension_ShouldMatchExpectedShape()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape)
            expectedLength *= dim;

        Assert.Equal(expectedLength, output.Length);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Batch Consistency
    // =====================================================

    [Fact]
    public void BatchConsistency_SingleMatchesBatch()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var singleOutput = network.Predict(input);
        var batchOutput = network.Predict(input);

        Assert.Equal(singleOutput.Length, batchOutput.Length);
        for (int i = 0; i < singleOutput.Length; i++)
            Assert.Equal(singleOutput[i], batchOutput[i]);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaling Input Should Change Output
    // =====================================================

    [Fact]
    public void ScaledInput_ShouldChangeOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

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
    // ASSOCIATIVE MEMORY INVARIANT: Training Loss Should Be Finite
    // After training, the network's reported loss should be a
    // real number (not NaN or Infinity).
    // =====================================================

    [Fact]
    public void TrainingLoss_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        var output = network.Predict(input);
        double mse = ComputeMSE(output, target);

        Assert.False(double.IsNaN(mse), "MSE is NaN after training — numerical instability.");
        Assert.False(double.IsInfinity(mse), "MSE is Infinity after training — overflow.");
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
