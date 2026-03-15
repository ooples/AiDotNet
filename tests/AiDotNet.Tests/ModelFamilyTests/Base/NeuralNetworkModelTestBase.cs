using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for neural network models implementing INeuralNetworkModel&lt;double&gt;.
/// Neural networks use Tensor&lt;T&gt; for input/output (not Matrix/Vector).
/// </summary>
public abstract class NeuralNetworkModelTestBase
{
    protected abstract INeuralNetworkModel<double> CreateNetwork();

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 1];
    protected virtual int TrainingIterations => 5;

    private Tensor<double> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble();
        return tensor;
    }

    // --- Forward Pass ---

    [Fact]
    public void Predict_ForwardPass_OutputNotNull()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output tensor should have at least one element.");
    }

    [Fact]
    public void Predict_OutputAllFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN on forward pass.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity on forward pass.");
        }
    }

    // --- Determinism ---

    [Fact]
    public void Predict_Deterministic_SameInputSameOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output1 = network.Predict(input);
        var output2 = network.Predict(input);

        Assert.Equal(output1.Length, output2.Length);
        for (int i = 0; i < output1.Length; i++)
        {
            Assert.Equal(output1[i], output2[i]);
        }
    }

    // --- Training ---

    [Fact]
    public void Train_DoesNotThrow()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        network.Train(input, target);
    }

    [Fact]
    public void Train_ParametersChange()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        var paramsBefore = network.GetParameters();
        var beforeValues = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++)
            beforeValues[i] = paramsBefore[i];

        for (int iter = 0; iter < TrainingIterations; iter++)
            network.Train(input, target);

        var paramsAfter = network.GetParameters();

        bool anyChanged = false;
        int minLen = Math.Min(beforeValues.Length, paramsAfter.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(beforeValues[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged,
            "Parameters should change after training iterations.");
    }

    // --- IParameterizable Contract ---

    [Fact]
    public void GetParameters_ReturnsNonEmptyVector()
    {
        var network = CreateNetwork();
        var parameters = network.GetParameters();

        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0,
            "Neural network should have at least one learnable parameter.");
    }

    // --- Metadata ---

    [Fact]
    public void GetModelMetadata_ReturnsNonNull()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        network.Train(input, target);

        var metadata = network.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    // --- Clone Contract ---

    [Fact]
    public void Clone_ProducesSameOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var original = network.Predict(input);
        var cloned = network.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], clonedOutput[i]);
        }
    }

    // --- Architecture Contract ---

    [Fact]
    public void GetArchitecture_ReturnsNonNull()
    {
        var network = CreateNetwork();
        var architecture = network.GetArchitecture();
        Assert.NotNull(architecture);
    }

    // --- Named Activations Contract ---

    [Fact]
    public void GetNamedLayerActivations_AfterForwardPass_ReturnsNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var activations = network.GetNamedLayerActivations(input);
        Assert.NotNull(activations);
        Assert.True(activations.Count > 0,
            "Named layer activations should not be empty after a forward pass.");
    }
}
