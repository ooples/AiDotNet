using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for diffusion models implementing IDiffusionModel&lt;double&gt;.
/// Tests mathematical invariants for generative diffusion models.
/// </summary>
public abstract class DiffusionModelTestBase
{
    protected abstract IDiffusionModel<double> CreateModel();

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 4];

    private Tensor<double> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble();
        return tensor;
    }

    // --- Forward Pass ---

    [Fact]
    public void Predict_OutputNotNull()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output = model.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output tensor should have at least one element.");
    }

    [Fact]
    public void Predict_OutputAllFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity.");
        }
    }

    // --- Determinism ---

    [Fact]
    public void Predict_Deterministic_SameInputSameOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output1 = model.Predict(input);
        var output2 = model.Predict(input);

        Assert.Equal(output1.Length, output2.Length);
        for (int i = 0; i < output1.Length; i++)
        {
            Assert.Equal(output1[i], output2[i]);
        }
    }

    // --- Different Inputs Different Outputs ---

    [Fact]
    public void Predict_DifferentInputs_DifferentOutputs()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input1 = CreateRandomTensor(InputShape, rng);
        var input2 = CreateRandomTensor(InputShape, rng);

        var output1 = model.Predict(input1);
        var output2 = model.Predict(input2);

        // At least some values should differ
        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-15)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Different inputs should produce different outputs (model may be ignoring input).");
    }

    // --- Training ---

    [Fact]
    public void Train_DoesNotThrow()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        model.Train(input, target);
    }

    [Fact]
    public void Train_OutputAllFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        model.Train(input, target);
        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN after training.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity after training.");
        }
    }

    // --- Metadata & Parameters ---

    [Fact]
    public void GetModelMetadata_ReturnsNonNull()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        model.Train(input, target);

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void GetParameters_ReturnsNonEmptyVector()
    {
        var model = CreateModel();
        var parameters = model.GetParameters();

        Assert.NotNull(parameters);
        Assert.True(parameters.Length > 0,
            "Diffusion model should have at least one learnable parameter.");
    }

    // --- Clone Contract ---

    [Fact]
    public void Clone_ProducesSameOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var original = model.Predict(input);
        var cloned = model.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], clonedOutput[i]);
        }
    }

    // --- Noise Scheduler Contract ---

    [Fact]
    public void Scheduler_NotNull()
    {
        var model = CreateModel();
        var scheduler = model.Scheduler;
        Assert.NotNull(scheduler);
    }
}
