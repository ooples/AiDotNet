using AiDotNet.HarmonicEngine.Models;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Harmonic Resonance Engine models.
/// Tests all standard IFullModel invariants plus HRE-specific spectral invariants
/// and numerical stress tests.
/// </summary>
public abstract class HREModelTestBase
{
    protected abstract HREModel<double> CreateModel();

    protected virtual int[] InputShape => [64];
    protected virtual int[] OutputShape => [1];
    protected virtual int TrainingIterations => 5;

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

    protected Tensor<double> CreateSineTensor(int[] shape, double frequency)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = Math.Sin(2 * Math.PI * frequency * i / tensor.Length);
        return tensor;
    }

    // ================================================================
    // STANDARD IFullModel INVARIANTS
    // ================================================================

    [Fact]
    public void ForwardPass_ShouldProduceFiniteOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        Assert.True(output.Length > 0, "Output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity.");
        }
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = model.Predict(input);
        var out2 = model.Predict(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i], 12);
    }

    [Fact]
    public void DifferentInputs_ShouldProduceDifferentOutputs()
    {
        var model = CreateModel();
        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        var output1 = model.Predict(input1);
        var output2 = model.Predict(input2);

        bool anyDifferent = false;
        for (int i = 0; i < Math.Min(output1.Length, output2.Length); i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Model produces identical output for inputs 0.1 and 0.9.");
    }

    [Fact]
    public void Training_ShouldChangeParameters()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        var paramsBefore = model.GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshot[i] = paramsBefore[i];

        for (int i = 0; i < TrainingIterations; i++) model.Train(input, target);

        var paramsAfter = model.GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < Math.Min(snapshot.Length, paramsAfter.Length); i++)
        {
            if (Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged, "Parameters did not change after training.");
    }

    [Fact]
    public void Parameters_ShouldBeFiniteAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations; i++) model.Train(input, target);

        var parameters = model.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.False(double.IsNaN(parameters[i]), $"Parameter[{i}] is NaN after training.");
            Assert.False(double.IsInfinity(parameters[i]), $"Parameter[{i}] is Infinity after training.");
        }
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var original = model.Predict(input);
        var cloned = model.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], clonedOutput[i], 10);
    }

    [Fact]
    public void OutputDimension_ShouldMatchExpectedShape()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape) expectedLength *= dim;
        Assert.Equal(expectedLength, output.Length);
    }

    [Fact]
    public void Output_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e8,
                $"Output[{i}] = {output[i]:E4} is unbounded.");
        }
    }

    // ================================================================
    // HRE SPECTRAL INVARIANTS
    // ================================================================

    [Fact]
    public void SineInput_ShouldProduceFiniteOutput()
    {
        var model = CreateModel();
        var input = CreateSineTensor(InputShape, 5.0);
        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Sine input: Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Sine input: Output[{i}] is Infinity.");
        }
    }

    [Fact]
    public void DifferentFrequencies_ShouldProduceDifferentOutputs()
    {
        var model = CreateModel();
        var input3 = CreateSineTensor(InputShape, 3.0);
        var input11 = CreateSineTensor(InputShape, 11.0);

        var output3 = model.Predict(input3);
        var output11 = model.Predict(input11);

        bool anyDifferent = false;
        for (int i = 0; i < Math.Min(output3.Length, output11.Length); i++)
        {
            if (Math.Abs(output3[i] - output11[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Model produces identical output for 3 Hz and 11 Hz sine waves — not frequency-sensitive.");
    }

    [Fact]
    public void ScaledInput_ShouldChangeOutput()
    {
        var model = CreateModel();
        var input = CreateSineTensor(InputShape, 5.0);
        var scaled = new Tensor<double>(InputShape);
        for (int i = 0; i < input.Length; i++) scaled[i] = input[i] * 10.0;

        var output1 = model.Predict(input);
        var output2 = model.Predict(scaled);

        bool anyDifferent = false;
        for (int i = 0; i < Math.Min(output1.Length, output2.Length); i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Output didn't change when input scaled 10x.");
    }

    // ================================================================
    // STRESS TESTS
    // ================================================================

    [Fact]
    public void ZeroInput_ShouldNotCrash()
    {
        var model = CreateModel();
        var zero = CreateConstantTensor(InputShape, 0.0);
        var output = model.Predict(zero);

        Assert.True(output.Length > 0);
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]), $"Zero input: Output[{i}] is NaN.");
    }

    [Fact]
    public void LargeInput_ShouldNotOverflow()
    {
        var model = CreateModel();
        var large = CreateConstantTensor(InputShape, 1000.0);
        var output = model.Predict(large);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Large input: Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Large input: Output[{i}] is Infinity.");
        }
    }

    [Fact]
    public void NegativeInput_ShouldNotCrash()
    {
        var model = CreateModel();
        var negative = CreateConstantTensor(InputShape, -5.0);
        var output = model.Predict(negative);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Negative input: Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Negative input: Output[{i}] is Infinity.");
        }
    }

    [Fact]
    public void SmallInput_NumericalStability()
    {
        var model = CreateModel();
        var small = CreateConstantTensor(InputShape, 1e-10);
        var output = model.Predict(small);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Small input: Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Small input: Output[{i}] is Infinity.");
        }
    }

    [Fact]
    public void MultipleTrainingCycles_ShouldNotDiverge()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations * 10; i++) model.Train(input, target);

        var output = model.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN after {TrainingIterations * 10} training iterations.");
            Assert.True(Math.Abs(output[i]) < 1e10,
                $"Output[{i}] = {output[i]:E4} diverged after extended training.");
        }
    }
}
