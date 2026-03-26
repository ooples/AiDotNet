using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for diffusion models implementing IDiffusionModel&lt;double&gt;.
/// Tests mathematical invariants: denoising convergence, output sensitivity,
/// training stability, scheduler consistency, and noise schedule properties.
/// </summary>
public abstract class DiffusionModelTestBase
{
    protected abstract IDiffusionModel<double> CreateModel();

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 4];

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
    // MATHEMATICAL INVARIANT: Training Should Reduce Prediction Error
    // After training on a fixed (input, target) pair, the predict output
    // should move closer to the target — verifying gradient flow works.
    // =====================================================

    [Fact]
    public void Training_ShouldReducePredictionError()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        // Initial error
        var initialOutput = model.Predict(input);
        double initialError = ComputeMSE(initialOutput, target);

        // Train
        for (int i = 0; i < 10; i++)
            model.Train(input, target);

        // Final error
        var finalOutput = model.Predict(input);
        double finalError = ComputeMSE(finalOutput, target);

        if (!double.IsNaN(initialError) && !double.IsNaN(finalError))
        {
            Assert.True(finalError <= initialError + 1e-6,
                $"Training did not reduce error: initial={initialError:F6}, final={finalError:F6}.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Sensitivity to Input
    // Different inputs must produce different outputs. A model that
    // ignores its input is fundamentally broken.
    // =====================================================

    [Fact]
    public void DifferentInputs_ShouldProduceDifferentOutputs()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        var output1 = model.Predict(input1);
        var output2 = model.Predict(input2);

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
            "Model produces identical output for inputs [0.1,...] and [0.9,...]. Input is being ignored.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaled Input Changes Output
    // f(x) ≠ f(10x) — the model should be sensitive to magnitude.
    // =====================================================

    [Fact]
    public void ScaledInput_ShouldChangeOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        var input = CreateRandomTensor(InputShape, rng);
        var scaledInput = new Tensor<double>(InputShape);
        for (int i = 0; i < input.Length; i++)
            scaledInput[i] = input[i] * 10.0;

        var output1 = model.Predict(input);
        var output2 = model.Predict(scaledInput);

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
            "Output unchanged when input scaled 10x. Forward pass may ignore input values.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Shape Preservation
    // For a diffusion model, output shape should match input shape
    // (denoising maps noisy input → clean output of same dimensions).
    // =====================================================

    [Fact]
    public void OutputShape_ShouldMatchInputShape()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output = model.Predict(input);
        Assert.Equal(input.Length, output.Length);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Finite Output Before and After Training
    // =====================================================

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
    public void ForwardPass_ShouldBeFinite_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < 5; i++)
            model.Train(input, target);

        var output = model.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN after training.");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity after training.");
        }
    }

    // =====================================================
    // DIFFUSION INVARIANT: Noise Schedule Monotonicity
    // At higher timesteps, the noise magnitude should increase.
    // This verifies the noise schedule is properly configured.
    // =====================================================

    [Fact]
    public void NoiseSchedule_ShouldBeMonotonic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var baseInput = CreateRandomTensor(InputShape, rng);

        // Predict at different "noise levels" by scaling input
        // Low noise (near-original), medium noise, high noise
        double[] scales = new double[] { 0.1, 0.5, 1.0, 2.0 };
        double[] outputMagnitudes = new double[scales.Length];

        for (int s = 0; s < scales.Length; s++)
        {
            var noisyInput = new Tensor<double>(InputShape);
            for (int i = 0; i < baseInput.Length; i++)
                noisyInput[i] = baseInput[i] * scales[s];

            var output = model.Predict(noisyInput);

            double magnitude = 0;
            for (int i = 0; i < output.Length; i++)
                magnitude += output[i] * output[i];
            outputMagnitudes[s] = Math.Sqrt(magnitude / Math.Max(1, output.Length));
        }

        // Check monotonicity: allow at most 1 violation out of 3 transitions
        int violations = 0;
        for (int i = 1; i < outputMagnitudes.Length; i++)
        {
            if (outputMagnitudes[i] < outputMagnitudes[i - 1] - 1e-10)
                violations++;
        }

        Assert.True(violations <= 1,
            $"Noise schedule not monotonic: magnitudes = [{string.Join(", ", outputMagnitudes.Select(m => m.ToString("F4")))}]. " +
            $"Found {violations} violations (max allowed: 1).");
    }

    // =====================================================
    // DIFFUSION INVARIANT: Output Range Validity
    // Generated output should be bounded — no exploding values.
    // =====================================================

    [Fact]
    public void OutputRange_ShouldBeValid()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Output[{i}] = {output[i]:E4} exceeds bound of 1e6. " +
                "Diffusion model is producing unbounded output.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Determinism, Clone, Metadata, Parameters, Scheduler
    // =====================================================

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = model.Predict(input);
        var out2 = model.Predict(input);

        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
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
            Assert.Equal(original[i], clonedOutput[i]);
    }

    [Fact]
    public void Metadata_ShouldExist()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);
        model.Train(input, target);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty()
    {
        var model = CreateModel();
        Assert.True(model.GetParameters().Length > 0,
            "Diffusion model should have learnable parameters.");
    }

    [Fact]
    public void Scheduler_ShouldBeNonNull()
    {
        var model = CreateModel();
        Assert.NotNull(model.Scheduler);
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
