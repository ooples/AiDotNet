using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for ILayer&lt;double&gt; implementations.
/// Tests mathematical invariants that every layer must satisfy:
/// finite forward output, backward gradient flow, parameter consistency,
/// serialization roundtrip, and input sensitivity.
///
/// Subclasses override CreateLayer() and optionally InputShape/OutputShape.
/// All invariant tests are inherited automatically.
/// </summary>
public abstract class LayerTestBase
{
    /// <summary>
    /// Factory method — create a fresh instance of the layer under test.
    /// </summary>
    protected abstract ILayer<double> CreateLayer();

    /// <summary>
    /// Shape of the tensor to feed into Forward. Override for layers that need
    /// specific shapes (e.g. [batch, channels, height, width] for conv layers).
    /// Default: [1, 4] — single sample, 4 features.
    /// </summary>
    protected virtual int[] InputShape => [1, 4];

    /// <summary>
    /// Whether the layer is expected to have trainable parameters.
    /// Override to false for pass-through layers (InputLayer, FlattenLayer, ActivationLayer, etc.)
    /// </summary>
    protected virtual bool ExpectsTrainableParameters => true;

    /// <summary>
    /// Whether the layer's Backward produces meaningful gradients.
    /// Some layers (ReservoirLayer, InputLayer) pass gradients through but
    /// don't compute weight gradients. Override to false for those.
    /// </summary>
    protected virtual bool ExpectsNonZeroGradients => true;

    /// <summary>
    /// Tolerance for numerical comparisons. Layers with stochastic behavior
    /// (dropout, noise) may need higher tolerance.
    /// </summary>
    protected virtual double Tolerance => 1e-12;

    /// <summary>
    /// Whether constant inputs (all 0.1 vs all 0.9) should produce different outputs.
    /// False for normalization layers (LayerNorm, BatchNorm on single-feature constant input)
    /// where constant inputs normalize to the same output by design.
    /// </summary>
    protected virtual bool ExpectsDifferentOutputForConstantInputs => true;

    // =========================================================================
    // Helpers
    // =========================================================================

    protected static Tensor<double> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() * 2.0 - 1.0; // [-1, 1]
        return tensor;
    }

    protected static Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = value;
        return tensor;
    }

    // =========================================================================
    // INVARIANT 1: Forward produces finite, non-empty output
    // If the forward pass returns NaN/Inf/empty, the layer is numerically broken.
    // =========================================================================

    [Fact]
    public void Forward_ShouldProduceFiniteOutput()
    {
        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);

        Assert.True(output.Length > 0, "Layer output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN — numerical instability in Forward.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity — overflow in Forward.");
        }
    }

    // =========================================================================
    // INVARIANT 2: Forward is deterministic (same input → same output)
    // Unless the layer has stochastic behavior (dropout), two calls with the
    // same input must produce bit-identical output.
    // =========================================================================

    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(false); // Disable dropout/stochastic behavior
        var input = CreateRandomTensor(InputShape);

        var out1 = layer.Forward(input);
        layer.ResetState(); // Reset any recurrent state
        var out2 = layer.Forward(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
        {
            Assert.Equal(out1[i], out2[i]);
        }
    }

    // =========================================================================
    // INVARIANT 3: Different inputs produce different outputs
    // A layer that maps all inputs to the same output is broken (zero weights,
    // dead neurons, or input-ignoring forward pass).
    // =========================================================================

    [Fact]
    public void Forward_DifferentInputs_ShouldProduceDifferentOutputs()
    {
        if (!ExpectsDifferentOutputForConstantInputs) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(false);

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        layer.ResetState();
        var output1 = layer.Forward(input1);
        layer.ResetState();
        var output2 = layer.Forward(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > Tolerance)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Layer produces identical output for inputs [0.1,...] and [0.9,...]. " +
            "Forward pass may ignore input values.");
    }

    // =========================================================================
    // INVARIANT 4: Output shape is consistent
    // GetOutputShape() must match the actual shape produced by Forward.
    // =========================================================================

    [Fact]
    public void Forward_OutputShape_ShouldMatchGetOutputShape()
    {
        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);
        var declaredShape = layer.GetOutputShape();

        // The output length should equal the product of declared output shape
        // (batch dimension may differ, so compare total feature count)
        int declaredFeatureCount = 1;
        foreach (var dim in declaredShape)
            declaredFeatureCount *= dim;

        // Allow for batch dimension: output.Length may be batch * declaredFeatureCount
        Assert.True(output.Length > 0, "Output should not be empty.");
        Assert.True(output.Length % declaredFeatureCount == 0 || declaredFeatureCount == 0,
            $"Output length {output.Length} is not a multiple of declared output shape " +
            $"[{string.Join(",", declaredShape)}] (product={declaredFeatureCount}).");
    }

    // =========================================================================
    // INVARIANT 5: Backward produces finite gradient
    // After Forward+Backward, the returned input gradient must be finite.
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceFiniteGradient()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor(InputShape);

        var output = layer.Forward(input);

        // Create gradient matching output shape
        var outputGrad = CreateRandomTensor(output.Shape, seed: 99);

        var inputGrad = layer.Backward(outputGrad);

        Assert.True(inputGrad.Length > 0, "Input gradient should not be empty.");
        for (int i = 0; i < inputGrad.Length; i++)
        {
            Assert.False(double.IsNaN(inputGrad[i]),
                $"InputGradient[{i}] is NaN — broken backward pass.");
            Assert.False(double.IsInfinity(inputGrad[i]),
                $"InputGradient[{i}] is Infinity — gradient explosion in backward pass.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Parameter count is non-negative and GetParameters matches
    // ParameterCount must equal GetParameters().Length for trainable layers.
    // =========================================================================

    [Fact]
    public void Parameters_CountShouldMatchVector()
    {
        var layer = CreateLayer();

        int count = layer.ParameterCount;
        var parameters = layer.GetParameters();

        Assert.True(count >= 0, "ParameterCount should be non-negative.");
        Assert.Equal(count, parameters.Length);

        if (ExpectsTrainableParameters)
        {
            Assert.True(count > 0,
                "Layer is expected to have trainable parameters but ParameterCount is 0.");
        }
    }

    // =========================================================================
    // INVARIANT 7: SetParameters → GetParameters roundtrip
    // Setting parameters and getting them back should return the same values.
    // =========================================================================

    [Fact]
    public void Parameters_SetGet_Roundtrip()
    {
        var layer = CreateLayer();
        if (layer.ParameterCount == 0) return; // Skip for non-trainable layers

        var original = layer.GetParameters();
        var modified = new Vector<double>(original.Length);
        for (int i = 0; i < original.Length; i++)
            modified[i] = original[i] + 0.001; // Small perturbation

        layer.SetParameters(modified);
        var retrieved = layer.GetParameters();

        Assert.Equal(modified.Length, retrieved.Length);
        for (int i = 0; i < modified.Length; i++)
        {
            Assert.Equal(modified[i], retrieved[i], 1e-15);
        }
    }

    // =========================================================================
    // INVARIANT 8: Backward produces non-zero weight gradients
    // For trainable layers, after Forward+Backward, GetParameterGradients
    // should have at least some non-zero values.
    // =========================================================================

    [Fact]
    public void Backward_ShouldProduceNonZeroWeightGradients()
    {
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(true);
        layer.ClearGradients();

        var input = CreateRandomTensor(InputShape);
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, seed: 99);

        layer.Backward(outputGrad);

        var gradients = layer.GetParameterGradients();
        Assert.True(gradients.Length > 0,
            "Trainable layer should have parameter gradients after Backward.");

        bool anyNonZero = false;
        for (int i = 0; i < gradients.Length; i++)
        {
            if (Math.Abs(gradients[i]) > 1e-15)
            {
                anyNonZero = true;
                break;
            }
        }
        Assert.True(anyNonZero,
            "All parameter gradients are zero after Backward. " +
            "Gradient computation is likely broken.");
    }

    // =========================================================================
    // INVARIANT 9: Serialization roundtrip preserves behavior
    // Serialize → Deserialize should produce a layer with identical Forward output.
    // =========================================================================

    [Fact]
    public void Serialize_Deserialize_ShouldPreserveBehavior()
    {
        var layer = CreateLayer();
        layer.SetTrainingMode(false);
        var input = CreateRandomTensor(InputShape);

        var originalOutput = layer.Forward(input);

        // Serialize
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer.Serialize(writer);
        }

        // Deserialize into a fresh layer
        var layer2 = CreateLayer();
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer2.Deserialize(reader);
        }

        layer2.SetTrainingMode(false);
        layer2.ResetState();
        var deserializedOutput = layer2.Forward(input);

        Assert.Equal(originalOutput.Length, deserializedOutput.Length);
        for (int i = 0; i < originalOutput.Length; i++)
        {
            Assert.True(Math.Abs(originalOutput[i] - deserializedOutput[i]) < 1e-12,
                $"Output[{i}] differs after serialization roundtrip: " +
                $"original={originalOutput[i]:G17}, deserialized={deserializedOutput[i]:G17}");
        }
    }

    // =========================================================================
    // INVARIANT 10: ResetState doesn't break the layer
    // After ResetState, Forward should still produce valid (finite) output.
    // =========================================================================

    [Fact]
    public void ResetState_ShouldNotBreakForward()
    {
        var layer = CreateLayer();
        var input = CreateRandomTensor(InputShape);

        // Forward once to populate state
        layer.Forward(input);

        // Reset
        layer.ResetState();

        // Forward again — should still work
        var output = layer.Forward(input);
        Assert.True(output.Length > 0, "Output should not be empty after ResetState.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN after ResetState + Forward.");
        }
    }

    // =========================================================================
    // INVARIANT 11: ClearGradients zeroes all gradients
    // After ClearGradients, GetParameterGradients should return all zeros.
    // =========================================================================

    [Fact]
    public void ClearGradients_ShouldZeroAllGradients()
    {
        if (!ExpectsTrainableParameters) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(true);

        // Accumulate some gradients
        var input = CreateRandomTensor(InputShape);
        var output = layer.Forward(input);
        var outputGrad = CreateRandomTensor(output.Shape, seed: 99);
        layer.Backward(outputGrad);

        // Clear
        layer.ClearGradients();

        var gradients = layer.GetParameterGradients();
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(0.0, gradients[i], 1e-15);
        }
    }

    // =========================================================================
    // INVARIANT 12: Numerical gradient check (finite differences)
    // For trainable layers, verify that analytical gradients match numerical
    // approximation: dL/dw ≈ (L(w+ε) - L(w-ε)) / 2ε
    // This is the gold standard for gradient correctness.
    // =========================================================================

    [Fact]
    public void Backward_NumericalGradientCheck()
    {
        if (!ExpectsTrainableParameters || !ExpectsNonZeroGradients) return;

        var layer = CreateLayer();
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(InputShape);
        double epsilon = 1e-5;

        // Forward + backward to get analytical gradients
        layer.ClearGradients();
        var output = layer.Forward(input);
        // Use simple MSE-like loss: L = sum(output^2) / 2
        // dL/dOutput = output
        var outputGrad = new Tensor<double>(output.Shape);
        for (int i = 0; i < output.Length; i++)
            outputGrad[i] = output[i];
        layer.Backward(outputGrad);
        var analyticalGradients = layer.GetParameterGradients();

        if (analyticalGradients.Length == 0) return;

        // Numerical gradient check for a sample of parameters
        var parameters = layer.GetParameters();
        int checkCount = Math.Min(10, parameters.Length); // Check first 10 params
        int failCount = 0;
        var debugInfo = new System.Text.StringBuilder();

        for (int p = 0; p < checkCount; p++)
        {
            // L(w + ε)
            var paramsPlus = parameters.Clone();
            paramsPlus[p] += epsilon;
            layer.SetParameters(paramsPlus);
            layer.ResetState();
            var outputPlus = layer.Forward(input);
            double lossPlus = 0;
            for (int i = 0; i < outputPlus.Length; i++)
                lossPlus += outputPlus[i] * outputPlus[i];
            lossPlus /= 2.0;

            // L(w - ε)
            var paramsMinus = parameters.Clone();
            paramsMinus[p] -= epsilon;
            layer.SetParameters(paramsMinus);
            layer.ResetState();
            var outputMinus = layer.Forward(input);
            double lossMinus = 0;
            for (int i = 0; i < outputMinus.Length; i++)
                lossMinus += outputMinus[i] * outputMinus[i];
            lossMinus /= 2.0;

            double numericalGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
            double analyticalGrad = analyticalGradients[p];

            // Relative error check (handle near-zero gradients)
            double absMax = Math.Max(Math.Abs(numericalGrad), Math.Abs(analyticalGrad));
            if (absMax < 1e-7) continue; // Both near zero, skip

            double relError = Math.Abs(numericalGrad - analyticalGrad) / (absMax + 1e-8);
            if (relError > 0.01) // 1% relative error threshold
            {
                failCount++;
                debugInfo.Append($"p[{p}]: analytical={analyticalGrad:G6} numerical={numericalGrad:G6} relErr={relError:G4} | ");
            }
        }

        // Restore original parameters
        layer.SetParameters(parameters);

        Assert.True(failCount <= checkCount / 3,
            $"Numerical gradient check failed for {failCount}/{checkCount} parameters. " +
            $"Analytical gradients don't match finite differences. " +
            $"Details: {debugInfo}");
    }
}
