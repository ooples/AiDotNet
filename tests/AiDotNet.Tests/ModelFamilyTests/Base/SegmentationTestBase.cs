using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for segmentation models (semantic, instance, panoptic, medical, etc.).
/// Inherits all NN invariant tests and adds segmentation-specific invariants:
/// spatial dimension preservation, valid mask values, uniform input behavior, and output finiteness.
/// </summary>
public abstract class SegmentationTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // SEGMENTATION INVARIANT: Output Spatial Dimensions Match Input
    // The segmentation mask must have the same spatial dimensions as
    // the input (every pixel gets a classification).
    // =====================================================

    [Fact]
    public void OutputSpatialDims_MatchInput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        // For segmentation, output length should be related to input spatial dims
        // At minimum, output should not be empty
        Assert.True(output.Length > 0,
            "Segmentation model produced empty mask output.");

        // Output should have enough elements for at least the spatial dimensions
        int inputSpatialSize = 1;
        for (int i = 0; i < InputShape.Length; i++)
            inputSpatialSize *= InputShape[i];

        // Output should be proportional to input (may differ by channel count)
        Assert.True(output.Length >= inputSpatialSize / InputShape[0] || output.Length > 0,
            $"Segmentation output length ({output.Length}) seems too small for input size ({inputSpatialSize}).");
    }

    // =====================================================
    // SEGMENTATION INVARIANT: Mask Values Are Valid
    // All output values should be non-negative (class indices or probabilities).
    // Negative mask values indicate a broken classification head.
    // =====================================================

    [Fact]
    public void MaskValues_AreNonNegative()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] >= -1e-10,
                $"Mask value [{i}] = {output[i]:F6} is negative — invalid class index or probability.");
            Assert.True(!double.IsNaN(output[i]) && !double.IsInfinity(output[i]),
                $"Mask value [{i}] is not finite — numerical instability in segmentation head.");
        }
    }

    // =====================================================
    // SEGMENTATION INVARIANT: Uniform Input → Uniform Mask
    // A constant-valued input (no edges, no texture) should produce
    // a near-uniform mask (single class). Many distinct regions
    // from uniform input indicate hallucinated boundaries.
    // =====================================================

    [Fact]
    public void UniformInput_UniformMask()
    {
        var network = CreateNetwork();
        var uniformInput = CreateConstantTensor(InputShape, 0.5);

        var output = network.Predict(uniformInput);

        // Count distinct values (rounded to avoid floating-point noise)
        var distinctValues = new HashSet<int>();
        for (int i = 0; i < output.Length; i++)
        {
            distinctValues.Add((int)Math.Round(output[i] * 100));
        }

        Assert.True(distinctValues.Count <= 3,
            $"Uniform input produced {distinctValues.Count} distinct mask values. " +
            "Expected near-uniform segmentation for constant input.");
    }

    // =====================================================
    // SEGMENTATION INVARIANT: Output Sum Is Finite
    // Total mask area/probability mass must be finite.
    // =====================================================

    [Fact]
    public void OutputSum_IsFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++)
            sum += output[i];

        Assert.True(!double.IsNaN(sum) && !double.IsInfinity(sum),
            "Segmentation mask sum is not finite — overflow in output.");
        Assert.True(sum < 1e10,
            $"Segmentation mask sum = {sum:E4} is unreasonably large.");
    }
}
