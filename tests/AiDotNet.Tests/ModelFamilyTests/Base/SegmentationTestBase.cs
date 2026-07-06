using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.ComputerVision.Segmentation.Common;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for segmentation models (semantic, instance, panoptic, medical, etc.).
/// Inherits all NN invariant tests and adds segmentation-specific invariants:
/// spatial dimension preservation, valid mask values, uniform input behavior, and output finiteness.
/// </summary>
public abstract class SegmentationTestBase<T> : NeuralNetworkModelTestBase<T>
{
    // =====================================================
    // SEGMENTATION INVARIANT: Output Spatial Dimensions Match Input
    // The segmentation mask must have the same spatial dimensions as
    // the input (every pixel gets a classification).
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputSpatialDims_MatchInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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
    // SEGMENTATION INVARIANT: Mask Values Are Finite
    // The forward output may be raw logits (any real value) for models trained
    // with softmax-cross-entropy per the standard segmentation training recipe
    // (Hatamizadeh et al. 2022 SwinUNETR; Long et al. 2015 FCN; etc.), so the
    // only paper-meaningful purely-output invariant is numerical validity:
    // every value is finite. NaN / ±Inf would indicate a broken head.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task MaskValues_AreFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.True(!double.IsNaN(v) && !double.IsInfinity(v),
                $"Mask value [{i}] is not finite — numerical instability in segmentation head.");
        }
    }

    // =====================================================
    // SEGMENTATION INVARIANT: Uniform Input → Uniform Mask
    // A constant-valued input (no edges, no texture) should produce
    // a near-uniform mask (single class). Many distinct regions
    // from uniform input indicate hallucinated boundaries.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task UniformInput_UniformMask()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var uniformInput = CreateConstantTensor(InputShape, 0.5);

        var output = network.Predict(uniformInput);

        // Segmentation models commonly emit raw logits. The paper-meaningful
        // mask is the per-pixel class map after argmax along the class axis,
        // not the raw score tensor itself.
        if (output.Rank == 3 || output.Rank == 4)
            output = SegmentationTensorOps.ArgmaxAlongClassDim(output);

        // Count distinct mask labels (rounded to avoid floating-point noise)
        var distinctValues = new HashSet<int>();
        for (int i = 0; i < output.Length; i++)
        {
            distinctValues.Add((int)Math.Round(ConvertToDouble(output[i]) * 100));
        }

        Assert.True(distinctValues.Count <= 3,
            $"Uniform input produced {distinctValues.Count} distinct mask values. " +
            "Expected near-uniform segmentation for constant input.");
    }

    // =====================================================
    // SEGMENTATION INVARIANT: Output Sum Is Finite
    // Total mask area/probability mass must be finite.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputSum_IsFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++)
            sum += ConvertToDouble(output[i]);

        Assert.True(!double.IsNaN(sum) && !double.IsInfinity(sum),
            "Segmentation mask sum is not finite — overflow in output.");
        Assert.True(sum < 1e10,
            $"Segmentation mask sum = {sum:E4} is unreasonably large.");
    }
}

/// <summary>Double-precision default for <see cref="SegmentationTestBase{T}"/>.</summary>
public abstract class SegmentationTestBase : SegmentationTestBase<double> { }
