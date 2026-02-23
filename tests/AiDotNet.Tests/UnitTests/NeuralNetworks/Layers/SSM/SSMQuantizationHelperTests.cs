using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="SSMQuantizationHelper{T}"/>.
/// </summary>
public class SSMQuantizationHelperTests
{
    [Fact]
    public void QuantizeSSMLayer_ReducesParameterPrecision()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var originalParams = block.GetParameters();
        var config = new QuantizationConfiguration { TargetBitWidth = 8 };

        SSMQuantizationHelper<float>.QuantizeSSMLayer(block, config);

        var quantizedParams = block.GetParameters();

        // Parameters should be modified (quantized)
        Assert.Equal(originalParams.Length, quantizedParams.Length);
        bool anyDifferent = false;
        for (int i = 0; i < originalParams.Length; i++)
        {
            if (MathF.Abs(originalParams[i] - quantizedParams[i]) > 1e-10f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Quantization should modify at least some parameters");
    }

    [Fact]
    public void QuantizeSSMLayer_WithDProtection_PreservesDParameter()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var originalD = block.GetDParameter();
        var originalDValues = new float[originalD.Length];
        for (int i = 0; i < originalD.Length; i++)
            originalDValues[i] = originalD[i];

        var config = new QuantizationConfiguration { TargetBitWidth = 4 }; // Aggressive quantization

        SSMQuantizationHelper<float>.QuantizeSSMLayer(block, config, protectDParameter: true);

        // D parameter should be preserved exactly
        var afterD = block.GetDParameter();
        for (int i = 0; i < originalDValues.Length; i++)
        {
            Assert.Equal(originalDValues[i], afterD[i]);
        }
    }

    [Fact]
    public void QuantizeSSMLayer_WithoutDProtection_QuantizesDParameter()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var originalD = block.GetDParameter();
        var originalDValues = new float[originalD.Length];
        for (int i = 0; i < originalD.Length; i++)
            originalDValues[i] = originalD[i];

        var config = new QuantizationConfiguration { TargetBitWidth = 4 }; // Aggressive quantization

        SSMQuantizationHelper<float>.QuantizeSSMLayer(block, config, protectDParameter: false);

        // D parameter may be different (quantized)
        var afterD = block.GetDParameter();
        bool anyDifferent = false;
        for (int i = 0; i < originalDValues.Length; i++)
        {
            if (MathF.Abs(originalDValues[i] - afterD[i]) > 1e-10f)
            {
                anyDifferent = true;
                break;
            }
        }
        // With aggressive 4-bit quantization, D parameter should have been modified
        Assert.True(anyDifferent, "D parameter should be quantized when protection is disabled");
    }

    [Fact]
    public void QuantizeSSMLayer_ThrowsOnNullLayer()
    {
        var config = new QuantizationConfiguration { TargetBitWidth = 8 };

        Assert.Throws<ArgumentNullException>(() =>
            SSMQuantizationHelper<float>.QuantizeSSMLayer(null!, config));
    }

    [Fact]
    public void QuantizeSSMLayer_ThrowsOnNullConfig()
    {
        var block = new MambaBlock<float>(4, 32, 8);

        Assert.Throws<ArgumentNullException>(() =>
            SSMQuantizationHelper<float>.QuantizeSSMLayer(block, null!));
    }

    [Fact]
    public void QuantizeSSMLayer_QuantizedBlockStillProducesValidOutput()
    {
        int seqLen = 4;
        int modelDim = 32;
        int stateDim = 8;
        var block = new MambaBlock<float>(seqLen, modelDim, stateDim);
        var config = new QuantizationConfiguration { TargetBitWidth = 8 };

        SSMQuantizationHelper<float>.QuantizeSSMLayer(block, config);

        // Forward pass should still work after quantization
        var input = CreateRandomTensor(new[] { 1, seqLen, modelDim });
        var output = block.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void QuantizeStateCache_ThrowsOnNullCache()
    {
        Assert.Throws<ArgumentNullException>(() =>
            SSMQuantizationHelper<float>.QuantizeStateCache(null!));
    }

    [Fact]
    public void QuantizeStateCache_ThrowsOnInvalidBitWidth()
    {
        var cache = new SSMStateCache<float>();

        Assert.Throws<ArgumentException>(() =>
            SSMQuantizationHelper<float>.QuantizeStateCache(cache, 0));

        Assert.Throws<ArgumentException>(() =>
            SSMQuantizationHelper<float>.QuantizeStateCache(cache, 33));
    }

    [Fact]
    public void QuantizeStateCache_EmptyCache_ReturnsEmptyCompressedCache()
    {
        var cache = new SSMStateCache<float>();
        var compressed = SSMQuantizationHelper<float>.QuantizeStateCache(cache, 8);

        Assert.True(compressed.CompressionEnabled);
        Assert.Equal(0, compressed.CachedLayerCount);
    }

    [Fact]
    public void QuantizeStateCache_MigratesExistingStates()
    {
        var cache = new SSMStateCache<float>();
        var state0 = CreateRandomTensor(new[] { 1, 64, 8 });
        var state1 = CreateRandomTensor(new[] { 1, 64, 8 }, seed: 99);
        cache.CacheSSMState(0, state0);
        cache.CacheSSMState(1, state1);

        var compressed = SSMQuantizationHelper<float>.QuantizeStateCache(cache, 8);

        Assert.True(compressed.CompressionEnabled);
        Assert.Equal(2, compressed.CachedLayerCount);
        Assert.True(compressed.HasSSMState(0));
        Assert.True(compressed.HasSSMState(1));

        // Compressed states should be approximately equal to originals
        var compressedState0 = compressed.GetSSMState(0)!;
        var originalArr = state0.ToArray();
        var compressedArr = compressedState0.ToArray();
        Assert.Equal(originalArr.Length, compressedArr.Length);

        // With 8-bit quantization, values should be close but not exact
        double maxError = 0;
        for (int i = 0; i < originalArr.Length; i++)
        {
            maxError = Math.Max(maxError, Math.Abs(originalArr[i] - compressedArr[i]));
        }
        // 8-bit quantization over [-1,1] range: max error ~= 2/255 ≈ 0.0078
        Assert.True(maxError < 0.02, $"Max quantization error {maxError} exceeds tolerance");
    }

    [Fact]
    public void QuantizeStateCache_MigratesConvBuffers()
    {
        var cache = new SSMStateCache<float>();
        var state = CreateRandomTensor(new[] { 1, 64, 8 });
        var convBuf = CreateRandomTensor(new[] { 1, 64, 3 }, seed: 77);
        cache.CacheSSMState(0, state);
        cache.CacheConvBuffer(0, convBuf);

        var compressed = SSMQuantizationHelper<float>.QuantizeStateCache(cache, 8);

        Assert.True(compressed.HasSSMState(0));
        Assert.True(compressed.HasConvBuffer(0));

        var retrievedBuf = compressed.GetConvBuffer(0)!;
        Assert.Equal(convBuf.Shape, retrievedBuf.Shape);
    }

    [Fact]
    public void QuantizeStateCache_LowerBitWidth_ProducesMoreError()
    {
        var cache = new SSMStateCache<float>();
        var state = CreateRandomTensor(new[] { 1, 32, 8 });
        cache.CacheSSMState(0, state);

        var compressed8 = SSMQuantizationHelper<float>.QuantizeStateCache(cache, 8);
        var compressed4 = SSMQuantizationHelper<float>.QuantizeStateCache(cache, 4);

        var original = state.ToArray();
        var arr8 = compressed8.GetSSMState(0)!.ToArray();
        var arr4 = compressed4.GetSSMState(0)!.ToArray();

        double error8 = 0, error4 = 0;
        for (int i = 0; i < original.Length; i++)
        {
            error8 += Math.Abs(original[i] - arr8[i]);
            error4 += Math.Abs(original[i] - arr4[i]);
        }

        // 4-bit should produce more error than 8-bit
        Assert.True(error4 >= error8, $"4-bit error ({error4}) should be >= 8-bit error ({error8})");
    }

    [Fact]
    public void EstimateMemorySavings_ThrowsOnNullLayer()
    {
        Assert.Throws<ArgumentNullException>(() =>
            SSMQuantizationHelper<float>.EstimateMemorySavings(null!, 8));
    }

    [Fact]
    public void EstimateMemorySavings_8bit_ReturnsApprox4xCompression()
    {
        var block = new MambaBlock<float>(4, 32, 8);

        var (originalBytes, quantizedBytes, ratio) =
            SSMQuantizationHelper<float>.EstimateMemorySavings(block, 8);

        Assert.True(originalBytes > 0);
        Assert.True(quantizedBytes > 0);
        Assert.True(quantizedBytes < originalBytes);
        // 8-bit from 32-bit should give roughly 3-4x compression (with overhead)
        Assert.True(ratio > 2.0, $"Compression ratio {ratio} should be > 2x for 8-bit");
        Assert.True(ratio < 5.0, $"Compression ratio {ratio} should be < 5x for 8-bit");
    }

    [Fact]
    public void EstimateMemorySavings_4bit_HigherCompressionThan8bit()
    {
        var block = new MambaBlock<float>(4, 32, 8);

        var (_, _, ratio8) = SSMQuantizationHelper<float>.EstimateMemorySavings(block, 8);
        var (_, _, ratio4) = SSMQuantizationHelper<float>.EstimateMemorySavings(block, 4);

        Assert.True(ratio4 > ratio8,
            $"4-bit ratio ({ratio4}) should be greater than 8-bit ratio ({ratio8})");
    }

    [Fact]
    public void ComputeQuantizationError_ThrowsOnNullLayer()
    {
        Assert.Throws<ArgumentNullException>(() =>
            SSMQuantizationHelper<float>.ComputeQuantizationError(null!, 8));
    }

    [Fact]
    public void ComputeQuantizationError_8bit_IsSmall()
    {
        var block = new MambaBlock<float>(4, 32, 8);

        double error = SSMQuantizationHelper<float>.ComputeQuantizationError(block, 8);

        Assert.True(error >= 0);
        // 8-bit quantization should have very small mean absolute error
        Assert.True(error < 0.1, $"8-bit quantization error {error} should be < 0.1");
    }

    [Fact]
    public void ComputeQuantizationError_LowerBitWidth_ProducesMoreError()
    {
        var block = new MambaBlock<float>(4, 32, 8);

        double error8 = SSMQuantizationHelper<float>.ComputeQuantizationError(block, 8);
        double error4 = SSMQuantizationHelper<float>.ComputeQuantizationError(block, 4);
        double error2 = SSMQuantizationHelper<float>.ComputeQuantizationError(block, 2);

        Assert.True(error4 >= error8,
            $"4-bit error ({error4}) should be >= 8-bit error ({error8})");
        Assert.True(error2 >= error4,
            $"2-bit error ({error2}) should be >= 4-bit error ({error4})");
    }

    [Fact]
    public void ComputeQuantizationError_DoesNotModifyLayer()
    {
        var block = new MambaBlock<float>(4, 32, 8);
        var originalParams = block.GetParameters();

        SSMQuantizationHelper<float>.ComputeQuantizationError(block, 8);

        var afterParams = block.GetParameters();
        for (int i = 0; i < originalParams.Length; i++)
        {
            Assert.Equal(originalParams[i], afterParams[i]);
        }
    }

    [Fact]
    public void QuantizeSSMLayer_NonMambaBlock_QuantizesNormally()
    {
        // Use S4DLayer which is not a MambaBlock - D parameter protection should be skipped
        var layer = new S4DLayer<float>(4, 32, 8);
        var config = new QuantizationConfiguration { TargetBitWidth = 8 };

        // Should not throw even with protectDParameter=true (not a MambaBlock)
        SSMQuantizationHelper<float>.QuantizeSSMLayer(layer, config, protectDParameter: true);

        // Layer should still work after quantization
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        var output = layer.Forward(input);
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void QuantizeSSMLayer_Double_Works()
    {
        var block = new MambaBlock<double>(4, 32, 8);
        var config = new QuantizationConfiguration { TargetBitWidth = 8 };

        SSMQuantizationHelper<double>.QuantizeSSMLayer(block, config);

        var input = CreateRandomDoubleTensor(new[] { 1, 4, 32 });
        var output = block.Forward(input);
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void EstimateMemorySavings_Double_Works()
    {
        var block = new MambaBlock<double>(4, 32, 8);

        var (originalBytes, quantizedBytes, ratio) =
            SSMQuantizationHelper<double>.EstimateMemorySavings(block, 8);

        Assert.True(originalBytes > 0);
        Assert.True(ratio > 1.0);
    }

    [Fact]
    public void ComputeQuantizationError_Double_Works()
    {
        var block = new MambaBlock<double>(4, 32, 8);

        double error = SSMQuantizationHelper<double>.ComputeQuantizationError(block, 8);

        Assert.True(error >= 0);
        Assert.True(error < 0.1);
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    private static Tensor<double> CreateRandomDoubleTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble() * 2 - 1;
        }
        return tensor;
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (float.IsNaN(value)) return true;
        }
        return false;
    }

    private static bool ContainsNaNDouble(Tensor<double> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (double.IsNaN(value)) return true;
        }
        return false;
    }

    #endregion
}
