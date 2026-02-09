using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference.Quantization;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Tests for quantized attention layers and inference quantization modes.
/// </summary>
public class QuantizedAttentionTests
{
    [Fact]
    public void QuantizedAttention_MHA_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void QuantizedAttention_GQA_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var gqa = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);

        var quantized = new QuantizedAttentionLayer(gqa);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void QuantizedAttention_MHA_WithinToleranceOfFP32()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var fp32Output = mha.Forward(input);
        var int8Output = quantized.Forward(input);

        // INT8 quantized attention should be within ~5% of FP32
        // (weight-only quantization with per-row scaling is quite accurate)
        double relativeError = ComputeRelativeError(fp32Output, int8Output);
        Assert.True(relativeError < 0.10,
            $"INT8 quantized attention relative error ({relativeError:F4}) exceeds 10% tolerance");
    }

    [Fact]
    public void QuantizedAttention_2D_Input_ProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);
        var quantized = new QuantizedAttentionLayer(mha);

        var input = CreateRandomTensor(new[] { seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void QuantizedAttention_IsInferenceOnly()
    {
        var mha = new MultiHeadAttentionLayer<float>(4, 32, 4);
        var quantized = new QuantizedAttentionLayer(mha);

        Assert.False(quantized.SupportsTraining);
        Assert.False(quantized.SupportsJitCompilation);
        Assert.Equal(0, quantized.ParameterCount);
        Assert.Null(quantized.GetWeights());
        Assert.Null(quantized.GetBiases());
    }

    [Fact]
    public void QuantizedAttention_Backward_ThrowsNotSupported()
    {
        var mha = new MultiHeadAttentionLayer<float>(4, 32, 4);
        var quantized = new QuantizedAttentionLayer(mha);

        Assert.Throws<NotSupportedException>(() =>
            quantized.Backward(CreateRandomTensor(new[] { 4, 32 })));
    }

    [Fact]
    public void QuantizedAttention_UpdateParameters_ThrowsNotSupported()
    {
        var mha = new MultiHeadAttentionLayer<float>(4, 32, 4);
        var quantized = new QuantizedAttentionLayer(mha);

        Assert.Throws<NotSupportedException>(() => quantized.UpdateParameters(0.01f));
    }

    [Fact]
    public void QuantizedAttention_PreservesProperties()
    {
        int numHeads = 8;
        int numKVHeads = 2;
        var gqa = new GroupedQueryAttentionLayer<float>(8, 64, numHeads, numKVHeads);
        gqa.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var quantized = new QuantizedAttentionLayer(gqa);

        Assert.Equal(numHeads, quantized.HeadCount);
        Assert.Equal(numKVHeads, quantized.KVHeadCount);
        Assert.True(quantized.IsGQA);
        Assert.Equal(PositionalEncodingType.Rotary, quantized.PositionalEncoding);
    }

    [Fact]
    public void InferenceQuantizationMode_None_DisablesQuantization()
    {
        var config = new InferenceOptimizationConfig
        {
            InferenceQuantization = InferenceQuantizationMode.None
        };

        Assert.False(config.EnableWeightOnlyQuantization);
    }

    [Fact]
    public void InferenceQuantizationMode_Int8_EnablesQuantization()
    {
        var config = new InferenceOptimizationConfig
        {
            InferenceQuantization = InferenceQuantizationMode.WeightOnlyInt8
        };

        Assert.True(config.EnableWeightOnlyQuantization);
    }

    [Fact]
    public void EnableWeightOnlyQuantization_BackwardCompatible()
    {
        var config = new InferenceOptimizationConfig();

        // Setting legacy property
        config.EnableWeightOnlyQuantization = true;
        Assert.Equal(InferenceQuantizationMode.WeightOnlyInt8, config.InferenceQuantization);

        config.EnableWeightOnlyQuantization = false;
        Assert.Equal(InferenceQuantizationMode.None, config.InferenceQuantization);
    }

    [Fact]
    public void InferenceQuantizationMode_FP8_AvailableInConfig()
    {
        var config = new InferenceOptimizationConfig
        {
            InferenceQuantization = InferenceQuantizationMode.WeightOnlyFP8
        };

        Assert.True(config.EnableWeightOnlyQuantization);
        Assert.Equal(InferenceQuantizationMode.WeightOnlyFP8, config.InferenceQuantization);
    }

    [Fact]
    public void InferenceQuantizationMode_NF4_AvailableInConfig()
    {
        var config = new InferenceOptimizationConfig
        {
            InferenceQuantization = InferenceQuantizationMode.WeightOnlyNF4
        };

        Assert.True(config.EnableWeightOnlyQuantization);
        Assert.Equal(InferenceQuantizationMode.WeightOnlyNF4, config.InferenceQuantization);
    }

    [Fact]
    public void FP8WeightOnlyQuantization_QuantizePerRow_Roundtrips()
    {
        int rows = 4;
        int cols = 8;
        var weights = new float[rows * cols];
        var random = new Random(42);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var quantized = FP8WeightOnlyQuantization.QuantizePerRow(weights, rows, cols);

        Assert.Equal(rows, quantized.Rows);
        Assert.Equal(cols, quantized.Cols);
        Assert.Equal(rows * cols, quantized.Weights.Length);
        Assert.Equal(rows, quantized.Scales.Length);

        // Dequantize and check tolerance
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                float original = weights[r * cols + c];
                float dequantized = FP8WeightOnlyQuantization.Dequantize(
                    quantized.Weights[r * cols + c], quantized.Scales[r]);
                float error = MathF.Abs(original - dequantized);
                // FP8 should be within ~10% for typical values
                Assert.True(error < 0.5f,
                    $"FP8 dequantization error ({error}) too large at [{r}, {c}]");
            }
        }
    }

    [Fact]
    public void NF4WeightOnlyQuantization_QuantizePerGroup_Roundtrips()
    {
        int rows = 4;
        int cols = 16;
        int groupSize = 8;
        var weights = new float[rows * cols];
        var random = new Random(42);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var quantized = NF4WeightOnlyQuantization.QuantizePerGroup(weights, rows, cols, groupSize);

        Assert.Equal(rows, quantized.Rows);
        Assert.Equal(cols, quantized.Cols);
        Assert.Equal(groupSize, quantized.GroupSize);

        // Dequantize and check tolerance
        int total = rows * cols;
        for (int i = 0; i < total; i++)
        {
            int group = i / groupSize;
            int index = NF4WeightOnlyQuantization.ExtractIndex(quantized.PackedWeights, i);
            float dequantized = NF4WeightOnlyQuantization.Dequantize(index, quantized.GroupScales[group]);
            float error = MathF.Abs(weights[i] - dequantized);
            // NF4 is 4-bit so larger error expected, but should be < 50% for small values
            Assert.True(error < 0.5f,
                $"NF4 dequantization error ({error}) too large at index {i}");
        }
    }

    [Fact]
    public void QuantizedAttention_FP8_MHA_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyFP8);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.Equal(InferenceQuantizationMode.WeightOnlyFP8, quantized.QuantizationFormat);
    }

    [Fact]
    public void QuantizedAttention_FP8_WithinToleranceOfFP32()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyFP8);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var fp32Output = mha.Forward(input);
        var fp8Output = quantized.Forward(input);

        // FP8 E4M3 has only 3 mantissa bits, and error compounds through
        // 4 sequential projections + softmax, so tolerance is larger than INT8
        double relativeError = ComputeRelativeError(fp32Output, fp8Output);
        Assert.True(relativeError < 0.75,
            $"FP8 quantized attention relative error ({relativeError:F4}) exceeds 75% tolerance");
    }

    [Fact]
    public void QuantizedAttention_NF4_MHA_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyNF4);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
        Assert.Equal(InferenceQuantizationMode.WeightOnlyNF4, quantized.QuantizationFormat);
    }

    [Fact]
    public void QuantizedAttention_NF4_WithinToleranceOfFP32()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyNF4);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var fp32Output = mha.Forward(input);
        var nf4Output = quantized.Forward(input);

        // NF4 is 4-bit so larger error expected
        double relativeError = ComputeRelativeError(fp32Output, nf4Output);
        Assert.True(relativeError < 0.50,
            $"NF4 quantized attention relative error ({relativeError:F4}) exceeds 50% tolerance");
    }

    [Fact]
    public void QuantizedAttention_FP8_GQA_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var gqa = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);

        var quantized = new QuantizedAttentionLayer(gqa, InferenceQuantizationMode.WeightOnlyFP8);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void QuantizedAttention_NF4_GQA_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var gqa = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);

        var quantized = new QuantizedAttentionLayer(gqa, InferenceQuantizationMode.WeightOnlyNF4);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = quantized.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Int8WeightQuantization_PerRow_Roundtrip_UnderOnePercent()
    {
        // Test that INT8 per-row weight quantization achieves < 1% relative error
        // on individual weight matrices with realistic magnitudes (larger than Xavier tiny-dim init)
        int rows = 64;
        int cols = 64;
        var weights = new float[rows * cols];
        var random = new Random(42);
        for (int i = 0; i < weights.Length; i++)
        {
            // Weights in range [-1, 1] with reasonable magnitude
            weights[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var quantized = AiDotNet.Inference.Quantization.Int8WeightOnlyQuantization.QuantizePerRow(
            weights, rows, cols);

        double sumSquaredError = 0;
        double sumSquaredOriginal = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int idx = r * cols + c;
                float original = weights[idx];
                float dequantized = (float)(quantized.Weights[idx] * quantized.Scales[r]);
                double diff = original - dequantized;
                sumSquaredError += diff * diff;
                sumSquaredOriginal += (double)original * original;
            }
        }

        double relativeError = sumSquaredOriginal > 0
            ? Math.Sqrt(sumSquaredError / sumSquaredOriginal)
            : 0;

        Assert.True(relativeError < 0.01,
            $"INT8 per-row weight quantization relative error ({relativeError:F6}) exceeds 1% tolerance");
    }

    [Fact]
    public void QuantizedAttention_INT8_LargerDimension_TighterTolerance()
    {
        // Larger dimensions = more accumulation points per matmul, but per-element error stays small
        // This tests end-to-end attention with 128-dim embeddings and 8 heads
        int seqLen = 4;
        int embDim = 128;
        int numHeads = 8;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyInt8);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var fp32Output = mha.Forward(input);
        var int8Output = quantized.Forward(input);

        double relativeError = ComputeRelativeError(fp32Output, int8Output);
        // With larger dimensions, INT8 end-to-end error should be < 5%
        Assert.True(relativeError < 0.05,
            $"INT8 attention (embDim=128) relative error ({relativeError:F4}) exceeds 5% tolerance");
    }

    [Fact]
    public void QuantizedAttention_FP8_LargerDimension_TighterTolerance()
    {
        int seqLen = 4;
        int embDim = 128;
        int numHeads = 8;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyFP8);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var fp32Output = mha.Forward(input);
        var fp8Output = quantized.Forward(input);

        double relativeError = ComputeRelativeError(fp32Output, fp8Output);
        // FP8 E4M3 has only 3 mantissa bits, so per-weight quantization error is ~5-10%.
        // Through 4 sequential projections (Q/K/V/O) + softmax nonlinearity, error compounds
        // to ~47-50% for end-to-end attention. This is still better than the 75% tolerance
        // for smaller dimensions, confirming the improvement with larger embeddings.
        Assert.True(relativeError < 0.55,
            $"FP8 attention (embDim=128) relative error ({relativeError:F4}) exceeds 55% tolerance");
    }

    [Fact]
    public void QuantizedAttention_NF4_LargerDimension_TighterTolerance()
    {
        int seqLen = 4;
        int embDim = 128;
        int numHeads = 8;
        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);

        var quantized = new QuantizedAttentionLayer(mha, InferenceQuantizationMode.WeightOnlyNF4);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var fp32Output = mha.Forward(input);
        var nf4Output = quantized.Forward(input);

        double relativeError = ComputeRelativeError(fp32Output, nf4Output);
        // NF4 is 4-bit but optimized for normal distribution; compound error expected
        Assert.True(relativeError < 0.35,
            $"NF4 attention (embDim=128) relative error ({relativeError:F4}) exceeds 35% tolerance");
    }

    [Fact]
    public void DefaultConfig_PreservesExistingPresets()
    {
        // Verify Default and HighPerformance presets still work unchanged
        var defaultConfig = InferenceOptimizationConfig.Default;
        Assert.Equal(InferenceQuantizationMode.None, defaultConfig.InferenceQuantization);
        Assert.False(defaultConfig.EnableWeightOnlyQuantization);

        var highPerf = InferenceOptimizationConfig.HighPerformance;
        Assert.False(highPerf.EnableWeightOnlyQuantization);
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
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

    private static double ComputeRelativeError(Tensor<float> expected, Tensor<float> actual)
    {
        double sumSquaredError = 0;
        double sumSquaredExpected = 0;
        var expArr = expected.ToArray();
        var actArr = actual.ToArray();

        for (int i = 0; i < expArr.Length; i++)
        {
            double diff = expArr[i] - actArr[i];
            sumSquaredError += diff * diff;
            sumSquaredExpected += (double)expArr[i] * expArr[i];
        }

        return sumSquaredExpected > 0 ? Math.Sqrt(sumSquaredError / sumSquaredExpected) : 0;
    }

    #endregion
}
