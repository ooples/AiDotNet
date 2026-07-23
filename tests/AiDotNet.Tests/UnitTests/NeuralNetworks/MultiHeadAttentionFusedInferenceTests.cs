using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Parity + correctness guards for the fused multi-head-attention inference path
/// (<see cref="MultiHeadAttentionLayer{T}"/> → <c>IEngine.MultiHeadAttentionForward</c>),
/// the P0 win of issue #1478. The fused single-call kernel must be numerically
/// identical to the decomposed Q/K/V + SDPA + output-projection walk it replaces,
/// across the AIsEval attention shape and from bs1 to bs128 (the batch range where
/// the decomposed path's 50× scaling cliff showed up).
/// </summary>
public class MultiHeadAttentionFusedInferenceTests
{
    private const int HeadCount = 4;
    private const int HeadDim = 16;          // dModel = 64, the AIsEval encoder shape
    private const int DModel = HeadCount * HeadDim;
    private const int SeqLen = 32;

    private static Tensor<float> RandomInput(int batch, int seq, int dModel, int seed)
    {
        var rng = new Random(seed);
        var data = new float[batch * seq * dModel];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, new[] { batch, seq, dModel });
    }

    /// <summary>
    /// The fused inference path (eval mode) must equal the decomposed path (train
    /// mode forces <c>TryFusedAttentionInference</c> to bail) on the SAME materialized
    /// weights, for representative batch sizes including bs128.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(32)]
    [InlineData(128)]
    public void FusedInference_MatchesDecomposed_SelfAttention(int batch)
    {
        AiDotNetEngine.ResetToCpu();

        var layer = new MultiHeadAttentionLayer<float>(
            HeadCount, HeadDim, (IActivationFunction<float>)new IdentityActivation<float>());

        var input = RandomInput(batch, SeqLen, DModel, seed: 1478 + batch);

        // Eval mode → fused single-call kernel. First call also materializes the
        // lazy Q/K/V/O weights; subsequent calls reuse them.
        layer.SetTrainingMode(false);
        var fused = layer.Forward(input);

        // Train mode → decomposed walk (IsTrainingMode gate bails out of the fused
        // path) on the SAME weights. No GradientTape is active, so this is the pure
        // forward math of the path the fused kernel replaces.
        layer.SetTrainingMode(true);
        var decomposed = layer.Forward(input);

        // .ToArray() + NaN/Infinity checks: net471-portable forms (TensorShape has
        // no IEnumerable<int> Assert.Equal overload there; float.IsFinite is net5+).
        Assert.Equal(decomposed.Shape.ToArray(), fused.Shape.ToArray());
        Assert.Equal(new[] { batch, SeqLen, DModel }, fused.Shape.ToArray());

        var f = fused.GetDataArray();
        var d = decomposed.GetDataArray();
        Assert.Equal(d.Length, f.Length);
        float maxAbsDiff = 0f;
        for (int i = 0; i < f.Length; i++)
        {
            Assert.True(!float.IsNaN(f[i]) && !float.IsInfinity(f[i]), $"fused output element {i} is not finite");
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(f[i] - d[i]));
        }
        // Float GEMM + softmax reassociation between the two kernels; 1e-3 is tight.
        Assert.True(maxAbsDiff < 1e-3f, $"fused vs decomposed max|diff| = {maxAbsDiff} at bs{batch}");
    }

    /// <summary>
    /// 2D [seq, dModel] (unbatched) input must also round-trip through the fused path
    /// and match the decomposed path, with the caller's rank preserved.
    /// </summary>
    [Fact]
    public void FusedInference_Preserves2DRank_AndMatchesDecomposed()
    {
        AiDotNetEngine.ResetToCpu();

        var layer = new MultiHeadAttentionLayer<float>(
            HeadCount, HeadDim, (IActivationFunction<float>)new IdentityActivation<float>());

        var rng = new Random(20260602);
        var data = new float[SeqLen * DModel];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var input = new Tensor<float>(data, new[] { SeqLen, DModel });

        layer.SetTrainingMode(false);
        var fused = layer.Forward(input);
        layer.SetTrainingMode(true);
        var decomposed = layer.Forward(input);

        Assert.Equal(new[] { SeqLen, DModel }, fused.Shape.ToArray());
        var f = fused.GetDataArray();
        var d = decomposed.GetDataArray();
        float maxAbsDiff = 0f;
        for (int i = 0; i < f.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(f[i] - d[i]));
        Assert.True(maxAbsDiff < 1e-3f, $"2D fused vs decomposed max|diff| = {maxAbsDiff}");
    }

    /// <summary>
    /// End-to-end: the AIsEval-shaped production Transformer's <c>Predict</c> (which
    /// now routes its encoder self-attention through the fused kernel) returns the
    /// right shape, finite values, and is deterministic across calls at bs1 and bs128.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(128)]
    public void Transformer_Predict_FusedAttention_ShapeAndDeterminism(int batch)
    {
        AiDotNetEngine.ResetToCpu();

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: HeadCount,
            modelDimension: DModel,
            feedForwardDimension: 128,
            inputSize: 32,
            outputSize: 10,
            dropoutRate: 0.0,
            maxSequenceLength: SeqLen,
            vocabularySize: 0,
            usePositionalEncoding: true,
            sequencePooling: SequencePoolingMode.MeanPool);
        var model = new Transformer<float>(arch);

        var rng = new Random(99);
        var data = new float[batch * SeqLen * 32];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var input = new Tensor<float>(data, new[] { batch, SeqLen, 32 });

        var first = model.Predict(input);
        var second = model.Predict(input);

        Assert.Equal(new[] { batch, 10 }, first.Shape.ToArray());
        var a = first.GetDataArray();
        var b = second.GetDataArray();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(!float.IsNaN(a[i]) && !float.IsInfinity(a[i]), $"prediction element {i} is not finite");
            Assert.Equal(a[i], b[i]);   // deterministic
        }
    }
}
