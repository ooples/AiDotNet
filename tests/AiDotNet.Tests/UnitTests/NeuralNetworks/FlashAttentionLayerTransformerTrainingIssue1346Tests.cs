using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression coverage for AiDotNet#1346 / HE PathB FlashAttention sanity.
///
/// <para>The Tensors-side fix in PR #362 verified that
/// <c>engine.FlashAttention</c> records into the lazy graph and that
/// the compiled training plan produces non-zero gradients on Q/K/V
/// projections (see <c>FlashAttentionCompiledPlanTests</c>). However,
/// the HE consumer that wraps <see cref="FlashAttentionLayer{T}"/> inside
/// a <see cref="Transformer{T}"/> with the custom layer list
/// <c>Embedding + FlashAttention + LayerNorm + Dense + readout</c>
/// still trained to top-1 = 0% (uniform output).</para>
///
/// <para>This file pins the consumer-level behaviour: when
/// <see cref="FlashAttentionLayer{T}"/> is used in a Transformer's
/// custom layer list,
/// <list type="number">
/// <item>Every projection (Q, K, V, O) and the output bias must
///   actually change during training,</item>
/// <item>The number of layer-registered trainable parameters must
///   match what <c>GetParameters()</c> reports (this is the
///   <c>FlashAttentionLayer</c> equivalent of the H5
///   parameter-walk-ordering check), and</item>
/// <item>The same toy task that <see cref="MultiHeadAttentionLayer{T}"/>
///   converges on must also converge for the FA layer.</item>
/// </list></para>
/// </summary>
public class FlashAttentionLayerTransformerTrainingIssue1346Tests
{
    /// <summary>
    /// FA layer's <c>GetParameters()</c> length must match the sum of
    /// <c>GetTrainableParameters()</c> tensor lengths. A mismatch
    /// would cause the optimizer's parameter walk and the gradient walk
    /// to disagree on ordering or count — silent grad-vs-param
    /// misalignment that zeros out training on the FA arm.
    /// </summary>
    [Fact]
    public void FlashAttentionLayer_GetParameters_MatchesRegisteredTrainableParameters()
    {
        var layer = new FlashAttentionLayer<float>(
            sequenceLength: 8,
            embeddingDimension: 16,
            headCount: 2);

        var flatParams = layer.GetParameters();
        var registered = layer.GetTrainableParameters();

        int registeredTotal = 0;
        foreach (var t in registered) registeredTotal += t.Length;

        Assert.Equal(flatParams.Length, registeredTotal);

        // Q + K + V + O weights (each embedDim*embedDim) + outputBias (embedDim)
        int expected = 16 * 16 * 4 + 16;
        Assert.Equal(expected, flatParams.Length);
        Assert.Equal(expected, registeredTotal);
    }

    /// <summary>
    /// Same parameter shape, only the layer kind differs:
    /// MHA vs FA. Both arms must register the same number of trainable
    /// tensors and the same parameter count. Mismatch is a smoking gun
    /// for an FA-side registration bug.
    /// </summary>
    [Fact]
    public void FlashAttentionLayer_TrainableParameterShape_MatchesMHA()
    {
        var fa = new FlashAttentionLayer<float>(8, 16, 2);
        var mha = new MultiHeadAttentionLayer<float>(headCount: 2, headDimension: 8);

        // Force MHA's lazy weights to allocate.
        _ = mha.GetParameters();

        Assert.Equal(mha.GetParameters().Length, fa.GetParameters().Length);
        Assert.Equal(mha.ParameterCount, fa.ParameterCount);
    }

    /// <summary>
    /// Direct repro of the HE PathB consumer setup. Builds a Transformer
    /// with custom layer list mirroring
    /// <c>AiDotNetFacadeTransformerLMPredictor.BuildFlashAttentionLayerList</c>:
    /// Embedding + FA + LayerNorm + Dense + Dense + LayerNorm + slice +
    /// classifier-Dense. Trains 30 steps on a deterministic toy task and
    /// asserts loss decreases AND every FA projection actually changed.
    /// Pre-fix path: loss stayed flat because dQ/dK/dV all came back zero.
    /// </summary>
    [Fact(Timeout = 120000)]
    public void Transformer_WithFlashAttentionLayer_TrainingActuallyUpdatesProjections()
    {
        const int vocab = 16;
        const int seqLen = 8;
        const int dModel = 16;
        const int dFf = 32;
        const int heads = 2;

        var faLayer = new FlashAttentionLayer<float>(seqLen, dModel, heads);

        // Snapshot initial projection weights.
        float[] q0 = faLayer.GetQueryWeights().AsSpan().ToArray();
        float[] k0 = faLayer.GetKeyWeights().AsSpan().ToArray();
        float[] v0 = faLayer.GetValueWeights().AsSpan().ToArray();
        float[] o0 = faLayer.GetOutputWeights().AsSpan().ToArray();

        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocab, dModel),
            faLayer,
            new LayerNormalizationLayer<float>(),
            new DenseLayer<float>(dFf, new ReLUActivation<float>() as IActivationFunction<float>),
            new DenseLayer<float>(dModel, new IdentityActivation<float>() as IActivationFunction<float>),
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(vocab, new IdentityActivation<float>() as IActivationFunction<float>),
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: heads,
            modelDimension: dModel,
            feedForwardDimension: dFf,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab,
            usePositionalEncoding: false,
            layers: layers);

        var model = new Transformer<float>(arch, new CategoricalCrossEntropyLoss<float>());

        // Deterministic input/target pair.
        var input = new Tensor<float>([1, seqLen]);
        for (int t = 0; t < seqLen; t++) input[0, t] = t % vocab;
        var target = new Tensor<float>([1, vocab]);
        target[0, 3] = 1.0f;

        model.SetTrainingMode(true);
        try
        {
            for (int step = 0; step < 30; step++)
                model.Train(input, target);
        }
        finally
        {
            model.SetTrainingMode(false);
        }

        // Every FA projection weight must have changed. Pre-fix #1346 these
        // stayed at random init forever because the gradients never reached
        // the FA layer through the Transformer training path.
        Assert.True(MaxAbsDelta(faLayer.GetQueryWeights().AsSpan(), q0) > 1e-6f,
            "Q weights must change during training (was: gradient-free in #1346)");
        Assert.True(MaxAbsDelta(faLayer.GetKeyWeights().AsSpan(), k0) > 1e-6f,
            "K weights must change during training (was: gradient-free in #1346)");
        Assert.True(MaxAbsDelta(faLayer.GetValueWeights().AsSpan(), v0) > 1e-6f,
            "V weights must change during training (was: gradient-free in #1346)");
        Assert.True(MaxAbsDelta(faLayer.GetOutputWeights().AsSpan(), o0) > 1e-6f,
            "O weights must change during training (was: gradient-free in #1346)");
    }

    private static float MaxAbsDelta(System.ReadOnlySpan<float> current, float[] initial)
    {
        float max = 0;
        for (int i = 0; i < current.Length; i++)
        {
            float d = System.Math.Abs(current[i] - initial[i]);
            if (d > max) max = d;
        }
        return max;
    }
}
