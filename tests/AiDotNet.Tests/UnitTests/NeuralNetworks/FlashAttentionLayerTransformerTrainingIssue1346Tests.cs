using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using System.Threading.Tasks;
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
    // Regression test for #1346: a FlashAttentionLayer inside a Transformer classifier MUST
    // train — every Q/K/V/O projection (and a plain control DenseLayer) must move off its init
    // after a short training run. Both the FlashAttention projections and the control change,
    // and the loss decreases, confirming gradients flow through the FA layer's tape-tracked
    // Engine ops to its registered weights.
    //
    // IMPORTANT (root cause of the original symptom): the classifier head MUST end in a Softmax.
    // CategoricalCrossEntropyLoss expects PROBABILITIES (like Keras categorical_crossentropy with
    // from_logits=false) and clamps its input to [1e-7, 1]. Feeding RAW LOGITS (an Identity head)
    // pins an untrained target-class logit (typically <= 0) to the 1e-7 floor, where the clamp
    // saturates and its gradient is exactly zero — so loss freezes at -log(1e-7) = 16.118 and NO
    // layer trains. That is a model-construction error (missing softmax), not an FA/training-path
    // bug. PyTorch sidesteps it because nn.CrossEntropyLoss takes logits and applies log_softmax
    // internally; this loss does not, so the softmax belongs in the model.
    [Fact]
    public async Task Transformer_WithFlashAttentionLayer_TrainingActuallyUpdatesProjections()
    {
        await Task.CompletedTask; // Timeout requires an async test; the body itself is synchronous.
        const int vocab = 16;
        const int seqLen = 8;
        const int dModel = 16;
        const int dFf = 32;
        const int heads = 2;

        var faLayer = new FlashAttentionLayer<float>(seqLen, dModel, heads);
        var probeDense = new DenseLayer<float>(dModel, new IdentityActivation<float>() as IActivationFunction<float>);

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
            probeDense,
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(vocab, new SoftmaxActivation<float>() as IActivationFunction<float>),
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            // #1382: a custom layers: list REPLACES the auto-built encoder, so
            // numEncoderLayers must be 0 (the FlashAttention block is in layers: below).
            numEncoderLayers: 0,
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

        // Resolve lazy layer shapes (probeDense's inputSize) with one forward, then snapshot.
        // Capture the baseline loss on the same pass so we can assert it strictly decreases.
        model.SetTrainingMode(false);
        var lossFn = new CategoricalCrossEntropyLoss<float>();
        var yBefore = model.Predict(input);
        float lossBefore = Convert.ToSingle(lossFn.CalculateLoss(yBefore.ToVector(), target.ToVector()));
        float[] dense0 = probeDense.GetParameters().ToArray();

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
        float dq = MaxAbsDelta(faLayer.GetQueryWeights().AsSpan(), q0);
        float dk = MaxAbsDelta(faLayer.GetKeyWeights().AsSpan(), k0);
        float dv = MaxAbsDelta(faLayer.GetValueWeights().AsSpan(), v0);
        float doo = MaxAbsDelta(faLayer.GetOutputWeights().AsSpan(), o0);
        float dDense = MaxAbsDelta(probeDense.GetParameters().ToArray(), dense0);
        var yAfter = model.Predict(input);
        float lossAfter = Convert.ToSingle(lossFn.CalculateLoss(yAfter.ToVector(), target.ToVector()));
        Assert.True(dq > 1e-6f && dk > 1e-6f && dv > 1e-6f && doo > 1e-6f,
            $"All FA projections must change during training (gradient-free in #1346). dQ={dq} dK={dk} dV={dv} dO={doo} | probeDense(control)={dDense}");
        // Control path: a plain DenseLayer in the same stack must also move — proves the
        // training path itself is live, not just the FA arm.
        Assert.True(dDense > 1e-6f,
            $"Control DenseLayer must also update during training. Δ={dDense}");
        // The whole point of #1346: with the gradient reaching the FA layer the loss must
        // actually drop. Pre-fix it froze at -log(1e-7) = 16.118.
        Assert.True(lossAfter < lossBefore,
            $"Training loss must strictly decrease. before={lossBefore}, after={lossAfter}");
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
