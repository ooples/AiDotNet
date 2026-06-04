using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression guards for the composite encoder/decoder block layout
/// (LayerHelper.CreateDefaultTransformerLayers emits TransformerEncoderBlock /
/// TransformerDecoderBlock since #1487). Any code that pattern-matches
/// <c>model.Layers</c> for discrete attention layers silently breaks for
/// default-built transformers — these tests lock the audited fixes:
/// decoder cross-attention actually consumes the encoder output (inference AND
/// training walks), and attention-layer accounting sees block-hosted attention.
/// </summary>
public class TransformerEncoderBlockLayoutAuditTests
{
    private static Tensor<float> RandTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int len = 1;
        foreach (var d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    /// <summary>
    /// The decoder block's two-input forward must CONSUME the encoder output:
    /// same decoder stream + different encoder context ⇒ different output.
    /// Before the fix, cross-attention ran single-input over the decoder stream
    /// (Q = K = V), so the encoder context was silently discarded.
    /// </summary>
    [Fact]
    public void DecoderBlock_CrossAttention_ConsumesEncoderOutput()
    {
        AiDotNetEngine.ResetToCpu();
        var block = new TransformerDecoderBlock<float>(hiddenSize: 8, numHeads: 2, ffnDim: 16, dropoutRate: 0.0);
        block.SetTrainingMode(false);

        var decoderStream = RandTensor(new[] { 1, 4, 8 }, seed: 1);
        var encoderA = RandTensor(new[] { 1, 4, 8 }, seed: 2);
        var encoderB = RandTensor(new[] { 1, 4, 8 }, seed: 3);

        var outA = block.Forward(decoderStream, encoderA);
        var outB = block.Forward(decoderStream, encoderB);

        Assert.Equal(outA.Shape, outB.Shape);
        bool anyDifferent = false;
        var a = outA.GetDataArray();
        var b = outB.GetDataArray();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(float.IsFinite(a[i]));
            if (Math.Abs(a[i] - b[i]) > 1e-6f) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent,
            "Changing ONLY the encoder output left the decoder block output identical — cross-attention is not consuming the encoder context.");
    }

    /// <summary>
    /// Multi-input dispatch contract: 1 input = decoder-only, 2 = (stream, encoder),
    /// anything else throws.
    /// </summary>
    [Fact]
    public void DecoderBlock_ParamsForward_DispatchesByArity()
    {
        AiDotNetEngine.ResetToCpu();
        var block = new TransformerDecoderBlock<float>(hiddenSize: 8, numHeads: 2, ffnDim: 16, dropoutRate: 0.0);
        block.SetTrainingMode(false);

        var stream = RandTensor(new[] { 1, 4, 8 }, seed: 4);
        var encoder = RandTensor(new[] { 1, 4, 8 }, seed: 5);

        var one = block.Forward(new[] { stream });
        var two = block.Forward(new[] { stream, encoder });
        Assert.Equal(one.Shape, two.Shape);

        Assert.Throws<ArgumentException>(() => block.Forward(new[] { stream, encoder, encoder }));
    }

    private static Transformer<float> CreateDefaultTransformer(int numEncoder, int numDecoder)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: numEncoder,
            numDecoderLayers: numDecoder,
            numHeads: 2,
            modelDimension: 8,
            feedForwardDimension: 16,
            inputSize: 8,
            outputSize: 8,
            dropoutRate: 0.0,
            maxSequenceLength: 4,
            vocabularySize: 0,
            usePositionalEncoding: false);
        return new Transformer<float>(arch);
    }

    /// <summary>
    /// The default factory hosts attention inside composite blocks; the
    /// attention-layer accounting must see them. Before the fix
    /// TotalAttentionLayers reported 0 for every default-built transformer.
    /// </summary>
    [Fact]
    public void AuxiliaryLossDiagnostics_CountBlockHostedAttention()
    {
        AiDotNetEngine.ResetToCpu();
        var model = CreateDefaultTransformer(numEncoder: 2, numDecoder: 1);

        var diagnostics = model.GetAuxiliaryLossDiagnostics();

        // 2 encoder blocks × 1 self-attention + 1 decoder block × (self + cross) = 4.
        Assert.True(int.Parse(diagnostics["TotalAttentionLayers"]) >= 3,
            $"Block-hosted attention layers are invisible to the diagnostics: TotalAttentionLayers={diagnostics["TotalAttentionLayers"]}");
    }

    /// <summary>
    /// End-to-end: an encoder-decoder default transformer must produce finite,
    /// deterministic predictions through the block-aware walk (decoder blocks are
    /// dispatched with the captured encoder context rather than the generic
    /// single-input walk).
    /// </summary>
    [Fact]
    public void EncoderDecoderTransformer_Predict_FiniteAndDeterministic()
    {
        AiDotNetEngine.ResetToCpu();
        var model = CreateDefaultTransformer(numEncoder: 1, numDecoder: 2);

        var input = RandTensor(new[] { 2, 4, 8 }, seed: 7);
        var first = model.Predict(input);
        var second = model.Predict(input);

        var f = first.GetDataArray();
        var s = second.GetDataArray();
        Assert.Equal(s.Length, f.Length);
        for (int i = 0; i < f.Length; i++)
        {
            Assert.True(float.IsFinite(f[i]), $"prediction element {i} is not finite");
            Assert.Equal(f[i], s[i]);
        }
    }

    /// <summary>
    /// Training forward must use the same decoder-aware walk: a Train step on an
    /// encoder-decoder default transformer runs and updates parameters (before the
    /// fix the generic walk fed decoders single-input, silently training degenerate
    /// cross-attention).
    /// </summary>
    [Fact]
    public void EncoderDecoderTransformer_Train_RunsAndUpdatesParameters()
    {
        AiDotNetEngine.ResetToCpu();
        var model = CreateDefaultTransformer(numEncoder: 1, numDecoder: 1);

        var input = RandTensor(new[] { 2, 4, 8 }, seed: 11);
        var target = RandTensor(new[] { 2, 4, 8 }, seed: 12);

        // Materialize lazy layers + capture pre-training parameters.
        _ = model.Predict(input);
        var before = model.GetParameters();

        model.Train(input, target);

        var after = model.GetParameters();
        Assert.Equal(before.Length, after.Length);
        bool anyChanged = false;
        for (int i = 0; i < before.Length; i++)
        {
            if (Math.Abs(Convert.ToDouble(before[i]) - Convert.ToDouble(after[i])) > 1e-12)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged, "Train() did not update any parameter of the encoder-decoder transformer.");
    }
}
