using System;
using System.Collections.Generic;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.Interfaces;
using AiDotNet.LoRA;
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
    // float.IsFinite is unavailable on net471 (the test project multi-targets).
    private static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

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
            Assert.True(IsFinite(a[i]));
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
            Assert.True(IsFinite(f[i]), $"prediction element {i} is not finite");
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

    /// <summary>
    /// Weight-only quantization must reach the sublayers hosted inside composite
    /// blocks: the encoder block's attention AND both FFN denses, and the decoder
    /// block's SELF-attention + FFNs. The decoder's CROSS-attention must stay a
    /// plain MultiHeadAttentionLayer — QuantizedAttentionLayer is single-input by
    /// design and would silently degrade true cross-attention to self-attention.
    /// </summary>
    [Fact]
    public void WeightOnlyQuantization_ReachesBlockHostedFfnAndDecoderSelfAttention()
    {
        AiDotNetEngine.ResetToCpu();
        var model = CreateDefaultTransformer(numEncoder: 1, numDecoder: 1);

        var input = RandTensor(new[] { 2, 4, 8 }, seed: 21);
        var baseline = model.Predict(input);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableWeightOnlyQuantization = true
        };
        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: true);

        Assert.True(anyApplied, "Quantization sweep reported nothing applied on a default block-layout transformer.");

        TransformerEncoderBlock<float>? encBlock = null;
        TransformerDecoderBlock<float>? decBlock = null;
        foreach (var layer in optimized.Layers)
        {
            encBlock ??= layer as TransformerEncoderBlock<float>;
            decBlock ??= layer as TransformerDecoderBlock<float>;
        }
        Assert.NotNull(encBlock);
        Assert.NotNull(decBlock);

        Assert.Contains("QuantizedAttentionLayer", encBlock!.AttentionLayer.GetType().Name);
        Assert.Contains("QuantizedDenseLayer", encBlock.FfnUpLayer.GetType().Name);
        Assert.Contains("QuantizedDenseLayer", encBlock.FfnDownLayer.GetType().Name);

        Assert.Contains("QuantizedAttentionLayer", decBlock!.SelfAttentionLayer.GetType().Name);
        Assert.IsType<MultiHeadAttentionLayer<float>>(decBlock.CrossAttentionLayer);
        Assert.Contains("QuantizedDenseLayer", decBlock.FfnUpLayer.GetType().Name);
        Assert.Contains("QuantizedDenseLayer", decBlock.FfnDownLayer.GetType().Name);

        // The original (un-cloned) model keeps its FP sublayers.
        foreach (var layer in model.Layers)
        {
            if (layer is TransformerEncoderBlock<float> originalEnc)
            {
                Assert.IsType<DenseLayer<float>>(originalEnc.FfnUpLayer);
                Assert.IsType<DenseLayer<float>>(originalEnc.FfnDownLayer);
            }
        }

        var y = optimized.Predict(input);
        Assert.Equal(baseline.Shape, y.Shape);
        var baseData = baseline.GetDataArray();
        var quantData = y.GetDataArray();
        for (int i = 0; i < quantData.Length; i++)
        {
            Assert.True(IsFinite(quantData[i]), $"quantized prediction element {i} is not finite");
            Assert.True(Math.Abs(baseData[i] - quantData[i]) < 2.5e-1f,
                $"Quantized output diverged at {i}: {baseData[i]} vs {quantData[i]}");
        }
    }

    /// <summary>
    /// ApplyLoRA must recurse into composite blocks: the encoder block's attention
    /// and FFN sublayers get LoRA adapters installed through the block's replace
    /// hooks (before the fix a default-built transformer got ZERO adapters because
    /// the per-layer loop only sees the block, which was not a whitelist type).
    /// </summary>
    [Fact]
    public void ApplyLoRA_RecursesIntoEncoderBlockSublayers()
    {
        AiDotNetEngine.ResetToCpu();
        var block = new TransformerEncoderBlock<float>(hiddenSize: 8, numHeads: 2, ffnDim: 16, dropoutRate: 0.0);
        block.SetTrainingMode(false);
        // Resolve lazy sublayer shapes (mirrors the warmup forward AiModelBuilder runs).
        _ = block.Forward(RandTensor(new[] { 1, 4, 8 }, seed: 23));

        var config = new DefaultLoRAConfiguration<float>(rank: 2);
        var result = config.ApplyLoRA(block);

        Assert.Same(block, result);
        Assert.IsAssignableFrom<ILoRAAdapter<float>>(block.AttentionLayer);
        Assert.IsAssignableFrom<ILoRAAdapter<float>>(block.FfnUpLayer);
        Assert.IsAssignableFrom<ILoRAAdapter<float>>(block.FfnDownLayer);
        Assert.True(config.IsLoRATarget(block), "Blocks must count as LoRA targets so the warmup pre-scan gates on them.");

        var output = block.Forward(RandTensor(new[] { 1, 4, 8 }, seed: 24));
        foreach (var v in output.GetDataArray())
            Assert.True(IsFinite(v), "Adapted encoder block produced a non-finite output.");
    }

    /// <summary>
    /// Decoder-block LoRA recursion: self-attention + FFNs adapted; CROSS-attention
    /// left untouched (single-input adapters would silently degrade true
    /// cross-attention to self-attention), and the two-input forward still consumes
    /// the encoder output after adaptation.
    /// </summary>
    [Fact]
    public void ApplyLoRA_RecursesIntoDecoderBlock_SkipsCrossAttention()
    {
        AiDotNetEngine.ResetToCpu();
        var block = new TransformerDecoderBlock<float>(hiddenSize: 8, numHeads: 2, ffnDim: 16, dropoutRate: 0.0);
        block.SetTrainingMode(false);
        _ = block.Forward(RandTensor(new[] { 1, 4, 8 }, seed: 25), RandTensor(new[] { 1, 4, 8 }, seed: 26));

        var config = new DefaultLoRAConfiguration<float>(rank: 2);
        var result = config.ApplyLoRA(block);

        Assert.Same(block, result);
        Assert.IsAssignableFrom<ILoRAAdapter<float>>(block.SelfAttentionLayer);
        Assert.IsType<MultiHeadAttentionLayer<float>>(block.CrossAttentionLayer);
        Assert.IsAssignableFrom<ILoRAAdapter<float>>(block.FfnUpLayer);
        Assert.IsAssignableFrom<ILoRAAdapter<float>>(block.FfnDownLayer);

        // Cross-attention must still consume the encoder output post-adaptation.
        var stream = RandTensor(new[] { 1, 4, 8 }, seed: 27);
        var outA = block.Forward(stream, RandTensor(new[] { 1, 4, 8 }, seed: 28));
        var outB = block.Forward(stream, RandTensor(new[] { 1, 4, 8 }, seed: 29));
        bool anyDifferent = false;
        var a = outA.GetDataArray();
        var b = outB.GetDataArray();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(IsFinite(a[i]));
            if (Math.Abs(a[i] - b[i]) > 1e-6f) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent, "Adapted decoder block no longer consumes the encoder output.");
    }

    /// <summary>
    /// Gradient checkpointing on the decoder-aware training path must produce the
    /// same training step as the un-checkpointed walk. The hazard this guards: the
    /// frozen encoder output crossing a checkpoint-segment boundary as a closure
    /// side-channel would silently DROP the cross-attention gradient contribution
    /// during the recompute VJP — the region segmentation in
    /// Transformer.RunCheckpointedLayerWalk exists precisely to prevent that.
    /// </summary>
    [Fact]
    public void EncoderDecoderTransformer_CheckpointedTraining_MatchesUncheckpointed()
    {
        AiDotNetEngine.ResetToCpu();
        var input = RandTensor(new[] { 2, 4, 8 }, seed: 31);
        var target = RandTensor(new[] { 2, 4, 8 }, seed: 32);

        var plain = CreateDefaultTransformer(numEncoder: 1, numDecoder: 1);
        var checkpointed = CreateDefaultTransformer(numEncoder: 1, numDecoder: 1);

        // Materialize lazy layers, then mirror parameters so both models start identical.
        _ = plain.Predict(input);
        _ = checkpointed.Predict(input);
        var sharedParams = plain.GetParameters();
        checkpointed.SetParameters(sharedParams);
        checkpointed.SetGradientCheckpointingSegmentSize(2);

        plain.Train(input, target);
        checkpointed.Train(input, target);

        var plainParams = plain.GetParameters();
        var checkpointedParams = checkpointed.GetParameters();
        Assert.Equal(plainParams.Length, checkpointedParams.Length);

        bool anyChanged = false;
        for (int i = 0; i < plainParams.Length; i++)
        {
            double p = Convert.ToDouble(plainParams[i]);
            double c = Convert.ToDouble(checkpointedParams[i]);
            double s = Convert.ToDouble(sharedParams[i]);
            if (Math.Abs(p - s) > 1e-12) anyChanged = true;
            Assert.True(Math.Abs(p - c) < 1e-5,
                $"Checkpointed training diverged from un-checkpointed at parameter {i}: {p} vs {c} (started at {s}).");
        }
        Assert.True(anyChanged, "Neither model updated any parameter — the parity comparison is vacuous.");
    }
}
