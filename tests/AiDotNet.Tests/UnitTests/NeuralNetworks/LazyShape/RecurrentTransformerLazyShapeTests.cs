using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.LazyShape;

/// <summary>
/// Validates the lazy input-feature contract for recurrent + transformer layers
/// introduced by issue #1212. Each layer's lazy ctor takes only the output-side dims
/// (hiddenSize, numHeads, feedForwardDim) and resolves the input feature dim from
/// <c>input.Shape[^1]</c> on first forward.
/// </summary>
public class RecurrentTransformerLazyShapeTests
{
    [Fact]
    public void LazyLSTM_BeforeForward_IsShapeResolvedIsFalse()
    {
        var lstm = new LSTMLayer<double>(hiddenSize: 32);
        Assert.False(lstm.IsShapeResolved);
    }

    [Fact]
    public void LazyLSTM_InfersInputSize_FromForward()
    {
        var lstm = new LSTMLayer<double>(hiddenSize: 32);
        var input = new Tensor<double>(new[] { 1, 10, 64 });

        var output = lstm.Forward(input);

        Assert.True(lstm.IsShapeResolved);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(32, output.Shape[output.Shape.Length - 1]);
    }

    [Fact]
    public void LazyLSTM_TwoInstances_ResolveDifferentInputSizes()
    {
        var lstm64 = new LSTMLayer<double>(hiddenSize: 32);
        var lstm128 = new LSTMLayer<double>(hiddenSize: 32);

        var out64 = lstm64.Forward(new Tensor<double>(new[] { 1, 10, 64 }));
        var out128 = lstm128.Forward(new Tensor<double>(new[] { 1, 10, 128 }));

        Assert.True(lstm64.IsShapeResolved);
        Assert.True(lstm128.IsShapeResolved);
        Assert.Equal(32, out64.Shape[^1]);
        Assert.Equal(32, out128.Shape[^1]);
    }

    [Fact]
    public void LazyGRU_InfersInputSize_FromForward()
    {
        var gru = new GRULayer<double>(hiddenSize: 24);
        var input = new Tensor<double>(new[] { 1, 8, 48 });

        var output = gru.Forward(input);

        Assert.True(gru.IsShapeResolved);
        Assert.Equal(24, output.Shape[^1]);
    }

    [Fact]
    public void LazyRecurrent_InfersInputSize_FromForward()
    {
        var rnn = new RecurrentLayer<double>(hiddenSize: 16);
        var input = new Tensor<double>(new[] { 1, 5, 32 });

        var output = rnn.Forward(input);

        Assert.True(rnn.IsShapeResolved);
        Assert.Equal(16, output.Shape[^1]);
    }

    [Fact]
    public void LazyTransformerEncoder_InfersModelDim_FromForward()
    {
        var enc = new TransformerEncoderLayer<double>(numHeads: 4, feedForwardDim: 128);
        var input = new Tensor<double>(new[] { 1, 8, 32 });

        var output = enc.Forward(input);

        Assert.True(enc.IsShapeResolved);
        Assert.Equal(input.Shape[^1], output.Shape[^1]);
    }

    [Fact]
    public void LazyTransformerEncoder_RejectsInputDimNotDivisibleByNumHeads()
    {
        var enc = new TransformerEncoderLayer<double>(numHeads: 5, feedForwardDim: 128);
        // 32 % 5 != 0 — should fail fast on first forward.
        var input = new Tensor<double>(new[] { 1, 8, 32 });

        Assert.Throws<ArgumentException>(() => enc.Forward(input));
    }

    [Fact]
    public void LazyTransformerDecoder_InfersModelDim_FromForward()
    {
        var dec = new TransformerDecoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        var input = new Tensor<double>(new[] { 1, 8, 32 });

        var output = dec.Forward(input);

        Assert.True(dec.IsShapeResolved);
        Assert.Equal(input.Shape[^1], output.Shape[^1]);
    }

    [Fact]
    public void LazyTransformerDecoder_RejectsMismatchedEncoderOutput()
    {
        var dec = new TransformerDecoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        var decoderInput = new Tensor<double>(new[] { 1, 8, 32 });
        // encoderOutput has 16 features, decoder has 32 — mismatched.
        var encoderOutput = new Tensor<double>(new[] { 1, 8, 16 });

        Assert.Throws<ArgumentException>(() => dec.Forward(decoderInput, encoderOutput));
    }

    // ---- Pre-resolution access regressions (the lazy ctor must not break public
    // contract surface for callers that touch parameter / state APIs before any
    // Forward has run). ----

    [Fact]
    public void LazyLSTM_BeforeForward_GetParameters_DoesNotThrow()
    {
        var lstm = new LSTMLayer<double>(hiddenSize: 16);
        // Layer has no real weights yet, but parameter collection should still
        // return *something* (typically empty or zero-length) instead of NRE.
        var p = lstm.GetParameters();
        Assert.NotNull(p);
    }

    [Fact]
    public void LazyTransformerEncoder_BeforeForward_ParameterCount_IsZero()
    {
        var enc = new TransformerEncoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        // Sublayers don't exist yet — ParameterCount must be a safe 0, not NRE
        // dereferencing the null sublayer fields.
        Assert.Equal(0, enc.ParameterCount);
    }

    [Fact]
    public void LazyTransformerEncoder_BeforeForward_GetParameters_ReturnsEmpty()
    {
        var enc = new TransformerEncoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        var p = enc.GetParameters();
        Assert.NotNull(p);
        Assert.Equal(0, p.Length);
    }

    [Fact]
    public void LazyTransformerEncoder_BeforeForward_ResetState_DoesNotThrow()
    {
        var enc = new TransformerEncoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        enc.ResetState(); // should be a safe no-op, not an NRE.
    }

    [Fact]
    public void LazyTransformerDecoder_BeforeForward_ParameterCount_IsZero()
    {
        var dec = new TransformerDecoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        Assert.Equal(0, dec.ParameterCount);
    }

    [Fact]
    public void LazyTransformerDecoder_BeforeForward_ClearGradients_DoesNotThrow()
    {
        var dec = new TransformerDecoderLayer<double>(numHeads: 4, feedForwardDim: 64);
        dec.ClearGradients();
    }
}
