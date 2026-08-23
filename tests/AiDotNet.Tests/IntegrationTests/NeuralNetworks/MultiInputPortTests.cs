using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for the named input/output port system on multi-input layers.
/// Validates that layers correctly declare ports, accept named inputs, and reject
/// missing required inputs.
/// </summary>
public class MultiInputPortTests
{
    private const double Tolerance = 1e-6;

    #region CrossAttentionLayer - Port Declaration

    [Fact(Timeout = 120000)]
    public async Task CrossAttention_InputPorts_DeclaresQueryAndContext()
    {
        var layer = new CrossAttentionLayer<double>(queryDim: 8, contextDim: 16, headCount: 2);

        Assert.Equal(2, layer.InputPorts.Count);
        Assert.Equal("query", layer.InputPorts[0].Name);
        Assert.True(layer.InputPorts[0].Required);
        Assert.Equal("context", layer.InputPorts[1].Name);
        Assert.False(layer.InputPorts[1].Required);
    }

    [Fact(Timeout = 120000)]
    public async Task CrossAttention_OutputPorts_DefaultsSingleOutput()
    {
        var layer = new CrossAttentionLayer<double>(queryDim: 8, contextDim: 16, headCount: 2);

        Assert.Single(layer.OutputPorts);
        Assert.Equal("output", layer.OutputPorts[0].Name);
    }

    #endregion

    #region CrossAttentionLayer - Named Forward

    [Fact(Timeout = 120000)]
    public async Task CrossAttention_NamedForward_QueryAndContext_ProducesOutput()
    {
        int queryDim = 8, contextDim = 16, headCount = 2, seqLen = 4;
        var layer = new CrossAttentionLayer<double>(queryDim, contextDim, headCount, seqLen);

        var query = Tensor<double>.CreateRandom([1, seqLen, queryDim]);
        var context = Tensor<double>.CreateRandom([1, seqLen, contextDim]);

        var inputs = new Dictionary<string, Tensor<double>>
        {
            ["query"] = query,
            ["context"] = context
        };

        var output = layer.Forward(inputs);

        Assert.NotNull(output);
        Assert.Equal(query.Shape[0], output.Shape[0]); // batch preserved
        Assert.Equal(queryDim, output.Shape[^1]); // output dim matches query dim
    }

    [Fact(Timeout = 120000)]
    public async Task CrossAttention_NamedForward_QueryOnly_FallsBackToSelfAttention()
    {
        int dim = 8, headCount = 2, seqLen = 4;
        var layer = new CrossAttentionLayer<double>(dim, dim, headCount, seqLen);

        var query = Tensor<double>.CreateRandom([1, seqLen, dim]);

        var inputs = new Dictionary<string, Tensor<double>>
        {
            ["query"] = query
        };

        var output = layer.Forward(inputs);

        Assert.NotNull(output);
        Assert.Equal(dim, output.Shape[^1]);
    }

    [Fact(Timeout = 120000)]
    public async Task CrossAttention_NamedForward_MissingQuery_Throws()
    {
        await Task.Yield();
        int dim = 8, headCount = 2;
        var layer = new CrossAttentionLayer<double>(dim, 16, headCount);

        var context = Tensor<double>.CreateRandom([1, 4, 16]);

        var inputs = new Dictionary<string, Tensor<double>>
        {
            ["context"] = context
        };

        Assert.ThrowsAny<ArgumentException>(() => layer.Forward(inputs));
    }

    [Fact(Timeout = 120000)]
    public async Task CrossAttention_NamedForward_MatchesPositionalForward()
    {
        int queryDim = 8, contextDim = 16, headCount = 2, seqLen = 4;
        var layer = new CrossAttentionLayer<double>(queryDim, contextDim, headCount, seqLen);

        var query = Tensor<double>.CreateRandom([1, seqLen, queryDim]);
        var context = Tensor<double>.CreateRandom([1, seqLen, contextDim]);

        // Named forward
        var namedOutput = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["query"] = query,
            ["context"] = context
        });

        // Positional forward
        var positionalOutput = layer.Forward(query, context);

        // Should produce identical results
        Assert.Equal(namedOutput.Length, positionalOutput.Length);
        for (int i = 0; i < namedOutput.Length; i++)
        {
            Assert.Equal(namedOutput.GetFlat(i), positionalOutput.GetFlat(i), Tolerance);
        }
    }

    #endregion

    #region Single-Input Layer - Backward Compatibility

    [Fact(Timeout = 120000)]
    public async Task SingleInputLayer_NamedForward_WithInputKey_Works()
    {
        var layer = new DenseLayer<double>(8);

        var input = Tensor<double>.CreateRandom([1, 4]);
        var inputs = new Dictionary<string, Tensor<double>>
        {
            ["input"] = input
        };

        var output = layer.Forward(inputs);

        Assert.NotNull(output);
        Assert.Equal(8, output.Shape[^1]);
    }

    [Fact(Timeout = 120000)]
    public async Task SingleInputLayer_InputPorts_DefaultsSingleInput()
    {
        var layer = new DenseLayer<double>(8);

        Assert.Single(layer.InputPorts);
        Assert.Equal("input", layer.InputPorts[0].Name);
        Assert.True(layer.InputPorts[0].Required);
    }

    #endregion

    #region MultiHeadAttentionLayer - Named Ports

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttention_InputPorts_DeclaresQueryKeyValue()
    {
        var layer = new MultiHeadAttentionLayer<double>(2, (8) / (2));

        Assert.Equal(3, layer.InputPorts.Count);
        Assert.Equal("query", layer.InputPorts[0].Name);
        Assert.True(layer.InputPorts[0].Required);
        Assert.Equal("key", layer.InputPorts[1].Name);
        Assert.False(layer.InputPorts[1].Required);
        Assert.Equal("value", layer.InputPorts[2].Name);
        Assert.False(layer.InputPorts[2].Required);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttention_NamedForward_QueryOnly_SelfAttention()
    {
        var layer = new MultiHeadAttentionLayer<double>(2, (8) / (2));
        var query = Tensor<double>.CreateRandom([1, 4, 8]);

        var output = layer.Forward(new Dictionary<string, Tensor<double>> { ["query"] = query });

        Assert.NotNull(output);
        Assert.Equal(8, output.Shape[^1]);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttention_NamedForward_QueryKeyValue_CrossAttention()
    {
        var layer = new MultiHeadAttentionLayer<double>(2, (8) / (2));
        var query = Tensor<double>.CreateRandom([1, 4, 8]);
        var key = Tensor<double>.CreateRandom([1, 6, 8]);
        var value = Tensor<double>.CreateRandom([1, 6, 8]);

        var output = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["query"] = query, ["key"] = key, ["value"] = value
        });

        Assert.NotNull(output);
        Assert.Equal(8, output.Shape[^1]);
    }

    #endregion

    #region DecoderLayer - Named Ports

    [Fact(Timeout = 120000)]
    public async Task DecoderLayer_InputPorts_DeclaresDecoderEncoderMask()
    {
        var layer = new DecoderLayer<double>(attentionSize: 8, feedForwardSize: 16, activation: (IActivationFunction<double>?)null);

        // ASSERT ON THE VARIANT, NOT ON THE FLAT UNION. DecoderLayer declares its ports under two
        // [TensorPort] variants -- "default" (decoder_input, encoder_output) and "named"
        // (decoder_input, encoder_output, mask) -- and InputPorts is deliberately the union of
        // both, which InputContractManifest then groups: Variants = InputPorts.GroupBy(p =>
        // p.Variant). So InputPorts.Count is 5 by design, and asserting 3 on it was measuring the
        // wrong surface; it only ever passed while "named" was the sole declared variant.
        //
        // The contract this test is about -- decoder input, encoder memory, optional mask -- is the
        // "named" variant, so resolve it and assert there. That is strictly more specific than
        // before: it pins the port order, requiredness AND the variant they belong to.
        var manifest = layer.GetInputContract();
        var named = Assert.Single(manifest.Variants.Where(variant => variant.Name == "named"));

        Assert.Equal(3, named.Ports.Count);
        Assert.Equal("decoder_input", named.Ports[0].Name);
        Assert.True(named.Ports[0].Required);
        Assert.Equal("encoder_output", named.Ports[1].Name);
        Assert.True(named.Ports[1].Required);
        Assert.Equal("mask", named.Ports[2].Name);
        Assert.False(named.Ports[2].Required);
    }

    #endregion

    #region AddLayer / MultiplyLayer / ConcatenateLayer - Named Ports

    [Fact(Timeout = 120000)]
    public async Task AddLayer_NamedForward_TwoInputs_ProducesOutput()
    {
        var layer = new AddLayer<double>(new[] { new[] { 4 }, new[] { 4 } }, (IActivationFunction<double>?)null);
        var a = Tensor<double>.CreateRandom([1, 4]);
        var b = Tensor<double>.CreateRandom([1, 4]);

        var output = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input_0"] = a, ["input_1"] = b
        });

        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[^1]);
    }

    [Fact(Timeout = 120000)]
    public async Task AddLayer_NamedForward_MissingInput1_Throws()
    {
        await Task.Yield();
        var layer = new AddLayer<double>(new[] { new[] { 4 }, new[] { 4 } }, (IActivationFunction<double>?)null);
        var a = Tensor<double>.CreateRandom([1, 4]);

        Assert.ThrowsAny<ArgumentException>(() =>
            layer.Forward(new Dictionary<string, Tensor<double>> { ["input_0"] = a }));
    }

    [Fact(Timeout = 120000)]
    public async Task MultiplyLayer_NamedForward_TwoInputs_ProducesOutput()
    {
        var layer = new MultiplyLayer<double>(new[] { new[] { 4 }, new[] { 4 } }, (IActivationFunction<double>?)null);
        var a = Tensor<double>.CreateRandom([1, 4]);
        var b = Tensor<double>.CreateRandom([1, 4]);

        var output = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input_0"] = a, ["input_1"] = b
        });

        Assert.NotNull(output);
        Assert.Equal(4, output.Shape[^1]);
    }

    [Fact(Timeout = 120000)]
    public async Task ConcatenateLayer_InputPorts_DeclaresMultipleInputs()
    {
        // axis=0 concat: two [4]-shaped inputs → [8] output
        var layer = new ConcatenateLayer<double>(new[] { new[] { 4 }, new[] { 4 } }, axis: 0, activationFunction: (IActivationFunction<double>?)null);

        Assert.Equal(2, layer.InputPorts.Count);
        Assert.Equal("input_0", layer.InputPorts[0].Name);
        Assert.Equal("input_1", layer.InputPorts[1].Name);
    }

    #endregion
}
