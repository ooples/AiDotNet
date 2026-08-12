using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Tests for the named input/output port infrastructure (#1058).
/// </summary>
public class LayerPortTests
{
    [Fact(Timeout = 120000)]
    public async Task EmbeddingLayer_GeneratedPortPublishesIndexRangeAndRole()
    {
        await Task.Yield();
        var layer = new EmbeddingLayer<double>(31, 8);

        var port = Assert.Single(layer.InputPorts);
        Assert.Equal("input", port.Name);
        Assert.Equal(TensorPortRole.TokenIds, port.Role);
        Assert.Equal(LayerInputDomainKind.IntegerIndices, port.ValueDomain.Kind);
        Assert.Equal(31, port.ValueDomain.MaxExclusive);
        Assert.Equal(LayerInputDomainKind.Continuous, Assert.Single(layer.OutputPorts).ValueDomain.Kind);
    }

    [Fact(Timeout = 120000)]
    public async Task ValueDomainValidator_RejectsContinuousOutputFeedingLookup()
    {
        await Task.Yield();
        ILayer<double>[] layers =
        [
            new DenseLayer<double>(8),
            new EmbeddingLayer<double>(31, 8)
        ];

        var mismatch = Assert.Single(LayerContractValidator.ValidateValueDomains(layers));
        Assert.StartsWith("ADNPORT004", mismatch.Message);
        Assert.Contains("generated composite or named graph branches", mismatch.Message);
    }

    [Fact(Timeout = 120000)]
    public async Task ValueDomainValidator_AcceptsLookupFeedingContinuousLayer()
    {
        await Task.Yield();
        ILayer<double>[] layers =
        [
            new EmbeddingLayer<double>(31, 8),
            new DenseLayer<double>(8)
        ];

        Assert.Empty(LayerContractValidator.ValidateValueDomains(layers));
    }

    [Fact(Timeout = 120000)]
    public async Task EncoderDecoderComposite_RegistersChildrenAndGeneratedPorts()
    {
        await Task.Yield();
        var composite = new TokenConditionedDecoderLayer<double>(
            encoderLayers: [new EmbeddingLayer<double>(31, 8)],
            decoderEmbedding: new EmbeddingLayer<double>(17, 8),
            decoderLayers: [new DenseLayer<double>(8)],
            outputLayer: new DenseLayer<double>(17),
            encoderVocabularySize: 31,
            decoderVocabularySize: 17,
            maximumDecoderLength: 12);

        Assert.Equal(4, composite.GetSubLayers().Count);
        Assert.Equal(TensorPortRole.EncoderInput, composite.InputPorts[0].Role);
        Assert.Equal(TensorPortRole.DecoderIds, composite.InputPorts[1].Role);
        Assert.False(composite.InputPorts[1].Required);
        Assert.Equal(31, composite.InputPorts[0].ValueDomain.MaxExclusive);
        Assert.Equal(17, composite.InputPorts[1].ValueDomain.MaxExclusive);
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_HasSingleInputPort()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(8);
        Assert.Single(layer.InputPorts);
        Assert.Equal("input", layer.InputPorts[0].Name);
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_HasSingleOutputPort()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(8);
        Assert.Single(layer.OutputPorts);
        Assert.Equal("output", layer.OutputPorts[0].Name);
    }

    [Fact(Timeout = 120000)]
    public async Task DenseLayer_MultiInputForward_DelegatesToSingleInput()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(2);
        var input = new Tensor<double>([1, 4]);
        for (int i = 0; i < 4; i++) input[0, i] = (i + 1) * 0.1;

        var singleResult = layer.Forward(input);

        // Reset layer state
        layer.ResetState();

        var multiResult = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input"] = input
        });

        Assert.Equal(singleResult.Length, multiResult.Length);
        for (int i = 0; i < singleResult.Length; i++)
        {
            Assert.Equal(singleResult[i], multiResult[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AddLayer_HasTwoInputPorts()
    {
        await Task.Yield();
        var layer = new AddLayer<double>(new int[][] { new[] { 4 }, new[] { 4 } }, (IActivationFunction<double>?)null);
        Assert.Equal(2, layer.InputPorts.Count);
        Assert.Equal("input_0", layer.InputPorts[0].Name);
        Assert.Equal("input_1", layer.InputPorts[1].Name);
    }

    [Fact(Timeout = 120000)]
    public async Task AddLayer_MultiInputForward_AddsCorrectly()
    {
        await Task.Yield();
        var layer = new AddLayer<double>(new int[][] { new[] { 3 }, new[] { 3 } }, (IActivationFunction<double>?)null);
        var a = new Tensor<double>([3], new Vector<double>([1.0, 2.0, 3.0]));
        var b = new Tensor<double>([3], new Vector<double>([4.0, 5.0, 6.0]));

        var result = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input_0"] = a,
            ["input_1"] = b
        });

        Assert.Equal(5.0, result[0], 10);
        Assert.Equal(7.0, result[1], 10);
        Assert.Equal(9.0, result[2], 10);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionResBlock_HasTimeEmbedPort_WhenConfigured()
    {
        await Task.Yield();
        var block = new AiDotNet.Diffusion.NoisePredictors.DiffusionResBlock<double>(
            inChannels: 4, outChannels: 4, spatialSize: 8, timeEmbedDim: 64);

        Assert.Equal(2, block.InputPorts.Count);
        Assert.Equal("input", block.InputPorts[0].Name);
        Assert.Equal("time_embed", block.InputPorts[1].Name);
        Assert.Equal(64, block.InputPorts[1].Shape[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionResBlock_SinglePort_WhenNoTimeEmbed()
    {
        await Task.Yield();
        var block = new AiDotNet.Diffusion.NoisePredictors.DiffusionResBlock<double>(
            inChannels: 4, outChannels: 4, spatialSize: 8, timeEmbedDim: 0);

        Assert.Single(block.InputPorts);
        Assert.Equal("input", block.InputPorts[0].Name);
    }

    [Fact(Timeout = 120000)]
    public async Task LayerPort_Record_HasCorrectProperties()
    {
        await Task.Yield();
        var port = new LayerPort("query", [8, 64], Required: true);

        Assert.Equal("query", port.Name);
        Assert.Equal(2, port.Shape.Count);
        Assert.Equal(8, port.Shape[0]);
        Assert.Equal(64, port.Shape[1]);
        Assert.True(port.Required);
    }

    [Fact(Timeout = 120000)]
    public async Task LayerPort_OptionalPort()
    {
        await Task.Yield();
        var port = new LayerPort("mask", [8, 8], Required: false);
        Assert.False(port.Required);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionResBlock_MultiInputForward_IncludesTimeConditioning()
    {
        await Task.Yield();
        int channels = 4;
        int spatial = 8;
        int timeEmbedDim = 16;

        var block = new AiDotNet.Diffusion.NoisePredictors.DiffusionResBlock<double>(
            inChannels: channels, outChannels: channels, spatialSize: spatial, timeEmbedDim: timeEmbedDim);

        var input = new Tensor<double>([1, channels, spatial, spatial]);
        var timeEmbed = new Tensor<double>([1, timeEmbedDim]);
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 0.5;
        for (int i = 0; i < timeEmbed.Length; i++) timeEmbed[i] = rng.NextDouble() * 2.0;

        var outputDirect = block.Forward(input, timeEmbed);
        block.ResetState();

        var outputWithTime = block.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input"] = input,
            ["time_embed"] = timeEmbed
        });

        // The multi-input Forward via dict should produce the same result as direct Forward(input, timeEmbed)
        for (int i = 0; i < Math.Min(outputDirect.Length, outputWithTime.Length); i++)
        {
            Assert.Equal(outputDirect[i], outputWithTime[i], 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LayerBase_ForwardGpu_ThrowsOnEmptyArgs()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(2);
        // 0 args → throws
        Assert.ThrowsAny<Exception>(() => layer.ForwardGpu());
    }

    // BackwardGpuMulti test removed — Backward deleted in tape-based autodiff migration

    [Fact(Timeout = 120000)]
    public async Task SingleInputLayer_MultiInputForward_IgnoresExtraKeys()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(2);
        var input = new Tensor<double>([1, 4]);
        for (int i = 0; i < 4; i++) input[0, i] = (i + 1) * 0.1;

        // Pass extra key that doesn't match any port — should be ignored
        var result = layer.Forward(new Dictionary<string, Tensor<double>>
        {
            ["input"] = input,
            ["extra_data"] = new Tensor<double>([3])
        });

        Assert.True(result.Length > 0);
    }
}
