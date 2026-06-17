using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using Xunit;

namespace AiDotNet.Tests.Onnx;

/// <summary>
/// Verifies the default behavior of <see cref="LayerBase{T}.ConvertToOnnx"/>:
/// any layer type that has NOT been given an explicit override must throw
/// <see cref="OnnxExportUnsupportedException"/> with the layer's type name
/// so callers get a clear error rather than a silent broken ONNX graph.
///
/// Uses <see cref="ConvolutionalLayer{T}"/> as the unsupported layer because
/// it is intentionally outside the v0.1 ONNX export scope (Dense + activations
/// + BatchNorm + Dropout only). When Conv export lands in a later PR, this
/// test should be retargeted to a layer type that still lacks an override.
/// </summary>
public class OnnxLayerBaseExportTests
{
    [Fact]
    public void ConvertToOnnx_DefaultImplementation_ThrowsOnnxExportUnsupportedException()
    {
        var layer = new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3);
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        var inputs = new OnnxLayerInputs("input_0");

        var ex = Assert.Throws<OnnxExportUnsupportedException>(
            () => layer.ConvertToOnnx(builder, inputs));

        Assert.Equal("ConvolutionalLayer`1", ex.ComponentTypeName);
        Assert.Contains("ConvolutionalLayer`1", ex.Message);
    }

    [Fact]
    public void ConvertToOnnx_DefaultImplementation_MessagePointsAtFollowUpAction()
    {
        var layer = new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3);
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        var inputs = new OnnxLayerInputs("input_0");

        var ex = Assert.Throws<OnnxExportUnsupportedException>(
            () => layer.ConvertToOnnx(builder, inputs));

        // The error message must give the developer a useful next step — either
        // "add the override" or "fall back to the legacy exporter".
        Assert.Contains("ConvertToOnnx override", ex.Message);
    }
}
