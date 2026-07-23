using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AdnTensor = AiDotNet.Tensors.LinearAlgebra.Tensor<float>;
#if !NET471
using Microsoft.ML.OnnxRuntime;
using OrtDenseTensor = Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>;
#endif
using Xunit;

namespace AiDotNet.Tests.Onnx;

/// <summary>
/// Verifies ActivationLayer.ConvertToOnnx emits the right op for each
/// v0.1-supported activation, and that the emitted ONNX produces numerically
/// equivalent output to AiDotNet's Forward via ORT round-trip.
/// </summary>
public class OnnxActivationLayerExportTests
{
    [Theory]
    [InlineData(typeof(ReLUActivation<float>), "Relu")]
    [InlineData(typeof(SigmoidActivation<float>), "Sigmoid")]
    [InlineData(typeof(TanhActivation<float>), "Tanh")]
    [InlineData(typeof(SoftmaxActivation<float>), "Softmax")]
    public void ConvertToOnnx_EmittsCorrectOpForEachSupportedActivation(System.Type activationType, string expectedOp)
    {
        var activation = (AiDotNet.Interfaces.IActivationFunction<float>)System.Activator.CreateInstance(activationType)!;
        var layer = new ActivationLayer<float>(activation);

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 4 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 4 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);

        // Spot-check the bytes contain the expected ONNX op type name.
        var bytes = ms.ToArray();
        var bytesAsText = System.Text.Encoding.UTF8.GetString(bytes);
        Assert.Contains(expectedOp, bytesAsText);
    }

    [Fact]
    public void ConvertToOnnx_IdentityActivation_ReturnsInputNameUnchanged()
    {
        var layer = new ActivationLayer<float>((AiDotNet.Interfaces.IActivationFunction<float>)new IdentityActivation<float>());
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));

        Assert.Equal("input", outputs.Primary);
    }

    [Fact]
    public void ConvertToOnnx_UnsupportedActivation_Throws()
    {
        var layer = new ActivationLayer<float>((AiDotNet.Interfaces.IActivationFunction<float>)new ELUActivation<float>());
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());

        var ex = Assert.Throws<OnnxExportUnsupportedException>(
            () => layer.ConvertToOnnx(builder, new OnnxLayerInputs("input")));

        Assert.Contains("ELU", ex.Message);
    }

#if !NET471
    [Theory]
    [InlineData(typeof(ReLUActivation<float>))]
    [InlineData(typeof(SigmoidActivation<float>))]
    [InlineData(typeof(TanhActivation<float>))]
    public void ActivationLayer_OnnxRoundTrip_MatchesAiDotNetOutputWithinTolerance(System.Type activationType)
    {
        var activation = (AiDotNet.Interfaces.IActivationFunction<float>)System.Activator.CreateInstance(activationType)!;
        var layer = new ActivationLayer<float>(activation);

        // Deterministic input crossing zero so ReLU, Sigmoid, Tanh all stress
        // their non-trivial regions.
        var sampleInput = new float[] { -1.5f, -0.5f, 0.5f, 1.5f };
        var inputTensor = new AdnTensor(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) inputTensor[0, i] = sampleInput[i];

        var aidotnetFlat = layer.Forward(inputTensor).ToArray();

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 4 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 4 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);
        var onnxBytes = ms.ToArray();

        using var session = new InferenceSession(onnxBytes);
        var ortInput = new OrtDenseTensor(sampleInput, new[] { 1, 4 });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input", ortInput),
        });
        var ortFlat = results.First().AsTensor<float>().ToArray();

        Assert.Equal(aidotnetFlat.Length, ortFlat.Length);
        for (int i = 0; i < aidotnetFlat.Length; i++)
        {
            Assert.True(
                System.Math.Abs(aidotnetFlat[i] - ortFlat[i]) < 1e-5f,
                $"{activationType.Name}[{i}]: AiDotNet={aidotnetFlat[i]} vs ONNX={ortFlat[i]}");
        }
    }
#endif
}
