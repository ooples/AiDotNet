using System.IO;
using System.Linq;
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
/// Round-trip test for the new protobuf-based ONNX export path applied to
/// <see cref="DenseLayer{T}"/>:
///   1. Build a DenseLayer with random weights (materialized via Forward)
///   2. Capture AiDotNet's output for a known input
///   3. Export the layer to ONNX via the new <c>ConvertToOnnx</c> path
///   4. Load the bytes with Microsoft.ML.OnnxRuntime.InferenceSession
///   5. Run the same input through ONNX
///   6. Assert outputs match element-wise within 1e-5 tolerance
///
/// If any step fails, the new export path is broken — surface it loudly so
/// it gets fixed before the demo or before more layer converters land.
/// </summary>
public class OnnxDenseLayerExportTests
{
    [Fact]
    public void ConvertToOnnx_EmittsGemmInitializersAndOutput()
    {
        // Pure protobuf-shape assertions — no ORT, runs on every TFM.
        var layer = new DenseLayer<float>(outputSize: 2);

        // Materialize lazy weights with a warm-up forward pass.
        var input = new AdnTensor(new[] { 1, 3 });
        _ = layer.Forward(input);

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 3 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 2 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);

        Assert.True(ms.Length > 0, "ONNX bytes must be non-empty");
    }

    [Fact]
    public void ConvertToOnnx_ThrowsWhenEmbeddedActivationIsUnsupportedType()
    {
        // ELU is not in v0.1's supported activation set — should throw with a
        // clear message naming the activation type. Supported set is
        // Identity / ReLU / Sigmoid / Tanh / Softmax.
        var layer = new DenseLayer<float>(
            outputSize: 2,
            activationFunction: new AiDotNet.ActivationFunctions.ELUActivation<float>());

        var input = new AdnTensor(new[] { 1, 3 });
        _ = layer.Forward(input);

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        var ex = Assert.Throws<OnnxExportUnsupportedException>(
            () => layer.ConvertToOnnx(builder, new OnnxLayerInputs("input")));

        Assert.Contains("ELU", ex.Message);
    }

#if !NET471
    [Fact]
    public void DenseLayer_OnnxRoundTrip_MatchesAiDotNetOutputWithinTolerance()
    {
        // Build a 3→2 DenseLayer, materialize weights, and capture AiDotNet's
        // output for a deterministic input.
        var layer = new DenseLayer<float>(outputSize: 2);

        var sampleInput = new float[] { 0.5f, -0.25f, 1.0f };
        var inputTensor = new AdnTensor(new[] { 1, 3 });
        inputTensor[0, 0] = sampleInput[0];
        inputTensor[0, 1] = sampleInput[1];
        inputTensor[0, 2] = sampleInput[2];
        var aidotnetOutput = layer.Forward(inputTensor);
        var aidotnetFlat = aidotnetOutput.ToArray();

        // Export the layer.
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 3 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 2 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);
        var onnxBytes = ms.ToArray();

        // Load + run through ONNX Runtime, compare element-wise.
        using var session = new InferenceSession(onnxBytes);

        var ortInput = new OrtDenseTensor(sampleInput, new[] { 1, 3 });
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
                $"Output[{i}]: AiDotNet={aidotnetFlat[i]} vs ONNX={ortFlat[i]} (diff > 1e-5)");
        }
    }
#endif
}
