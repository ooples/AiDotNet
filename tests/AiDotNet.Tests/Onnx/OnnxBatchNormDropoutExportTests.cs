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
/// Covers ConvertToOnnx for BatchNormalizationLayer (emits BatchNormalization
/// op with 4 initializers + epsilon attribute) and DropoutLayer (emits
/// Identity in inference mode — dropout is a no-op outside training).
/// </summary>
public class OnnxBatchNormDropoutExportTests
{
    [Fact]
    public void BatchNormalization_ConvertToOnnx_EmitsBatchNormalizationOpWithEpsilon()
    {
        var layer = new BatchNormalizationLayer<float>(numFeatures: 3);
        var input = new AdnTensor(new[] { 1, 3 });
        _ = layer.Forward(input); // materialise running stats

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 3 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 3 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);
        var bytes = ms.ToArray();
        Assert.True(bytes.Length > 0);

        // Op name appears in the protobuf as a length-prefixed string. Spot check.
        var text = System.Text.Encoding.UTF8.GetString(bytes);
        Assert.Contains("BatchNormalization", text);
        Assert.Contains("epsilon", text);
    }

    [Fact]
    public void Dropout_ConvertToOnnx_EmitsIdentityOp()
    {
        var layer = new DropoutLayer<float>(dropoutRate: 0.3);
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));

        // Identity emission should produce a new output name (not the same as input).
        Assert.NotEqual("input", outputs.Primary);

        using var ms = new MemoryStream();
        builder.AddFloatInput("input", new[] { -1, 4 });
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 4 });
        builder.WriteTo(ms);

        var text = System.Text.Encoding.UTF8.GetString(ms.ToArray());
        Assert.Contains("Identity", text);
    }

#if !NET471
    [Fact]
    public void Dropout_OnnxRoundTrip_IsIdentityOnInputData()
    {
        var layer = new DropoutLayer<float>(dropoutRate: 0.5);
        var sampleInput = new float[] { 1f, -2f, 3.14f, 0f };

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 4 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 4 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);

        using var session = new InferenceSession(ms.ToArray());
        var ortInput = new OrtDenseTensor(sampleInput, new[] { 1, 4 });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input", ortInput),
        });
        var ortFlat = results.First().AsTensor<float>().ToArray();

        Assert.Equal(sampleInput.Length, ortFlat.Length);
        for (int i = 0; i < sampleInput.Length; i++)
        {
            Assert.Equal(sampleInput[i], ortFlat[i]);
        }
    }

    [Fact]
    public void BatchNormalization_OnnxRoundTrip_MatchesAiDotNetWithinTolerance()
    {
        // BN with 3 features. Materialize stats via a forward pass on
        // representative data so running_mean / running_var are non-default.
        var layer = new BatchNormalizationLayer<float>(numFeatures: 3);
        var trainInput = new AdnTensor(new[] { 4, 3 });
        // Populate with simple deterministic values.
        for (int b = 0; b < 4; b++)
            for (int f = 0; f < 3; f++)
                trainInput[b, f] = (b + 1) * (f + 1) * 0.1f;
        _ = layer.Forward(trainInput);

        // Switch to inference mode for the comparison (so we hit the running-stats path).
        layer.SetTrainingMode(false);

        var inferInput = new AdnTensor(new[] { 1, 3 });
        inferInput[0, 0] = 0.5f;
        inferInput[0, 1] = 0.25f;
        inferInput[0, 2] = -0.1f;
        var aidotnetFlat = layer.Forward(inferInput).ToArray();

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 3 });
        var outputs = layer.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        builder.AddFloatOutput(outputs.Primary, new[] { -1, 3 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);

        using var session = new InferenceSession(ms.ToArray());
        var ortInput = new OrtDenseTensor(new[] { 0.5f, 0.25f, -0.1f }, new[] { 1, 3 });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input", ortInput),
        });
        var ortFlat = results.First().AsTensor<float>().ToArray();

        Assert.Equal(aidotnetFlat.Length, ortFlat.Length);
        for (int i = 0; i < aidotnetFlat.Length; i++)
        {
            Assert.True(
                System.Math.Abs(aidotnetFlat[i] - ortFlat[i]) < 1e-4f,
                $"BN[{i}]: AiDotNet={aidotnetFlat[i]} vs ONNX={ortFlat[i]} (diff > 1e-4)");
        }
    }
#endif
}
