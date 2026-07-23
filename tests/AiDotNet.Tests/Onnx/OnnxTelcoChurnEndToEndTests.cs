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
/// End-to-end ONNX export of a multi-layer model matching the shape used in
/// the Databricks training session's Telco Customer Churn demo:
///
///   Input(4 features) → Dense(8, ReLU) → Dense(1, Sigmoid) → output ∈ [0, 1]
///
/// This is the integration test the new protobuf-based ONNX export path
/// needs to pass before we trust the Databricks demo will work:
///   1. Build a chain of layers
///   2. Materialise weights and run AiDotNet Forward end-to-end
///   3. Export the chain to ONNX bytes via the new path
///   4. Load with Microsoft.ML.OnnxRuntime.InferenceSession
///   5. Run the same input through ORT
///   6. Assert outputs match element-wise within tolerance
/// </summary>
public class OnnxTelcoChurnEndToEndTests
{
#if !NET471
    [Fact]
    public void TelcoChurnShaped_OnnxRoundTrip_MatchesAiDotNetOutput()
    {
        // 4-feature input → 8-unit ReLU hidden → 1-unit Sigmoid output.
        // Standard binary classifier shape; Telco-Churn-equivalent at this scale.
        var hidden = new DenseLayer<float>(
            outputSize: 8,
            activationFunction: new ReLUActivation<float>());
        var output = new DenseLayer<float>(
            outputSize: 1,
            activationFunction: new SigmoidActivation<float>());

        // Materialise weights with a warm-up forward pass.
        var warmup = new AdnTensor(new[] { 1, 4 });
        var afterHidden = hidden.Forward(warmup);
        _ = output.Forward(afterHidden);

        // Deterministic input — mix of small positives and negatives so ReLU
        // gates some hidden units and lets others through, exercising both
        // sides of the activation.
        var sampleInput = new float[] { 0.6f, -0.3f, 1.2f, 0.05f };
        var inputTensor = new AdnTensor(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) inputTensor[0, i] = sampleInput[i];

        var aidotnetHidden = hidden.Forward(inputTensor);
        var aidotnetOutput = output.Forward(aidotnetHidden);
        var aidotnetFlat = aidotnetOutput.ToArray();

        // Export the chain using the new ConvertToOnnx path on each layer.
        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 4 });

        var hiddenOutputs = hidden.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        var finalOutputs = output.ConvertToOnnx(builder, new OnnxLayerInputs(hiddenOutputs.Primary));

        builder.AddFloatOutput(finalOutputs.Primary, new[] { -1, 1 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);
        var onnxBytes = ms.ToArray();

        Assert.True(onnxBytes.Length > 0, "ONNX bytes must be non-empty");

        // Load + score via ORT.
        using var session = new InferenceSession(onnxBytes);
        var ortInput = new OrtDenseTensor(sampleInput, new[] { 1, 4 });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input", ortInput),
        });
        var ortFlat = results.First().AsTensor<float>().ToArray();

        Assert.Equal(aidotnetFlat.Length, ortFlat.Length);
        Assert.Single(aidotnetFlat); // Binary classifier output is rank-2 [batch=1, 1]

        // Sigmoid output should be in [0, 1].
        Assert.InRange(aidotnetFlat[0], 0.0f, 1.0f);
        Assert.InRange(ortFlat[0], 0.0f, 1.0f);

        for (int i = 0; i < aidotnetFlat.Length; i++)
        {
            Assert.True(
                System.Math.Abs(aidotnetFlat[i] - ortFlat[i]) < 1e-5f,
                $"End-to-end output[{i}]: AiDotNet={aidotnetFlat[i]} vs ONNX={ortFlat[i]} (diff > 1e-5)");
        }
    }

    [Fact]
    public void TelcoChurnShaped_OnnxOutput_StaysConsistentAcrossBatchSizes()
    {
        // Same model shape, but verify the symbolic batch axis works — score
        // multiple inputs through one exported model.
        var hidden = new DenseLayer<float>(
            outputSize: 8,
            activationFunction: new ReLUActivation<float>());
        var output = new DenseLayer<float>(
            outputSize: 1,
            activationFunction: new SigmoidActivation<float>());

        var warmup = new AdnTensor(new[] { 1, 4 });
        _ = output.Forward(hidden.Forward(warmup));

        var builder = new OnnxGraphBuilder(new OnnxExportOptions());
        builder.AddFloatInput("input", new[] { -1, 4 });
        var hiddenOutputs = hidden.ConvertToOnnx(builder, new OnnxLayerInputs("input"));
        var finalOutputs = output.ConvertToOnnx(builder, new OnnxLayerInputs(hiddenOutputs.Primary));
        builder.AddFloatOutput(finalOutputs.Primary, new[] { -1, 1 });

        using var ms = new MemoryStream();
        builder.WriteTo(ms);

        using var session = new InferenceSession(ms.ToArray());

        // Score 1 input.
        var single = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };
        var ortSingle = new OrtDenseTensor(single, new[] { 1, 4 });
        using var r1 = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortSingle) });
        var p1 = r1.First().AsTensor<float>().ToArray();
        Assert.Single(p1);

        // Score 3 inputs in one batch — symbolic batch dim must accept this.
        var batch = new float[] { 0.5f, 0.5f, 0.5f, 0.5f,   1.0f, 0f, 0f, 0f,   0f, 0f, 0f, 1.0f };
        var ortBatch = new OrtDenseTensor(batch, new[] { 3, 4 });
        using var r3 = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", ortBatch) });
        var p3 = r3.First().AsTensor<float>().ToArray();
        Assert.Equal(3, p3.Length);

        // First row of the batch should match the single-row prediction.
        Assert.True(System.Math.Abs(p1[0] - p3[0]) < 1e-5f,
            $"Batch row 0 should match single-row: single={p1[0]} batch[0]={p3[0]}");
    }
#endif
}
