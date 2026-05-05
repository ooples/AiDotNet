using System.Linq;
using AiDotNet.Onnx;
using Microsoft.ML.OnnxRuntime;
using OrtDenseTensor = Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>;
using Xunit;

namespace AiDotNet.Tests.Onnx;

/// <summary>
/// Issue #1211 end-to-end coverage: an ONNX graph emitted with
/// <see cref="OnnxAxisSpec.Symbolic"/> on the batch / H / W axes must run
/// through ONNX Runtime at multiple input shapes WITHOUT re-export. The
/// pre-existing <c>OnnxSymbolicAxisTests</c> validate the protobuf-level
/// <c>dim_param</c> wire format; this test confirms ORT actually binds
/// those axes dynamically at inference time, which is the contract a
/// downstream consumer (ONNX Runtime, OpenVINO, TensorRT) cares about.
/// </summary>
/// <remarks>
/// <para>
/// The issue spec asks for "ResNet50 at 224×224 + 320×320 without
/// re-export". The exporter's per-layer op coverage (Dense / ReLU /
/// Sigmoid / Tanh / Softmax / Dropout / Flatten) doesn't yet include Conv
/// or Pooling ops — adding those is its own scope (would let us export a
/// real ResNet50 chain). Until that lands, we exercise the symbolic-axis
/// machinery via a minimal supported op (ReLU) so the test runs today and
/// blocks any regression in the wire format / dim_param emission.
/// </para>
/// <para>
/// If this test fails ONLY for the second / third inference (320×320 or
/// non-square), the symbolic axes didn't actually become dynamic in the
/// exported bytes — ORT pinned the graph to the trace shape. If it fails
/// for the FIRST inference (matches trace), the wire format itself is
/// malformed.
/// </para>
/// </remarks>
public class OnnxSymbolicAxisRuntimeTests
{
    [Fact]
    public void Graph_WithSymbolicAxes_RunsAtMultipleSpatialSizes_WithoutReExport()
    {
        // Build a minimal ONNX graph manually:
        //   input[batch, 3, H, W]  →  ReLU  →  output[batch, 3, H, W]
        // batch / H / W are dim_param ("batch" / "H" / "W"); the channel
        // axis stays a concrete dim_value=3.
        var builder = new OnnxModelBuilder();
        var inputAxes = new[]
        {
            OnnxAxisSpec.Symbolic("batch"),
            OnnxAxisSpec.Fixed(3),
            OnnxAxisSpec.Symbolic("H"),
            OnnxAxisSpec.Symbolic("W"),
        };
        builder.AddInput("input", inputAxes);
        builder.AddRelu("input", "output");
        builder.AddOutput("output", inputAxes);

        var onnxBytes = builder.Build();
        Assert.True(onnxBytes.Length > 0);

        // Sanity: every symbolic name should round-trip through the wire.
        var asString = System.Text.Encoding.UTF8.GetString(onnxBytes);
        Assert.Contains("batch", asString);
        Assert.Contains("H", asString);
        Assert.Contains("W", asString);

        // Hand the exported bytes to ORT and run inference at THREE
        // different shapes through the SAME session. If the dim_param
        // axes didn't bind dynamically, the second / third call throws a
        // shape-mismatch error.
        using var session = new InferenceSession(onnxBytes);

        // First inference: 1×3×224×224 (paper-canonical ImageNet size).
        AssertReluRoundtrip(session, batch: 1, height: 224, width: 224);

        // Second inference: 1×3×320×320 — the #1211 contract: a single
        // exported graph runs at any (batch, H, W) without re-export.
        AssertReluRoundtrip(session, batch: 1, height: 320, width: 320);

        // Third inference: 2×3×256×192 — confirms batch axis is dynamic
        // AND H ≠ W is accepted (rectangular non-square inputs).
        AssertReluRoundtrip(session, batch: 2, height: 256, width: 192);
    }

    private static void AssertReluRoundtrip(InferenceSession session, int batch, int height, int width)
    {
        // ReLU is identity for non-negative inputs and zero for negative.
        // Build a deterministic ramp that crosses zero so the assertion
        // catches a graph that's running the wrong op (e.g. a copy that
        // ignored the ReLU). Negative inputs must come out as zero.
        var inputData = new float[batch * 3 * height * width];
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (i % 7) - 3; // values in {-3, -2, -1, 0, 1, 2, 3}

        var ortInput = new OrtDenseTensor(inputData, new[] { batch, 3, height, width });
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input", ortInput),
        });

        var output = results.First().AsTensor<float>();
        var outShape = output.Dimensions.ToArray();

        // ReLU preserves shape.
        Assert.Equal(4, outShape.Length);
        Assert.Equal(batch, outShape[0]);
        Assert.Equal(3, outShape[1]);
        Assert.Equal(height, outShape[2]);
        Assert.Equal(width, outShape[3]);

        // Spot-check: input values < 0 → output 0; input ≥ 0 → identity.
        var flat = output.ToArray();
        for (int i = 0; i < System.Math.Min(64, flat.Length); i++)
        {
            float expected = inputData[i] < 0 ? 0f : inputData[i];
            Assert.Equal(expected, flat[i]);
        }
    }
}
