using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using Microsoft.ML.OnnxRuntime;
using OrtDenseTensor = Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>;
using AdnTensor = AiDotNet.Tensors.LinearAlgebra.Tensor<float>;
using AiDotNet.Tensors.LinearAlgebra;
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
#if !NET471
    // ONNX Runtime's native library binding fails to initialize on net471 in
    // this test environment (AccessViolationException inside SessionOptions
    // ctor — Microsoft.ML.OnnxRuntime targets netstandard2.0 but its native
    // P/Invoke layer is not stable on full-framework MSBuild test hosts).
    // The wire-format reflection assertion below (test #2) does NOT need ORT
    // and runs on both TFMs; only this end-to-end inference test is gated.
    [Fact]
    public void Graph_WithSymbolicAxes_RunsAtMultipleSpatialSizes_WithoutReExport()
    {
        // Build a minimal ONNX graph manually:
        //   input[batch, 3, H, W]  →  ReLU  →  output[batch, 3, H, W]
        // The three symbolic-axis names are deliberately long sentinel
        // strings (NOT the framework-default "batch" / "H" / "W") so the
        // wire-format assertion below is unambiguous: any single-letter
        // dim_param like "H" or "W" could appear by chance in protobuf
        // binary as a 1-byte UTF-8 character; sentinels of 30+ ASCII
        // characters cannot. Closes review-comment #1269.pQdo's concern
        // about substring scans being too weak. (The actual ORT
        // inference at three different shapes below is the strongest
        // proof — but the wire-format assertions also need to be
        // independently meaningful so a regression that breaks the
        // wire emission while ORT still tolerates it gets caught.)
        const string BatchSentinel = "DYNAMIC_BATCH_AXIS_SENTINEL_TEST";
        const string HeightSentinel = "DYNAMIC_HEIGHT_AXIS_SENTINEL_TEST";
        const string WidthSentinel = "DYNAMIC_WIDTH_AXIS_SENTINEL_TEST";
        var builder = new OnnxModelBuilder();
        var inputAxes = new[]
        {
            OnnxAxisSpec.Symbolic(BatchSentinel),
            OnnxAxisSpec.Fixed(3),
            OnnxAxisSpec.Symbolic(HeightSentinel),
            OnnxAxisSpec.Symbolic(WidthSentinel),
        };
        builder.AddInput("input", inputAxes);
        builder.AddRelu("input", "output");
        builder.AddOutput("output", inputAxes);

        var onnxBytes = builder.Build();
        Assert.True(onnxBytes.Length > 0);

        // Wire format for ONNX TensorShapeProto.Dimension's dim_param field
        // (field tag 2, length-delimited UTF-8 string): each symbolic name
        // appears in the bytes preceded by a varint-encoded length byte. We
        // assert the EXACT length-prefixed pattern so a partial-string
        // accident in unrelated binary is caught: the length byte has to
        // match exactly. With 32-character sentinels the probability of a
        // false positive is effectively zero — the only way these byte
        // sequences appear is if the dim_param emission produced them.
        AssertLengthPrefixedStringInBytes(onnxBytes, BatchSentinel);
        AssertLengthPrefixedStringInBytes(onnxBytes, HeightSentinel);
        AssertLengthPrefixedStringInBytes(onnxBytes, WidthSentinel);

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
#endif

    /// <summary>
    /// Validates the FULL <see cref="OnnxExporter.ExportToBytes"/> path —
    /// including the <c>HasDynamicSpatialAxes</c> reflection probe that PR
    /// #1269 fixes (was reading <c>Architecture</c> as a property when
    /// it's a public field, so symbolic-axis emission silently no-op'd
    /// for every NN-derived model).
    /// </summary>
    /// <remarks>
    /// The first test in this file builds the ONNX graph manually via
    /// <see cref="OnnxModelBuilder"/>, which doesn't exercise the
    /// reflection path. This test puts a real <see cref="NeuralNetworkBase{T}"/>
    /// subclass with <c>HasDynamicSpatialDims = true</c> through
    /// <see cref="OnnxExporter.ExportToBytes"/> and asserts the symbolic
    /// names appear in the produced bytes — i.e. the reflection probe
    /// found the field and propagated the dynamic-axis intent into the
    /// graph. Without the field-aware probe, this test fails (no
    /// "batch" / "H" / "W" strings in the output).
    /// </remarks>
    [Fact]
    public void OnnxExporter_RealModel_EmitsSymbolicAxes_ViaReflection()
    {
        var model = new TinyVisionNet();
        Assert.Equal(2, model.Layers.Count);
        // Architecture must be dynamic-spatial for the symbolic-axis
        // emission to fire — pin the precondition explicitly so a
        // regression in CreateDynamicSpatial flows here as a clear
        // assertion rather than a confusing "no symbolic name in bytes".
        Assert.True(model.Architecture.HasDynamicSpatialDims);

        // Warm the layer chain so each layer reports IsShapeResolved=true
        // (the exporter rejects unresolved layers with InvalidOperationException).
        // The warmup shape MUST equal the shape we hand to ExportToBytes
        // below — the exporter wires the input tensor's declared shape
        // into every downstream node's expected rank/dim, and a mismatch
        // between the layers' concretized weight shape and the graph's
        // declared input shape produces a graph that ORT would reject
        // (MatMul rank/inner-dim misalignment). Closes review-comment
        // #1269.xZd-. The 4D [B, C, H, W] shape matches the dynamic-
        // spatial architecture's canonical contract that triggers the
        // symbolic dim_param emission this test is verifying.
        var inputShape = new[] { 1, 3, 32, 32 };
        AdnTensor x = MakeRamp(inputShape);
        x = model.Layers[0].Forward(x);
        x = model.Layers[1].Forward(x);

        // Run the model through the public exporter path — exercises the
        // HasDynamicSpatialAxes reflection probe end-to-end.
        var onnxBytes = OnnxExporter.ExportToBytes(model, inputShape);
        Assert.True(onnxBytes.Length > 0);

        // The reflection probe MUST find the public Architecture field
        // and report HasDynamicSpatialDims=true. If the probe regresses
        // back to property-only lookup, none of these symbolic names
        // appear in the wire bytes (axes get emitted as fixed dims).
        // Use the protobuf length-prefixed-string pattern (varint length +
        // utf-8 name) rather than naive substring scan so single-letter
        // axis names like "H" / "W" are matched only when they appear
        // with the framework-emitted length prefix in front of them —
        // not when 'H' or 'W' appear as 1-byte ASCII anywhere else in
        // the protobuf binary by chance. Closes review-comment #1269.pQdo.
        AssertLengthPrefixedStringInBytes(onnxBytes, "batch");
        AssertLengthPrefixedStringInBytes(onnxBytes, "H");
        AssertLengthPrefixedStringInBytes(onnxBytes, "W");
    }

    /// <summary>
    /// Asserts that <paramref name="haystack"/> contains the protobuf
    /// length-delimited UTF-8 encoding of <paramref name="needle"/>:
    /// a single varint length byte followed by the UTF-8 bytes of
    /// <paramref name="needle"/>. Stronger than a naive substring scan
    /// because it also matches the protobuf framing — random binary
    /// is extremely unlikely to contain `<exact-length-byte> <name-bytes>`
    /// in sequence by chance, especially for multi-byte names. For
    /// names &lt; 128 bytes the varint length is a single byte equal
    /// to the byte count of the UTF-8 string (the high bit isn't set
    /// for values 0-127).
    /// </summary>
    private static void AssertLengthPrefixedStringInBytes(byte[] haystack, string needle)
    {
        var nameBytes = System.Text.Encoding.UTF8.GetBytes(needle);
        if (nameBytes.Length >= 0x80)
        {
            // Multi-byte varint length encoding — not used by any current
            // test sentinel. Add support if a future test needs it.
            throw new System.NotSupportedException(
                $"Test sentinel '{needle}' is {nameBytes.Length} bytes; the assertion " +
                "helper currently only supports single-byte varint length prefixes. " +
                "Either shorten the sentinel or extend the helper for multi-byte varints.");
        }
        byte lengthPrefix = (byte)nameBytes.Length;

        // Slide a window across the bytes looking for the FULL protobuf
        // tag-length-value triple for TensorShapeProto.Dimension.dim_param:
        //   - Field tag byte: 0x12 (field number 2 << 3 | wire type 2 = LEN)
        //   - Length byte:    nameBytes.Length (single-byte varint, since
        //                     all test sentinels are < 128 bytes)
        //   - Name bytes:     UTF-8 encoding of the symbolic name
        // Closes review-comment #1269.vzGE — the previous version checked
        // only `<length> <name>` without the 0x12 field-tag byte, so
        // matches could land on (length+name) sequences that aren't
        // actually dim_param emissions (e.g., elsewhere in the protobuf
        // where field tag 1 / 3 / etc. precedes a similar length+payload).
        // Adding the 0x12 anchor ties the match to the correct field tag.
        const byte DimParamFieldTag = 0x12; // (2 << 3) | wire_type=2 (LEN)
        for (int start = 0; start <= haystack.Length - 2 - nameBytes.Length; start++)
        {
            if (haystack[start] != DimParamFieldTag) continue;
            if (haystack[start + 1] != lengthPrefix) continue;
            bool match = true;
            for (int j = 0; j < nameBytes.Length; j++)
            {
                if (haystack[start + 2 + j] != nameBytes[j]) { match = false; break; }
            }
            if (match) return;
        }
        Assert.Fail(
            $"Could not find protobuf TensorShapeProto.Dimension.dim_param " +
            $"encoding of '{needle}' (0x{DimParamFieldTag:X2} 0x{lengthPrefix:X2} followed by the UTF-8 " +
            $"of '{needle}', total {2 + nameBytes.Length} bytes) anywhere in " +
            $"the {haystack.Length}-byte ONNX graph. This means the symbolic " +
            $"axis was NOT emitted as a dim_param (field tag 2) in the wire " +
            $"format — the axis is being written as a fixed dim_value (field " +
            $"tag 1) or omitted entirely, contradicting the issue #1211 contract.");
    }

    /// <summary>
    /// Minimal lazy-spatial model used by the reflection-path test. The
    /// architecture is dynamic-spatial (rank-3 / 4D [B, C, H, W]) and
    /// uses ONLY shape-preserving activation layers (ReLU). The choice
    /// of activation-only layers is deliberate: this test verifies the
    /// HasDynamicSpatialDims reflection probe drives symbolic-axis
    /// emission via OnnxExporter.ExportToBytes, NOT the exporter's
    /// per-layer op-fidelity for a particular kernel — so we want the
    /// model contents to be a no-op pass-through that produces no
    /// rank/shape mismatches between layer warm-up and graph export.
    /// Activation layers are rank-agnostic (preserve the input shape
    /// exactly), so the same 4D [1, 3, 32, 32] tensor flows from
    /// warm-up through both layers and into the export call without
    /// any MatMul/Conv weight-shape concerns. Closes review-comment
    /// #1269.xZd- (the prior Dense-based stub had a 2D-warmup vs
    /// 4D-export mismatch that produced a non-runnable ONNX graph).
    /// </summary>
    private sealed class TinyVisionNet : NeuralNetworkBase<float>
    {
        public TinyVisionNet()
            : base(NeuralNetworkArchitecture<float>.CreateDynamicSpatial(
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.ImageClassification,
                    channels: 3,
                    outputSize: 1),
                  new MeanSquaredErrorLoss<float>())
        {
            // Eagerly populate Layers so the test can warmup + export
            // without going through Train/Predict's lazy-init path.
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            // Idempotency: framework EnsureArchitectureInitialized may
            // call this AGAIN after the ctor — skip the second call so
            // we don't double-stack the activation chain.
            if (Layers.Count > 0) return;
            Layers.Add(new ActivationLayer<float>(
                (AiDotNet.Interfaces.IActivationFunction<float>)new ReLUActivation<float>()));
            Layers.Add(new ActivationLayer<float>(
                (AiDotNet.Interfaces.IActivationFunction<float>)new ReLUActivation<float>()));
        }

        protected override IFullModel<float, AdnTensor, AdnTensor> CreateNewInstance()
            => new TinyVisionNet();

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        public override void UpdateParameters(Vector<float> parameters)
            => throw new System.NotSupportedException(
                "TinyVisionNet is an export-only test stub — UpdateParameters is not wired.");

        public override ModelMetadata<float> GetModelMetadata()
            => new ModelMetadata<float> { Name = "TinyVisionNet" };
    }

    private static AdnTensor MakeRamp(int[] shape)
    {
        var t = new AdnTensor(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.001f * (i + 1);
        return t;
    }
}
