using AiDotNet.Onnx;
using Xunit;

namespace AiDotNet.Tests.Onnx;

/// <summary>
/// Validates the ONNX <c>dim_param</c> (symbolic axis) encoding added under
/// issue #1211. A graph emitted with <c>OnnxAxisSpec.Symbolic("batch")</c>
/// must round-trip the symbolic name through the protobuf wire format so
/// downstream runtimes (ONNX Runtime, OpenVINO, TensorRT) bind the axis
/// dynamically.
/// </summary>
public class OnnxSymbolicAxisTests
{
    [Fact]
    public void Fixed_ProducesIntegerDim()
    {
        var spec = OnnxAxisSpec.Fixed(3);
        Assert.Null(spec.SymbolicName);
        Assert.Equal(3, spec.FixedDim);
    }

    [Fact]
    public void Symbolic_ProducesNamedDim()
    {
        var spec = OnnxAxisSpec.Symbolic("batch");
        Assert.Equal("batch", spec.SymbolicName);
    }

    [Fact]
    public void Symbolic_RejectsEmptyName()
    {
        Assert.Throws<ArgumentException>(() => OnnxAxisSpec.Symbolic(""));
        Assert.Throws<ArgumentException>(() => OnnxAxisSpec.Symbolic("   "));
    }

    /// <summary>
    /// ONNX TensorShapeProto.Dimension protobuf wire format:
    ///   dim_value (int64) — field tag 1
    ///   dim_param (string) — field tag 2
    /// A graph with one symbolic axis must contain the symbolic-name bytes
    /// somewhere in the serialized output. We don't reach for a full
    /// protobuf decoder here — just assert the symbolic name is present
    /// (it cannot appear by accident — the test name is the only place
    /// "DynamicH" exists in the input).
    /// </summary>
    [Fact]
    public void Builder_EmitsSymbolicNameInGraphBytes()
    {
        var builder = new OnnxModelBuilder();
        var inputAxes = new[]
        {
            OnnxAxisSpec.Symbolic("batch"),
            OnnxAxisSpec.Fixed(3),
            OnnxAxisSpec.Symbolic("DynamicH"),
            OnnxAxisSpec.Symbolic("DynamicW"),
        };
        builder.AddInput("input", inputAxes);
        builder.AddOutput("output", inputAxes);

        var bytes = builder.Build();
        var asString = System.Text.Encoding.UTF8.GetString(bytes);

        Assert.Contains("batch", asString);
        Assert.Contains("DynamicH", asString);
        Assert.Contains("DynamicW", asString);
    }

    [Fact]
    public void Builder_FixedAxes_DoNotEmitSymbolicNames()
    {
        var builder = new OnnxModelBuilder();
        builder.AddInput("input", new[] { OnnxAxisSpec.Fixed(1), OnnxAxisSpec.Fixed(3), OnnxAxisSpec.Fixed(224), OnnxAxisSpec.Fixed(224) });
        builder.AddOutput("output", new[] { OnnxAxisSpec.Fixed(1), OnnxAxisSpec.Fixed(1000) });

        var bytes = builder.Build();
        var asString = System.Text.Encoding.UTF8.GetString(bytes);

        // The strings "batch", "H", "W" should not be present in a fully-fixed-axis graph.
        Assert.DoesNotContain("batch", asString);
    }
}
