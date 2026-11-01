using System;
using System.IO;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.Onnx;
using AiDotNet.Onnx.Protobuf;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ModelLoading;

/// <summary>
/// The adapter that exposes a frozen ONNX inference graph as a full model: it must run the graph it was given,
/// carry those exact bytes through every serialization path, and refuse to train.
/// </summary>
public class OnnxFullModelAdapterTests
{
    // y = x + [1, 2, 3], so the expected output is known exactly.
    private static byte[] AddBiasGraph()
    {
        var builder = new OnnxGraphBuilder(new OnnxExportOptions { OpsetVersion = 17 });
        builder.AddInput(Info("x"));
        string bias = builder.AddFloatInitializer("bias", new[] { 1f, 2f, 3f }, new[] { 1, 3 });
        builder.AddOp("Add", new[] { "x", bias }, new[] { "y" });
        builder.AddOutput(Info("y"));
        using var stream = new MemoryStream();
        builder.WriteTo(stream);
        return stream.ToArray();
    }

    private static ValueInfoProto Info(string name)
    {
        var shape = new TensorShapeProto();
        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = 1 });
        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = 3 });
        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto
            {
                TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorProto.Types.DataType.Float, Shape = shape }
            }
        };
    }

    private static Tensor<float> Input()
    {
        var x = new Tensor<float>(new[] { 1, 3 });
        x[0, 0] = 0.5f;
        x[0, 1] = -1f;
        x[0, 2] = 2f;
        return x;
    }

    [Fact]
    public void Predict_RunsTheGraph()
    {
        using var adapter = new OnnxFullModelAdapter<float>(AddBiasGraph());

        var y = adapter.Predict(Input());

        Assert.Equal(new[] { 1.5f, 1f, 5f }, y.ToArray());
    }

    [Fact]
    public void Serialization_RoundTripsTheExactGraph()
    {
        var bytes = AddBiasGraph();
        using var adapter = new OnnxFullModelAdapter<float>(bytes);

        Assert.Equal(bytes, adapter.Serialize());

        using var stream = new MemoryStream();
        adapter.SaveState(stream);
        stream.Position = 0;
        using var restored = new OnnxFullModelAdapter<float>(AddBiasGraph());
        restored.LoadState(stream);

        Assert.Equal(adapter.Predict(Input()).ToArray(), restored.Predict(Input()).ToArray());
    }

    [Fact]
    public void Train_IsRefused()
    {
        using var adapter = new OnnxFullModelAdapter<float>(AddBiasGraph());
        Assert.Throws<NotSupportedException>(() => adapter.Train(Input(), Input()));
    }
}
