using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

public class TensorCheckpointTests
{
    [Fact]
    public void Capture_DetachesFromAnActiveTape()
    {
        var live = new Tensor<double>(new[] { 1 }); live[0] = 2;
        using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>();
        var tracked = AiDotNetEngine.Current.TensorMultiply(live, live);
        var snapshot = TensorCheckpoint<double>.Capture(new[] { tracked });
        Assert.Null(snapshot[0].GradFn);
        Assert.Equal(4, snapshot[0][0]);
        var continued = AiDotNetEngine.Current.TensorMultiply(live, live);
        var gradients = tape.ComputeGradients(continued, new[] { live });
        Assert.Equal(4, gradients[live][0]);
    }

    [Fact]
    public void Capture_ReusesStorage_AndRestoresLiveIdentity()
    {
        var parameter = new Tensor<double>(new[] { 2 });
        parameter[0] = 3; parameter[1] = 4;
        var live = new[] { parameter };
        var saved = TensorCheckpoint<double>.Capture(live);
        parameter[0] = 9;
        Assert.Equal(3, saved[0][0]);
        var storage = saved[0];
        Assert.Same(saved, TensorCheckpoint<double>.Capture(live, saved));
        Assert.Same(storage, saved[0]);
        parameter[0] = -1;
        TensorCheckpoint<double>.Restore(live, saved);
        Assert.Same(parameter, live[0]);
        Assert.Equal(9, parameter[0]);
        Assert.Equal(4, parameter[1]);
    }

    [Fact]
    public void Restore_ValidatesAllShapesBeforeWriting()
    {
        var first = new Tensor<double>(new[] { 1 }); first[0] = 7;
        var live = new[] { first, new Tensor<double>(new[] { 2, 2 }) };
        var saved = new[] { new Tensor<double>(new[] { 1 }), new Tensor<double>(new[] { 4 }) };
        Assert.Throws<ArgumentException>(() => TensorCheckpoint<double>.Restore(live, saved));
        Assert.Equal(7, first[0]);
        Assert.Throws<ArgumentException>(() => TensorCheckpoint<double>.Capture(live, saved));
        Assert.Throws<ArgumentException>(() => TensorCheckpoint<double>.Restore(live, new[] { saved[0] }));
    }

    [Fact]
    public void Capture_RejectsAliasingAndSupportsEmptyCollections()
    {
        var live = new[] { new Tensor<float>(new[] { 2 }), new Tensor<float>(new[] { 2 }) };
        Assert.Throws<ArgumentException>(() => TensorCheckpoint<float>.Capture(live, new[] { live[1], live[0] }));
        var saved = new Tensor<float>(new[] { 2 });
        Assert.Throws<ArgumentException>(() => TensorCheckpoint<float>.Capture(live, new[] { saved, saved }));
        Assert.Empty(TensorCheckpoint<float>.Capture(Array.Empty<Tensor<float>>()));
        TensorCheckpoint<float>.Restore(Array.Empty<Tensor<float>>(), Array.Empty<Tensor<float>>());
    }
}
