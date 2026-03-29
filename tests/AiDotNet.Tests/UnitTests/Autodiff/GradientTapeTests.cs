using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

public class GradientTapeTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    [Fact]
    public void Tape_RecordOp_And_Gradient_SimpleAdd()
    {
        var a = new Tensor<double>([2], new Vector<double>([3.0, 4.0]));
        var b = new Tensor<double>([2], new Vector<double>([1.0, 2.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(a);
        tape.Watch(b);

        // c = a + b (element-wise)
        var c = new Tensor<double>([2]);
        for (int i = 0; i < 2; i++) c[i] = a[i] + b[i];

        tape.RecordOp("add", [a, b], c,
            grad => [grad, grad]); // d(a+b)/da = 1, d(a+b)/db = 1

        var grads = tape.Gradient(c);

        Assert.True(grads.ContainsKey(a));
        Assert.True(grads.ContainsKey(b));
        Assert.Equal(1.0, grads[a][0]);
        Assert.Equal(1.0, grads[a][1]);
        Assert.Equal(1.0, grads[b][0]);
        Assert.Equal(1.0, grads[b][1]);
    }

    [Fact]
    public void Tape_RecordOp_Multiply_CorrectGradients()
    {
        var a = new Tensor<double>([2], new Vector<double>([3.0, 4.0]));
        var b = new Tensor<double>([2], new Vector<double>([5.0, 6.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(a);
        tape.Watch(b);

        // c = a * b (element-wise)
        var c = new Tensor<double>([2]);
        for (int i = 0; i < 2; i++) c[i] = a[i] * b[i];

        tape.RecordOp("mul", [a, b], c,
            grad =>
            {
                // d(a*b)/da = b, d(a*b)/db = a
                var ga = new Tensor<double>(grad.Shape.ToArray());
                var gb = new Tensor<double>(grad.Shape.ToArray());
                for (int i = 0; i < grad.Length; i++)
                {
                    ga[i] = grad[i] * b[i];
                    gb[i] = grad[i] * a[i];
                }
                return [ga, gb];
            });

        var grads = tape.Gradient(c);

        // dc/da = b = [5, 6]
        Assert.Equal(5.0, grads[a][0]);
        Assert.Equal(6.0, grads[a][1]);
        // dc/db = a = [3, 4]
        Assert.Equal(3.0, grads[b][0]);
        Assert.Equal(4.0, grads[b][1]);
    }

    [Fact]
    public void Tape_ChainedOps_GradientFlowsCorrectly()
    {
        var x = new Tensor<double>([1], new Vector<double>([2.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        // y = x * x (square)
        var y = new Tensor<double>([1]);
        y[0] = x[0] * x[0];
        tape.RecordOp("square", [x], y,
            grad =>
            {
                var gx = new Tensor<double>([1]);
                gx[0] = grad[0] * 2.0 * x[0]; // d(x^2)/dx = 2x
                return [gx];
            });

        // z = y + 1 (add constant)
        var z = new Tensor<double>([1]);
        z[0] = y[0] + 1.0;
        tape.RecordOp("add_const", [y], z,
            grad => [grad]); // d(y+1)/dy = 1

        var grads = tape.Gradient(z);

        // dz/dx = dz/dy * dy/dx = 1 * 2x = 2*2 = 4
        Assert.Equal(4.0, grads[x][0]);
    }

    [Fact]
    public void Tape_GradientAccumulation_TensorUsedTwice()
    {
        var x = new Tensor<double>([1], new Vector<double>([3.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        // y = x + x = 2x
        var y = new Tensor<double>([1]);
        y[0] = x[0] + x[0];
        tape.RecordOp("add", [x, x], y,
            grad => [grad, grad]); // Both inputs are x

        var grads = tape.Gradient(y);

        // dy/dx = 2 (gradient accumulated from both uses)
        Assert.Equal(2.0, grads[x][0]);
    }

    [Fact]
    public void Tape_NonPersistent_ThrowsOnSecondGradient()
    {
        var x = new Tensor<double>([1], new Vector<double>([1.0]));
        using var tape = new GradientTape<double>();
        tape.Watch(x);

        tape.RecordOp("id", [x], x, grad => [grad]);
        tape.Gradient(x);

        Assert.Throws<InvalidOperationException>(() => tape.Gradient(x));
    }

    [Fact]
    public void Tape_Persistent_AllowsMultipleGradients()
    {
        var x = new Tensor<double>([1], new Vector<double>([1.0]));
        using var tape = new GradientTape<double>(persistent: true);
        tape.Watch(x);

        tape.RecordOp("id", [x], x, grad => [grad]);

        var g1 = tape.Gradient(x);
        var g2 = tape.Gradient(x);

        Assert.Equal(g1[x][0], g2[x][0]);
    }

    [Fact]
    public void NoGradScope_PausesRecording()
    {
        var x = new Tensor<double>([1], new Vector<double>([2.0]));
        using var tape = new GradientTape<double>();
        tape.Watch(x);

        Assert.True(tape.IsRecording);

        using (var _ = new NoGradScope<double>())
        {
            Assert.False(tape.IsRecording);
            // Ops here would not be recorded
        }

        Assert.True(tape.IsRecording);
    }

    [Fact]
    public void Tape_UnwatchedTensor_NotInGradients()
    {
        var x = new Tensor<double>([1], new Vector<double>([1.0]));
        var y = new Tensor<double>([1], new Vector<double>([2.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x); // Only watch x, not y

        var z = new Tensor<double>([1]);
        z[0] = x[0] + y[0];
        tape.RecordOp("add", [x, y], z, grad => [grad, grad]);

        var grads = tape.Gradient(z);

        Assert.True(grads.ContainsKey(x));
        Assert.False(grads.ContainsKey(y)); // y not watched
    }

    [Fact]
    public async Task Tape_AsyncLocal_FlowsAcrossAwait()
    {
        using var tape = new GradientTape<double>();
        Assert.Same(tape, GradientTape<double>.Current);

        // AsyncLocal flows across await — child task sees parent's tape
        GradientTape<double>? taskSaw = null;
        await Task.Run(() => { taskSaw = GradientTape<double>.Current; });

        Assert.NotNull(taskSaw);
    }

    [Fact]
    public void Tape_NullWhenNoTapeActive()
    {
        // No tape pushed — Current should be null
        // (After all other tests dispose their tapes)
        Assert.Null(GradientTape<double>.Current);
    }

    [Fact]
    public void Tape_Disposed_ThrowsOnWatch()
    {
        var tape = new GradientTape<double>();
        tape.Dispose();

        var x = new Tensor<double>([1]);
        Assert.Throws<ObjectDisposedException>(() => tape.Watch(x));
    }
}
