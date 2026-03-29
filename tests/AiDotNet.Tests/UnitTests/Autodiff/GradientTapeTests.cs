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

    // ─── Advanced chain tests ────────────────────────────────────────

    [Fact]
    public void Tape_DeepChain_10Ops_GradientFlows()
    {
        // x → *2 → +1 → *3 → +2 → *0.5 → +0.1 → *4 → +3 → *0.25 → +0.5
        // Tests that gradients flow correctly through 10 chained ops
        var x = new Tensor<double>([1], new Vector<double>([2.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        var current = x;
        // Apply alternating multiply and add, 10 ops total
        current = DifferentiableOps<double>.MultiplyScalar(current, 2.0);
        current = DifferentiableOps<double>.AddScalar(current, 1.0);
        current = DifferentiableOps<double>.MultiplyScalar(current, 3.0);
        current = DifferentiableOps<double>.AddScalar(current, 2.0);
        current = DifferentiableOps<double>.MultiplyScalar(current, 0.5);
        current = DifferentiableOps<double>.AddScalar(current, 0.1);
        current = DifferentiableOps<double>.MultiplyScalar(current, 4.0);
        current = DifferentiableOps<double>.AddScalar(current, 3.0);
        current = DifferentiableOps<double>.MultiplyScalar(current, 0.25);
        current = DifferentiableOps<double>.AddScalar(current, 0.5);

        var loss = DifferentiableOps<double>.Sum(current);
        var grads = tape.Gradient(loss);

        // Chain rule: d/dx = 2 * 3 * 0.5 * 4 * 0.25 = 3.0
        Assert.True(grads.ContainsKey(x));
        Assert.Equal(3.0, grads[x][0], 10);
    }

    [Fact]
    public void Tape_BranchAndMerge_GradientAccumulatesCorrectly()
    {
        // x → branch into y=x*2 and z=x*3, then merge: loss = sum(y) + sum(z)
        // dx = 2 + 3 = 5
        var x = new Tensor<double>([2], new Vector<double>([1.0, 2.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        var y = DifferentiableOps<double>.MultiplyScalar(x, 2.0);
        var z = DifferentiableOps<double>.MultiplyScalar(x, 3.0);
        var sumY = DifferentiableOps<double>.Sum(y);
        var sumZ = DifferentiableOps<double>.Sum(z);
        var loss = DifferentiableOps<double>.Add(sumY, sumZ);

        var grads = tape.Gradient(loss);

        // dx = 2 + 3 = 5 for each element
        Assert.Equal(5.0, grads[x][0], 10);
        Assert.Equal(5.0, grads[x][1], 10);
    }

    [Fact]
    public void Tape_MultipleWatchedTensors_IndependentGradients()
    {
        // loss = sum(w1 * x) + sum(w2 * x)
        // dw1 = x, dw2 = x, dx = w1 + w2
        var x = new Tensor<double>([3], new Vector<double>([1.0, 2.0, 3.0]));
        var w1 = new Tensor<double>([3], new Vector<double>([0.1, 0.2, 0.3]));
        var w2 = new Tensor<double>([3], new Vector<double>([0.4, 0.5, 0.6]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);
        tape.Watch(w1);
        tape.Watch(w2);

        var prod1 = DifferentiableOps<double>.Multiply(w1, x);
        var prod2 = DifferentiableOps<double>.Multiply(w2, x);
        var sum1 = DifferentiableOps<double>.Sum(prod1);
        var sum2 = DifferentiableOps<double>.Sum(prod2);
        var loss = DifferentiableOps<double>.Add(sum1, sum2);

        var grads = tape.Gradient(loss);

        // dw1 = x
        Assert.Equal(1.0, grads[w1][0], 10);
        Assert.Equal(2.0, grads[w1][1], 10);
        Assert.Equal(3.0, grads[w1][2], 10);

        // dw2 = x
        Assert.Equal(1.0, grads[w2][0], 10);
        Assert.Equal(2.0, grads[w2][1], 10);
        Assert.Equal(3.0, grads[w2][2], 10);

        // dx = w1 + w2
        Assert.Equal(0.5, grads[x][0], 10); // 0.1 + 0.4
        Assert.Equal(0.7, grads[x][1], 10); // 0.2 + 0.5
        Assert.Equal(0.9, grads[x][2], 10); // 0.3 + 0.6
    }

    [Fact]
    public void Tape_MeanOfMatMulChain_GradientCheck()
    {
        // loss = mean(sigmoid(x @ w + b))
        var x = new Tensor<double>([1, 3], new Vector<double>([0.5, -0.3, 0.8]));
        var w = new Tensor<double>([3, 2], new Vector<double>([0.1, 0.2, -0.3, 0.4, 0.5, -0.1]));
        var b = new Tensor<double>([1, 2], new Vector<double>([0.1, -0.2]));

        // Autodiff
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(w);
            var linear = DifferentiableOps<double>.MatMul(x, w);
            var biased = DifferentiableOps<double>.Add(linear, b);
            var activated = DifferentiableOps<double>.Sigmoid(biased);
            var loss = DifferentiableOps<double>.Mean(activated);
            grads = tape.Gradient(loss);
        }

        Assert.True(grads.ContainsKey(w));

        // Numerical check for w
        double eps = 1e-5;
        for (int i = 0; i < w.Length; i++)
        {
            double orig = w[i];

            w[i] = orig + eps;
            double fPlus = ComputeLoss(x, w, b);
            w[i] = orig - eps;
            double fMinus = ComputeLoss(x, w, b);
            w[i] = orig;

            double numerical = (fPlus - fMinus) / (2 * eps);
            double autodiff = grads[w][i];
            double denom = Math.Max(Math.Abs(autodiff), Math.Max(Math.Abs(numerical), 1e-8));
            double relError = Math.Abs(autodiff - numerical) / denom;

            Assert.True(relError < 1e-4,
                $"w[{i}]: autodiff={autodiff:G6}, numerical={numerical:G6}, relError={relError:G4}");
        }

        static double ComputeLoss(Tensor<double> x, Tensor<double> w, Tensor<double> b)
        {
            using var _ = new NoGradScope<double>();
            var linear = DifferentiableOps<double>.MatMul(x, w);
            var biased = DifferentiableOps<double>.Add(linear, b);
            var activated = DifferentiableOps<double>.Sigmoid(biased);
            double sum = 0;
            for (int j = 0; j < activated.Length; j++) sum += activated[j];
            return sum / activated.Length;
        }
    }

    // ─── Edge cases ──────────────────────────────────────────────────

    [Fact]
    public void Tape_VerySmallValues_NoNaN()
    {
        var x = new Tensor<double>([3], new Vector<double>([1e-15, 1e-10, 1e-8]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        var y = DifferentiableOps<double>.Multiply(x, x);
        var loss = DifferentiableOps<double>.Sum(y);
        var grads = tape.Gradient(loss);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(double.IsNaN(grads[x][i]), $"Gradient[{i}] is NaN for small input");
            Assert.False(double.IsInfinity(grads[x][i]), $"Gradient[{i}] is Infinity for small input");
        }
    }

    [Fact]
    public void Tape_LargeValues_NoOverflow()
    {
        var x = new Tensor<double>([3], new Vector<double>([100.0, 200.0, 50.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        // Sigmoid saturates at large values — gradient should be near zero, not NaN
        var y = DifferentiableOps<double>.Sigmoid(x);
        var loss = DifferentiableOps<double>.Sum(y);
        var grads = tape.Gradient(loss);

        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(double.IsNaN(grads[x][i]), $"Gradient[{i}] is NaN for large input");
            Assert.False(double.IsInfinity(grads[x][i]), $"Gradient[{i}] is Infinity for large input");
            // Sigmoid gradient at x=100 is ~0 (saturated)
            Assert.True(Math.Abs(grads[x][i]) < 1e-10, $"Sigmoid gradient at x={x[i]} should be ~0");
        }
    }

    [Fact]
    public void Tape_EmptyOps_GradientIsIdentity()
    {
        // If no ops recorded, gradient of watched tensor should be the seed (ones)
        var x = new Tensor<double>([3], new Vector<double>([1.0, 2.0, 3.0]));

        using var tape = new GradientTape<double>();
        tape.Watch(x);

        // x IS the loss — no ops between watch and gradient
        var grads = tape.Gradient(x);

        Assert.True(grads.ContainsKey(x));
        Assert.Equal(1.0, grads[x][0]);
        Assert.Equal(1.0, grads[x][1]);
        Assert.Equal(1.0, grads[x][2]);
    }

    [Fact]
    public void Tape_NestedTapes_InnerDoesNotAffectOuter()
    {
        var x = new Tensor<double>([1], new Vector<double>([3.0]));

        using var outer = new GradientTape<double>(persistent: true);
        outer.Watch(x);

        var y = DifferentiableOps<double>.MultiplyScalar(x, 2.0); // Recorded on outer

        using (var inner = new GradientTape<double>())
        {
            // Inner tape is now Current
            Assert.Same(inner, GradientTape<double>.Current);

            var z = DifferentiableOps<double>.MultiplyScalar(y, 3.0); // Recorded on inner
        }

        // After inner disposed, outer is Current again
        Assert.Same(outer, GradientTape<double>.Current);

        // Outer tape only has the first op (y = x*2)
        var outerLoss = DifferentiableOps<double>.Sum(y);
        var grads = outer.Gradient(outerLoss);

        Assert.Equal(2.0, grads[x][0]); // dy/dx = 2
    }

    // ─── Thread safety ───────────────────────────────────────────────

    [Fact]
    public async Task Tape_ConcurrentTasks_Isolated()
    {
        // Two concurrent tasks should have independent tapes
        var results = await Task.WhenAll(
            Task.Run(() =>
            {
                var x = new Tensor<double>([1], new Vector<double>([2.0]));
                using var tape = new GradientTape<double>();
                tape.Watch(x);
                var y = DifferentiableOps<double>.MultiplyScalar(x, 3.0);
                var loss = DifferentiableOps<double>.Sum(y);
                return tape.Gradient(loss)[x][0]; // Should be 3.0
            }),
            Task.Run(() =>
            {
                var x = new Tensor<double>([1], new Vector<double>([2.0]));
                using var tape = new GradientTape<double>();
                tape.Watch(x);
                var y = DifferentiableOps<double>.MultiplyScalar(x, 5.0);
                var loss = DifferentiableOps<double>.Sum(y);
                return tape.Gradient(loss)[x][0]; // Should be 5.0
            })
        );

        Assert.Equal(3.0, results[0]);
        Assert.Equal(5.0, results[1]);
    }

    [Fact]
    public void Tape_Reset_AllowsReuse()
    {
        var x = new Tensor<double>([1], new Vector<double>([2.0]));

        using var tape = new GradientTape<double>(persistent: true);
        tape.Watch(x);

        var y1 = DifferentiableOps<double>.MultiplyScalar(x, 3.0);
        var loss1 = DifferentiableOps<double>.Sum(y1);
        var grads1 = tape.Gradient(loss1);
        Assert.Equal(3.0, grads1[x][0]);

        // Reset and compute different gradient
        tape.Reset();
        tape.Watch(x);

        var y2 = DifferentiableOps<double>.MultiplyScalar(x, 7.0);
        var loss2 = DifferentiableOps<double>.Sum(y2);
        var grads2 = tape.Gradient(loss2);
        Assert.Equal(7.0, grads2[x][0]);
    }
}
