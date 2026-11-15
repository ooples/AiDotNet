using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

/// <summary>
/// The gradient behaviour AiDotNet trains with, checked on the tape it now uses (AiDotNet.Tensors
/// <see cref="GradientTape{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// These replace ComputationNodeBackwardTests and TensorOperationsVerificationTests, which tested the deleted legacy
/// graph. Every behaviour those tests checked is checked here on the replacement: each factor of a product gets the
/// other factor; a chain matches its closed form; a node that feeds two branches collects both paths; a scalar loss
/// reaches every requested parameter and nothing else; a non-unit output cotangent scales the gradient; running the
/// backward again does not accumulate; and every op the legacy verifier covered (ReLU, Sigmoid, Tanh, Negate, Exp,
/// Log, Sqrt, Square, LeakyReLU, and both operands of Add, Subtract, Multiply and Divide) matches central finite
/// differences in float and in double.
/// </para>
/// <para>
/// They live here rather than only in the Tensors repository because AiDotNet pins a Tensors package version: a
/// regression in the version this library consumes must fail this library's CI.
/// </para>
/// </remarks>
public class TapeGradientParityTests
{
    /// <summary>The unary ops the legacy verifier covered.</summary>
    public enum UnaryOp { ReLU, Sigmoid, Tanh, Negate, Exp, Log, Sqrt, Square, LeakyReLU }

    /// <summary>The binary ops the legacy verifier covered, each checked on both operands.</summary>
    public enum BinaryOp { Add, Subtract, Multiply, Divide }

    private static IEngine Engine => AiDotNetEngine.Current;

    private static Tensor<T> TensorOf<T>(params double[] values) =>
        new Tensor<double>(values, new[] { values.Length }).Cast<T>();

    private static double[] ToDoubles<T>(Tensor<T> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++)
            values[i] = Convert.ToDouble(tensor[i], System.Globalization.CultureInfo.InvariantCulture);
        return values;
    }

    private static T Scalar<T>(double value) =>
        (T)Convert.ChangeType(value, typeof(T), System.Globalization.CultureInfo.InvariantCulture);

    private static Tensor<T> SumAll<T>(Tensor<T> t) =>
        Engine.ReduceSum(t, Enumerable.Range(0, t.Rank).ToArray(), keepDims: false);

    private static double[] GradientOf<T>(Tensor<T> loss, Tensor<T> source, GradientTape<T> tape)
    {
        var grads = tape.ComputeGradients(loss, new[] { source });
        Assert.True(grads.TryGetValue(source, out var gradient), "the source received no gradient");
        return gradient is null ? Array.Empty<double>() : ToDoubles(gradient);
    }

    // ── Behaviours of the backward driver ─────────────────────────────────────────────────────────────

    [Fact(Timeout = 60000)]
    public async Task Product_GivesEachFactorTheOtherFactor()
    {
        await Task.Yield();
        var a = TensorOf<double>(2.0, -3.0, 0.5);
        var b = TensorOf<double>(5.0, 7.0, -4.0);

        double[] da, db;
        using (var tape = new GradientTape<double>())
        {
            var grads = tape.ComputeGradients(SumAll(Engine.TensorMultiply(a, b)), new[] { a, b });
            da = ToDoubles(grads[a]);
            db = ToDoubles(grads[b]);
        }

        // d(a*b)/da = b and d(a*b)/db = a, exactly.
        Assert.Equal(ToDoubles(b), da);
        Assert.Equal(ToDoubles(a), db);
    }

    [Fact(Timeout = 60000)]
    public async Task Chain_MatchesTheClosedForm()
    {
        await Task.Yield();
        var x = TensorOf<double>(0.3, -1.1, 2.0);

        double[] g;
        using (var tape = new GradientTape<double>())
        {
            // f = exp(x) * x, so df/dx = exp(x)(1 + x).
            g = GradientOf(SumAll(Engine.TensorMultiply(Engine.TensorExp(x), x)), x, tape);
        }

        var values = ToDoubles(x);
        Assert.Equal(values.Length, g.Length);
        for (int i = 0; i < values.Length; i++)
        {
            double expected = Math.Exp(values[i]) * (1.0 + values[i]);
            Assert.True(Math.Abs(g[i] - expected) < 1e-12, $"at {i}: got {g[i]:G17}, closed form {expected:G17}");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task NodeFeedingTwoBranches_CollectsBothPaths()
    {
        await Task.Yield();
        var x = TensorOf<double>(1.5, -2.0);

        double[] g;
        using (var tape = new GradientTape<double>())
        {
            // f = x*x + x, so df/dx = 2x + 1: x's gradient is complete only after both the product and the sum.
            g = GradientOf(SumAll(Engine.TensorAdd(Engine.TensorMultiply(x, x), x)), x, tape);
        }

        Assert.Equal(2, g.Length);
        Assert.Equal(2.0 * 1.5 + 1.0, g[0], 12);
        Assert.Equal(2.0 * -2.0 + 1.0, g[1], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task ScalarLoss_ReachesEveryRequestedParameter_AndNothingElse()
    {
        await Task.Yield();
        var w = TensorOf<double>(0.5, -0.25, 2.0);
        var x = TensorOf<double>(1.0, 4.0, -0.5);

        double[] dw;
        bool xGotGradient;
        using (var tape = new GradientTape<double>())
        {
            // loss = sum(w * x): its gradient with respect to w is x.
            var grads = tape.ComputeGradients(SumAll(Engine.TensorMultiply(w, x)), new[] { w });
            dw = ToDoubles(grads[w]);
            xGotGradient = grads.ContainsKey(x);
        }

        var xs = ToDoubles(x);
        Assert.Equal(xs.Length, dw.Length);
        for (int i = 0; i < xs.Length; i++) Assert.Equal(xs[i], dw[i], 12);
        // x was not requested, so nothing is computed for it.
        Assert.False(xGotGradient);
    }

    [Fact(Timeout = 60000)]
    public async Task Gradient_MatchesCentralDifferences()
    {
        await Task.Yield();
        var start = new[] { 0.7, -0.4, 1.3 };
        var x = TensorOf<double>(start);

        double[] g;
        using (var tape = new GradientTape<double>())
        {
            g = GradientOf(SumAll(Engine.Tanh(Engine.TensorMultiply(x, x))), x, tape);
        }

        static double SumOfTanhSquares(double[] v) => v.Sum(e => Math.Tanh(e * e));
        const double h = 1e-6;
        Assert.Equal(start.Length, g.Length);
        for (int i = 0; i < start.Length; i++)
        {
            var up = (double[])start.Clone();
            up[i] += h;
            var down = (double[])start.Clone();
            down[i] -= h;
            double numeric = (SumOfTanhSquares(up) - SumOfTanhSquares(down)) / (2.0 * h);
            Assert.True(Math.Abs(g[i] - numeric) < 1e-7, $"at {i}: tape {g[i]:G17}, central difference {numeric:G17}");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task NonUnitOutputCotangent_ScalesTheGradient()
    {
        await Task.Yield();
        var x = TensorOf<double>(3.0, -1.0);
        var cotangent = TensorOf<double>(2.0, -0.5);

        double[] g;
        using (var tape = new GradientTape<double>())
        {
            // The vector-Jacobian product of f = x^2 with cotangent c is c * 2x.
            g = GradientOf(SumAll(Engine.TensorMultiply(Engine.TensorMultiply(x, x), cotangent)), x, tape);
        }

        Assert.Equal(2, g.Length);
        Assert.Equal(2.0 * 2.0 * 3.0, g[0], 12);
        Assert.Equal(-0.5 * 2.0 * -1.0, g[1], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task RunningTheBackwardAgain_DoesNotAccumulate()
    {
        await Task.Yield();
        var x = TensorOf<double>(2.0);

        double[] once, twice;
        using (var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true }))
        {
            var loss = SumAll(Engine.TensorMultiply(x, x));
            once = GradientOf(loss, x, tape);
            twice = GradientOf(loss, x, tape);
        }

        // d(x^2)/dx = 2x = 4 at x = 2. Each backward returns the gradient of this loss; summing passes (micro-batch
        // accumulation) is the optimizer's job, so a second pass must not leave 8.
        Assert.Single(once);
        Assert.Single(twice);
        Assert.Equal(4.0, once[0], 12);
        Assert.Equal(4.0, twice[0], 12);
    }

    // ── Per-op finite-difference verification (float and double) ─────────────────────────────────────

    private static Tensor<T> Apply<T>(UnaryOp op, Tensor<T> x) => op switch
    {
        UnaryOp.ReLU => Engine.ReLU(x),
        UnaryOp.Sigmoid => Engine.Sigmoid(x),
        UnaryOp.Tanh => Engine.Tanh(x),
        UnaryOp.Negate => Engine.TensorNegate(x),
        UnaryOp.Exp => Engine.TensorExp(x),
        UnaryOp.Log => Engine.TensorLog(x),
        UnaryOp.Sqrt => Engine.TensorSqrt(x),
        UnaryOp.Square => Engine.TensorSquare(x),
        UnaryOp.LeakyReLU => Engine.LeakyReLU(x, Scalar<T>(0.1)),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, "unknown op"),
    };

    private static Tensor<T> Apply<T>(BinaryOp op, Tensor<T> a, Tensor<T> b) => op switch
    {
        BinaryOp.Add => Engine.TensorAdd(a, b),
        BinaryOp.Subtract => Engine.TensorSubtract(a, b),
        BinaryOp.Multiply => Engine.TensorMultiply(a, b),
        BinaryOp.Divide => Engine.TensorDivide(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, "unknown op"),
    };

    // Values stay away from the ReLU/LeakyReLU kink at 0 and inside the domain of Log and Sqrt.
    private static double[] UnaryProbe(UnaryOp op) => op is UnaryOp.Log or UnaryOp.Sqrt
        ? new[] { 0.5, 1.0, 2.0, 3.5 }
        : new[] { -2.0, -0.6, 0.4, 1.7 };

    // A non-unit, non-uniform cotangent, so a gradient that is right only up to a constant or a permutation fails.
    private static readonly double[] Cotangent = { 0.7, -1.3, 0.4, 2.1 };

    private static void AssertMatchesCentralDifferences<T>(
        Func<Tensor<T>, Tensor<T>> forward, double[] start, string label,
        double epsilon, double relativeTolerance, double absoluteTolerance)
    {
        var weights = TensorOf<T>(Cotangent);
        var x = TensorOf<T>(start);
        double[] analytic;
        using (var tape = new GradientTape<T>())
        {
            analytic = GradientOf(SumAll(Engine.TensorMultiply(forward(x), weights)), x, tape);
        }

        double Objective(double[] values)
        {
            var y = ToDoubles(forward(TensorOf<T>(values)));
            double total = 0;
            for (int i = 0; i < y.Length; i++) total += y[i] * Cotangent[i];
            return total;
        }

        Assert.Equal(start.Length, analytic.Length);
        for (int i = 0; i < start.Length; i++)
        {
            var up = (double[])start.Clone();
            up[i] += epsilon;
            var down = (double[])start.Clone();
            down[i] -= epsilon;
            double numeric = (Objective(up) - Objective(down)) / (2.0 * epsilon);
            double scale = Math.Max(Math.Abs(numeric), Math.Abs(analytic[i]));
            double error = Math.Abs(analytic[i] - numeric);
            Assert.True(error <= absoluteTolerance || error <= relativeTolerance * scale,
                $"{label}[{i}]: tape {analytic[i]:G9}, central difference {numeric:G9}");
        }
    }

    private static void VerifyUnary<T>(UnaryOp op, double epsilon, double relativeTolerance, double absoluteTolerance)
        => AssertMatchesCentralDifferences<T>(x => Apply(op, x), UnaryProbe(op), $"{op}<{typeof(T).Name}>",
            epsilon, relativeTolerance, absoluteTolerance);

    private static void VerifyBinary<T>(BinaryOp op, double epsilon, double relativeTolerance, double absoluteTolerance)
    {
        // Both operands bounded away from zero, so the divisor of Divide is well conditioned.
        var left = new[] { 1.5, -2.0, 0.8, 3.0 };
        var right = new[] { 2.5, 1.2, -1.6, 0.9 };
        AssertMatchesCentralDifferences<T>(a => Apply(op, a, TensorOf<T>(right)), left,
            $"{op}<{typeof(T).Name}>.left", epsilon, relativeTolerance, absoluteTolerance);
        AssertMatchesCentralDifferences<T>(b => Apply(op, TensorOf<T>(left), b), right,
            $"{op}<{typeof(T).Name}>.right", epsilon, relativeTolerance, absoluteTolerance);
    }

    // Float: about 7 significant digits, so the central difference at eps = 1e-3 is good to about 1e-4 relative;
    // 5e-2 still rejects any wrong derivative. Double: eps = 1e-5 leaves an error near 1e-10.
    private const double FloatEpsilon = 1e-3, FloatRelative = 5e-2, FloatAbsolute = 1e-4;
    private const double DoubleEpsilon = 1e-5, DoubleRelative = 1e-6, DoubleAbsolute = 1e-9;

    [Theory(Timeout = 60000)]
    [InlineData(UnaryOp.ReLU)]
    [InlineData(UnaryOp.Sigmoid)]
    [InlineData(UnaryOp.Tanh)]
    [InlineData(UnaryOp.Negate)]
    [InlineData(UnaryOp.Exp)]
    [InlineData(UnaryOp.Log)]
    [InlineData(UnaryOp.Sqrt)]
    [InlineData(UnaryOp.Square)]
    [InlineData(UnaryOp.LeakyReLU)]
    public async Task UnaryOp_GradientMatchesCentralDifferences(UnaryOp op)
    {
        await Task.Yield();
        VerifyUnary<float>(op, FloatEpsilon, FloatRelative, FloatAbsolute);
        VerifyUnary<double>(op, DoubleEpsilon, DoubleRelative, DoubleAbsolute);
    }

    [Theory(Timeout = 60000)]
    [InlineData(BinaryOp.Add)]
    [InlineData(BinaryOp.Subtract)]
    [InlineData(BinaryOp.Multiply)]
    [InlineData(BinaryOp.Divide)]
    public async Task BinaryOp_GradientMatchesCentralDifferences_ForBothOperands(BinaryOp op)
    {
        await Task.Yield();
        VerifyBinary<float>(op, FloatEpsilon, FloatRelative, FloatAbsolute);
        VerifyBinary<double>(op, DoubleEpsilon, DoubleRelative, DoubleAbsolute);
    }
}
