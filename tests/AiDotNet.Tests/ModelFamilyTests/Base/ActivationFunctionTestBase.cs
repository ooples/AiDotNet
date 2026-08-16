using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for IActivationFunction&lt;T&gt; implementations.
/// Tests mathematical invariants that every activation function must satisfy:
/// finite output, derivative correctness (numerical gradient check),
/// monotonicity properties, and edge case handling.
/// </summary>
public abstract class ActivationFunctionTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected static T ToT(double value) => NumOps.FromDouble(value);

    protected static double ToD(T value) => Convert.ToDouble(value);

    protected abstract IActivationFunction<T> CreateActivation();

    protected virtual double Tolerance => typeof(T) == typeof(float) ? 1e-6 : 1e-10;

    /// <summary>
    /// Whether the activation is monotonically non-decreasing.
    /// True for: ReLU, Sigmoid, Tanh, Identity, Softplus, GELU, Swish, SiLU
    /// False for: none of the standard ones (but some exotic ones might not be)
    /// </summary>
    protected virtual bool IsMonotonic => true;

    /// <summary>
    /// Whether Activate(0) should be exactly 0.
    /// True for: ReLU, Tanh, Identity, LeakyReLU, ELU, SELU, Swish, GELU, SoftSign
    /// False for: Sigmoid (0.5), Softplus (ln2)
    /// </summary>
    protected virtual bool ZeroMapsToZero => true;

    /// <summary>
    /// Whether the activation output is bounded (e.g. Sigmoid [0,1], Tanh [-1,1]).
    /// True for: Sigmoid, Tanh, SoftSign
    /// False for: ReLU, Identity, Softplus, ELU, SELU, Swish, GELU, LeakyReLU
    /// </summary>
    protected virtual bool IsBounded => false;

    /// <summary>
    /// Lower bound of the output range when IsBounded is true.
    /// Default: -1.0 (for tanh-like activations). Override for activations like
    /// Sigmoid (0.0) or ReLU6 (0.0).
    /// </summary>
    protected virtual double BoundLower => -1.0;

    /// <summary>
    /// Upper bound of the output range when IsBounded is true.
    /// Default: 1.0 (for tanh-like activations). Override for activations like
    /// ReLU6 (6.0).
    /// </summary>
    protected virtual double BoundUpper => 1.0;

    /// <summary>
    /// Whether the activation uses randomness during training (e.g., RReLU).
    /// When true, the test helper sets the activation to inference mode (deterministic)
    /// before running invariant tests. If the activation doesn't support inference mode,
    /// determinism and gradient tests are skipped.
    /// </summary>
    protected virtual bool IsStochastic => false;

    /// <summary>
    /// Creates an activation and sets it to inference mode if it's stochastic.
    /// This ensures deterministic behavior for gradient checks and consistency tests.
    /// </summary>
    protected IActivationFunction<T> CreateTestActivation()
    {
        var fn = CreateActivation();
        if (IsStochastic)
        {
            // Use reflection to call SetTrainingMode(false) if available
            var method = fn.GetType().GetMethod("SetTrainingMode", new[] { typeof(bool) });
            method?.Invoke(fn, new object[] { false });
        }
        return fn;
    }

    // =========================================================================
    // INVARIANT 1: Scalar Activate produces finite output for normal inputs
    // =========================================================================

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    [InlineData(0.5)]
    [InlineData(-0.5)]
    [InlineData(5.0)]
    [InlineData(-5.0)]
    public void Activate_ShouldProduceFiniteOutput(double input)
    {
        var fn = CreateActivation();
        double output = ToD(fn.Activate(ToT(input)));

        Assert.False(double.IsNaN(output), $"Activate({input}) returned NaN.");
        Assert.False(double.IsInfinity(output), $"Activate({input}) returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 2: Derivative produces finite output
    // =========================================================================

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    [InlineData(0.5)]
    [InlineData(-0.5)]
    public void Derivative_ShouldProduceFiniteOutput(double input)
    {
        var fn = CreateActivation();
        double deriv = ToD(fn.Derivative(ToT(input)));

        Assert.False(double.IsNaN(deriv), $"Derivative({input}) returned NaN.");
        Assert.False(double.IsInfinity(deriv), $"Derivative({input}) returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 3: Numerical gradient check
    // The analytical derivative should match the finite-difference approximation.
    // This is the most important test — it catches wrong gradient formulas.
    // =========================================================================

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(-0.3)]
    [InlineData(1.5)]
    [InlineData(-1.5)]
    public void Derivative_ShouldMatchNumericalGradient(double input)
    {
        var fn = CreateTestActivation();
        double epsilon = typeof(T) == typeof(float) ? 1e-3 : 1e-5;
        T center = ToT(input);
        T plus = ToT(input + epsilon);
        T minus = ToT(input - epsilon);
        double plusValue = ToD(plus);
        double minusValue = ToD(minus);

        double analyticalDeriv = ToD(fn.Derivative(center));
        double numericalDeriv =
            (ToD(fn.Activate(plus)) - ToD(fn.Activate(minus))) /
            (plusValue - minusValue);

        double absMax = Math.Max(Math.Abs(analyticalDeriv), Math.Abs(numericalDeriv));
        if (absMax < 1e-7) return; // Both near zero, skip

        double relError = Math.Abs(analyticalDeriv - numericalDeriv) / (absMax + 1e-8);
        Assert.True(relError < 0.01,
            $"Derivative({input}): analytical={analyticalDeriv:G10}, numerical={numericalDeriv:G10}, " +
            $"relError={relError:G6}. Gradient formula may be wrong.");
    }

    // =========================================================================
    // INVARIANT 4: Activate(0) == 0 for zero-preserving activations
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Activate_ZeroInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!ZeroMapsToZero) return;

        var fn = CreateActivation();
        double output = ToD(fn.Activate(ToT(0.0)));
        Assert.True(Math.Abs(output) < Tolerance,
            $"Expected Activate(0) ≈ 0 but got {output}.");
    }

    // =========================================================================
    // INVARIANT 5: Monotonicity — for monotonic activations, larger input → larger output
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Activate_ShouldBeMonotonic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsMonotonic) return;

        var fn = CreateTestActivation();
        double prev = ToD(fn.Activate(ToT(-10.0)));
        for (double x = -9.0; x <= 10.0; x += 0.5)
        {
            double curr = ToD(fn.Activate(ToT(x)));
            Assert.True(curr >= prev - Tolerance,
                $"Monotonicity violated: f({x - 0.5})={prev} > f({x})={curr}.");
            prev = curr;
        }
    }

    // =========================================================================
    // INVARIANT 6: Bounded activations stay within bounds
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Activate_ShouldRespectBounds()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsBounded) return;

        var fn = CreateTestActivation();
        double margin = 0.1; // small margin for numerical precision
        for (double x = -20.0; x <= 20.0; x += 0.5)
        {
            double y = ToD(fn.Activate(ToT(x)));
            Assert.True(y >= BoundLower - margin && y <= BoundUpper + margin,
                $"Bounded activation produced out-of-range value: f({x})={y}, " +
                $"expected [{BoundLower}, {BoundUpper}].");
        }
    }

    // =========================================================================
    // INVARIANT 7: Large input stability — no NaN/Inf for extreme values
    // =========================================================================

    [Theory]
    [InlineData(100.0)]
    [InlineData(-100.0)]
    [InlineData(1000.0)]
    [InlineData(-1000.0)]
    public void Activate_LargeInput_ShouldBeStable(double input)
    {
        var fn = CreateActivation();
        double output = ToD(fn.Activate(ToT(input)));

        Assert.False(double.IsNaN(output), $"Activate({input}) returned NaN — overflow.");
        Assert.False(double.IsInfinity(output), $"Activate({input}) returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 8: Tensor-level Activate matches scalar Activate
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task TensorActivate_ShouldMatchScalarActivate()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var fn = CreateTestActivation();
        var input = new Tensor<T>([5]);
        var rng = new Random(42);
        for (int i = 0; i < 5; i++)
            input[i] = ToT(rng.NextDouble() * 4.0 - 2.0); // [-2, 2]

        var tensorOutput = fn.Activate(input);

        for (int i = 0; i < 5; i++)
        {
            double scalarOutput = ToD(fn.Activate(input[i]));
            double tensorValue = ToD(tensorOutput[i]);
            Assert.True(Math.Abs(tensorValue - scalarOutput) < Tolerance,
                $"Tensor Activate[{i}]={tensorValue} != scalar Activate({ToD(input[i])})={scalarOutput}.");
        }
    }

    // =========================================================================
    // INVARIANT 9: Derivative is non-negative for non-decreasing activations
    // For monotonically non-decreasing activations, f'(x) >= 0 everywhere.
    // =========================================================================

    [Fact(Timeout = 30000)]
    public async Task Derivative_ShouldBeNonNegativeForMonotonicActivation()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        if (!IsMonotonic) return;

        var fn = CreateActivation();
        for (double x = -5.0; x <= 5.0; x += 0.25)
        {
            double deriv = ToD(fn.Derivative(ToT(x)));
            Assert.True(deriv >= -Tolerance,
                $"Monotonic activation has negative derivative: f'({x})={deriv}.");
        }
    }
}

/// <summary>Default-precision alias for existing hand-written fixtures.</summary>
public abstract class ActivationFunctionTestBase : ActivationFunctionTestBase<double> { }
