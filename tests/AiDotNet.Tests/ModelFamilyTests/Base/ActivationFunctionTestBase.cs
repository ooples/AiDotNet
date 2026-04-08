using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for IActivationFunction&lt;double&gt; implementations.
/// Tests mathematical invariants that every activation function must satisfy:
/// finite output, derivative correctness (numerical gradient check),
/// monotonicity properties, and edge case handling.
/// </summary>
public abstract class ActivationFunctionTestBase
{
    protected abstract IActivationFunction<double> CreateActivation();

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
    protected IActivationFunction<double> CreateTestActivation()
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
        double output = fn.Activate(input);

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
        double deriv = fn.Derivative(input);

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
        double epsilon = 1e-5;

        double analyticalDeriv = fn.Derivative(input);
        double numericalDeriv = (fn.Activate(input + epsilon) - fn.Activate(input - epsilon)) / (2 * epsilon);

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

    [Fact]
    public void Activate_ZeroInput()
    {
        if (!ZeroMapsToZero) return;

        var fn = CreateActivation();
        double output = fn.Activate(0.0);
        Assert.True(Math.Abs(output) < 1e-10,
            $"Expected Activate(0) ≈ 0 but got {output}.");
    }

    // =========================================================================
    // INVARIANT 5: Monotonicity — for monotonic activations, larger input → larger output
    // =========================================================================

    [Fact]
    public void Activate_ShouldBeMonotonic()
    {
        if (!IsMonotonic) return;

        var fn = CreateTestActivation();
        double prev = fn.Activate(-10.0);
        for (double x = -9.0; x <= 10.0; x += 0.5)
        {
            double curr = fn.Activate(x);
            Assert.True(curr >= prev - 1e-10,
                $"Monotonicity violated: f({x - 0.5})={prev} > f({x})={curr}.");
            prev = curr;
        }
    }

    // =========================================================================
    // INVARIANT 6: Bounded activations stay within bounds
    // =========================================================================

    [Fact]
    public void Activate_ShouldRespectBounds()
    {
        if (!IsBounded) return;

        var fn = CreateTestActivation();
        double margin = 0.1; // small margin for numerical precision
        for (double x = -20.0; x <= 20.0; x += 0.5)
        {
            double y = fn.Activate(x);
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
        double output = fn.Activate(input);

        Assert.False(double.IsNaN(output), $"Activate({input}) returned NaN — overflow.");
        Assert.False(double.IsInfinity(output), $"Activate({input}) returned Infinity.");
    }

    // =========================================================================
    // INVARIANT 8: Tensor-level Activate matches scalar Activate
    // =========================================================================

    [Fact]
    public void TensorActivate_ShouldMatchScalarActivate()
    {
        var fn = CreateTestActivation();
        var input = new Tensor<double>([5]);
        var rng = new Random(42);
        for (int i = 0; i < 5; i++)
            input[i] = rng.NextDouble() * 4.0 - 2.0; // [-2, 2]

        var tensorOutput = fn.Activate(input);

        for (int i = 0; i < 5; i++)
        {
            double scalarOutput = fn.Activate(input[i]);
            Assert.True(Math.Abs(tensorOutput[i] - scalarOutput) < 1e-12,
                $"Tensor Activate[{i}]={tensorOutput[i]} != scalar Activate({input[i]})={scalarOutput}.");
        }
    }

    // =========================================================================
    // INVARIANT 9: Derivative is non-negative for non-decreasing activations
    // For monotonically non-decreasing activations, f'(x) >= 0 everywhere.
    // =========================================================================

    [Fact]
    public void Derivative_ShouldBeNonNegativeForMonotonicActivation()
    {
        if (!IsMonotonic) return;

        var fn = CreateActivation();
        for (double x = -5.0; x <= 5.0; x += 0.25)
        {
            double deriv = fn.Derivative(x);
            Assert.True(deriv >= -1e-10,
                $"Monotonic activation has negative derivative: f'({x})={deriv}.");
        }
    }
}
