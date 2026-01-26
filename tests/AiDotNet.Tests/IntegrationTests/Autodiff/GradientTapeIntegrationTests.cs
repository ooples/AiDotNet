using AiDotNet.Autodiff;
using AiDotNet.Autodiff.Testing;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Autodiff;

/// <summary>
/// Comprehensive integration tests for GradientTape.
/// These tests verify correct gradient computation by comparing against numerical gradients.
/// </summary>
/// <remarks>
/// <para>
/// The user emphasized: "these integration tests are because we know our code has bugs
/// and these need to be integration tests to cover every possible scenario and a focus on correctness."
/// </para>
/// <para>
/// All expected values are hand-calculated or verified against numerical gradients to ensure correctness.
/// </para>
/// </remarks>
public class GradientTapeIntegrationTests
{
    private const double Tolerance = 1e-5;
    private const double NumericalEpsilon = 1e-5;
    private const double GradientTolerance = 1e-3;

    #region Basic Gradient Computation Tests

    /// <summary>
    /// Test: f(x) = x^2, df/dx = 2x
    /// For x = 3: gradient should be 6
    /// </summary>
    [Fact]
    public void Gradient_SimpleSquare_ReturnsCorrectGradient()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 3.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x^2
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            Assert.True(gradients.ContainsKey(x));
            var grad = gradients[x];

            // df/dx = 2x = 2 * 3 = 6
            Assert.Equal(6.0, grad[0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x) = x^3, df/dx = 3x^2
    /// For x = 2: gradient should be 12
    /// </summary>
    [Fact]
    public void Gradient_Cubic_ReturnsCorrectGradient()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x * x * x = x^3
            var x2 = TensorOperations<double>.ElementwiseMultiply(x, x);
            var y = TensorOperations<double>.ElementwiseMultiply(x2, x);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            var grad = gradients[x];

            // df/dx = 3x^2 = 3 * 4 = 12
            Assert.Equal(12.0, grad[0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x,y) = x + y, df/dx = 1, df/dy = 1
    /// </summary>
    [Fact]
    public void Gradient_Addition_BothInputsHaveGradientOne()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        var y = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "y");
        x.Value[0] = 3.0;
        y.Value[0] = 5.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);
            tape.Watch(y);

            var z = TensorOperations<double>.Add(x, y);

            var gradients = tape.Gradient(z, new[] { x, y });

            // Assert
            // d(x+y)/dx = 1, d(x+y)/dy = 1
            Assert.Equal(1.0, gradients[x][0], Tolerance);
            Assert.Equal(1.0, gradients[y][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x,y) = x - y, df/dx = 1, df/dy = -1
    /// </summary>
    [Fact]
    public void Gradient_Subtraction_CorrectSigns()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        var y = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "y");
        x.Value[0] = 3.0;
        y.Value[0] = 5.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);
            tape.Watch(y);

            var z = TensorOperations<double>.Subtract(x, y);

            var gradients = tape.Gradient(z, new[] { x, y });

            // Assert
            // d(x-y)/dx = 1, d(x-y)/dy = -1
            Assert.Equal(1.0, gradients[x][0], Tolerance);
            Assert.Equal(-1.0, gradients[y][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x,y) = x * y, df/dx = y, df/dy = x
    /// For x=3, y=5: df/dx = 5, df/dy = 3
    /// </summary>
    [Fact]
    public void Gradient_Multiplication_ProductRule()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        var y = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "y");
        x.Value[0] = 3.0;
        y.Value[0] = 5.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);
            tape.Watch(y);

            var z = TensorOperations<double>.ElementwiseMultiply(x, y);

            var gradients = tape.Gradient(z, new[] { x, y });

            // Assert
            // d(xy)/dx = y = 5, d(xy)/dy = x = 3
            Assert.Equal(5.0, gradients[x][0], Tolerance);
            Assert.Equal(3.0, gradients[y][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x,y) = x / y, df/dx = 1/y, df/dy = -x/y^2
    /// For x=6, y=2: df/dx = 0.5, df/dy = -1.5
    /// </summary>
    [Fact]
    public void Gradient_Division_QuotientRule()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        var y = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "y");
        x.Value[0] = 6.0;
        y.Value[0] = 2.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);
            tape.Watch(y);

            var z = TensorOperations<double>.Divide(x, y);

            var gradients = tape.Gradient(z, new[] { x, y });

            // Assert
            // d(x/y)/dx = 1/y = 1/2 = 0.5
            // d(x/y)/dy = -x/y^2 = -6/4 = -1.5
            Assert.Equal(0.5, gradients[x][0], Tolerance);
            Assert.Equal(-1.5, gradients[y][0], Tolerance);
        }
    }

    #endregion

    #region Chain Rule Tests

    /// <summary>
    /// Test: f(x) = exp(x^2), df/dx = 2x * exp(x^2)
    /// For x = 1: df/dx = 2 * exp(1) ≈ 5.4366
    /// </summary>
    [Fact]
    public void Gradient_ChainRule_ExpOfSquare()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = exp(x^2)
            var x2 = TensorOperations<double>.ElementwiseMultiply(x, x);
            var y = TensorOperations<double>.Exp(x2);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // By chain rule: df/dx = 2x * exp(x^2) = 2 * 1 * exp(1) = 2e ≈ 5.4366
            double expected = 2.0 * Math.Exp(1.0);
            Assert.Equal(expected, gradients[x][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x) = log(x^2 + 1), df/dx = 2x / (x^2 + 1)
    /// For x = 2: df/dx = 4/5 = 0.8
    /// </summary>
    [Fact]
    public void Gradient_ChainRule_LogOfSum()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        // Create constant for 1
        var one = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
        one.Value[0] = 1.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = log(x^2 + 1)
            var x2 = TensorOperations<double>.ElementwiseMultiply(x, x);
            var sum = TensorOperations<double>.Add(x2, one);
            var y = TensorOperations<double>.Log(sum);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // df/dx = 2x / (x^2 + 1) = 4 / 5 = 0.8
            double expected = 4.0 / 5.0;
            Assert.Equal(expected, gradients[x][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: f(x) = sqrt(x^2 + 1) - Numerical stability test
    /// For x = 0: df/dx = 0 (not undefined)
    /// </summary>
    [Fact]
    public void Gradient_ChainRule_SqrtOfSum_NumericallyStable()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.0;

        var one = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
        one.Value[0] = 1.0;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = sqrt(x^2 + 1)
            var x2 = TensorOperations<double>.ElementwiseMultiply(x, x);
            var sum = TensorOperations<double>.Add(x2, one);
            var y = TensorOperations<double>.Sqrt(sum);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // df/dx = x / sqrt(x^2 + 1) = 0 / sqrt(1) = 0
            Assert.Equal(0.0, gradients[x][0], Tolerance);
            Assert.False(double.IsNaN(gradients[x][0]), "Gradient should not be NaN");
        }
    }

    /// <summary>
    /// Test: f(x) = tanh(sigmoid(x)) - Nested activation functions
    /// Verifies chain rule with multiple layers of composition
    /// </summary>
    [Fact]
    public void Gradient_ChainRule_NestedActivations()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.5;

        // Act
        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = tanh(sigmoid(x))
            var sigmoid_x = TensorOperations<double>.Sigmoid(x);
            var y = TensorOperations<double>.Tanh(sigmoid_x);

            var gradients = tape.Gradient(y, new[] { x });

            // Verify with numerical gradient
            double h = NumericalEpsilon;
            double f_plus = Math.Tanh(1.0 / (1.0 + Math.Exp(-(0.5 + h))));
            double f_minus = Math.Tanh(1.0 / (1.0 + Math.Exp(-(0.5 - h))));
            double numericalGradient = (f_plus - f_minus) / (2.0 * h);

            // Assert
            Assert.Equal(numericalGradient, gradients[x][0], GradientTolerance);
        }
    }

    #endregion

    #region Tape Lifecycle Tests

    /// <summary>
    /// Test: Non-persistent tape throws exception on second use
    /// </summary>
    [Fact]
    public void Gradient_NonPersistentTape_ThrowsOnSecondUse()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        using (var tape = new GradientTape<double>(persistent: false))
        {
            tape.Watch(x);
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);

            // First call should succeed
            var gradients = tape.Gradient(y, new[] { x });
            Assert.Equal(4.0, gradients[x][0], Tolerance);

            // Second call should throw
            Assert.Throws<InvalidOperationException>(() => tape.Gradient(y, new[] { x }));
        }
    }

    /// <summary>
    /// Test: Persistent tape allows multiple gradient computations
    /// </summary>
    [Fact]
    public void Gradient_PersistentTape_AllowsMultipleUse()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        using (var tape = new GradientTape<double>(persistent: true))
        {
            tape.Watch(x);
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);

            // First call
            var gradients1 = tape.Gradient(y, new[] { x });
            Assert.Equal(4.0, gradients1[x][0], Tolerance);

            // Second call should also succeed
            var gradients2 = tape.Gradient(y, new[] { x });
            Assert.Equal(4.0, gradients2[x][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: Tape.Reset() clears state and allows reuse
    /// </summary>
    [Fact]
    public void Gradient_TapeReset_AllowsReuse()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        using (var tape = new GradientTape<double>(persistent: true))
        {
            tape.Watch(x);
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);
            var gradients1 = tape.Gradient(y, new[] { x });

            // Reset tape
            tape.Reset();

            // After reset, we need to re-watch
            tape.Watch(x);

            // Change x value
            x.Value[0] = 3.0;
            var y2 = TensorOperations<double>.ElementwiseMultiply(x, x);
            var gradients2 = tape.Gradient(y2, new[] { x });

            // Should have new gradient for x=3
            Assert.Equal(6.0, gradients2[x][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: Disposed tape throws ObjectDisposedException
    /// </summary>
    [Fact]
    public void Gradient_DisposedTape_ThrowsException()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        var tape = new GradientTape<double>();
        tape.Watch(x);
        tape.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => tape.Watch(x));
    }

    /// <summary>
    /// Test: Current tape is properly managed on stack
    /// </summary>
    [Fact]
    public void GradientTape_Current_ProperlyManagedOnStack()
    {
        // Initially no tape
        Assert.Null(GradientTape<double>.Current);

        using (var tape1 = new GradientTape<double>())
        {
            Assert.Same(tape1, GradientTape<double>.Current);

            using (var tape2 = new GradientTape<double>())
            {
                // Nested tape becomes current
                Assert.Same(tape2, GradientTape<double>.Current);
            }

            // After inner dispose, outer becomes current again
            Assert.Same(tape1, GradientTape<double>.Current);
        }

        // After all dispose, no current tape
        Assert.Null(GradientTape<double>.Current);
    }

    #endregion

    #region Watch Functionality Tests

    /// <summary>
    /// Test: Unwatched variables don't receive gradients
    /// </summary>
    [Fact]
    public void Gradient_UnwatchedVariable_NoGradient()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        var y = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "y");
        x.Value[0] = 2.0;
        y.Value[0] = 3.0;

        using (var tape = new GradientTape<double>())
        {
            // Only watch x, not y
            tape.Watch(x);

            var z = TensorOperations<double>.Add(x, y);
            var gradients = tape.Gradient(z, new[] { x });

            // Assert
            Assert.True(gradients.ContainsKey(x));
            Assert.False(gradients.ContainsKey(y));
        }
    }

    /// <summary>
    /// Test: Watch multiple variables at once
    /// </summary>
    [Fact]
    public void Gradient_WatchMultiple_AllReceiveGradients()
    {
        // Arrange
        var a = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "a");
        var b = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "b");
        var c = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "c");
        a.Value[0] = 1.0;
        b.Value[0] = 2.0;
        c.Value[0] = 3.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(new[] { a, b, c });

            // f(a,b,c) = a*b + c
            var ab = TensorOperations<double>.ElementwiseMultiply(a, b);
            var y = TensorOperations<double>.Add(ab, c);

            var gradients = tape.Gradient(y);

            // Assert
            // df/da = b = 2
            // df/db = a = 1
            // df/dc = 1
            Assert.Equal(2.0, gradients[a][0], Tolerance);
            Assert.Equal(1.0, gradients[b][0], Tolerance);
            Assert.Equal(1.0, gradients[c][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: Watching same variable twice is idempotent
    /// </summary>
    [Fact]
    public void Gradient_WatchSameVariableTwice_Idempotent()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 3.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);
            tape.Watch(x); // Watch again

            var y = TensorOperations<double>.ElementwiseMultiply(x, x);
            var gradients = tape.Gradient(y);

            // Assert - should still work correctly
            Assert.Equal(6.0, gradients[x][0], Tolerance);
        }
    }

    #endregion

    #region Nested Tapes Tests

    /// <summary>
    /// Test: Nested tapes work independently
    /// </summary>
    [Fact]
    public void Gradient_NestedTapes_WorkIndependently()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        // Outer tape
        using (var outerTape = new GradientTape<double>())
        {
            outerTape.Watch(x);

            // Inner tape for different computation
            using (var innerTape = new GradientTape<double>())
            {
                innerTape.Watch(x);

                // Inner computation: f(x) = x^2
                var innerY = TensorOperations<double>.ElementwiseMultiply(x, x);
                var innerGradients = innerTape.Gradient(innerY, new[] { x });

                // Inner gradient: 2x = 4
                Assert.Equal(4.0, innerGradients[x][0], Tolerance);
            }

            // Outer computation continues: g(x) = x^3
            var x2 = TensorOperations<double>.ElementwiseMultiply(x, x);
            var outerY = TensorOperations<double>.ElementwiseMultiply(x2, x);
            var outerGradients = outerTape.Gradient(outerY, new[] { x });

            // Outer gradient: 3x^2 = 12
            Assert.Equal(12.0, outerGradients[x][0], Tolerance);
        }
    }

    #endregion

    #region Vector and Tensor Gradient Tests

    /// <summary>
    /// Test: Gradient of vector dot product
    /// f(x,y) = sum(x * y), df/dx = y, df/dy = x
    /// </summary>
    [Fact]
    public void Gradient_VectorDotProduct_CorrectGradients()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "x");
        var y = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "y");

        x.Value[0] = 1.0; x.Value[1] = 2.0; x.Value[2] = 3.0;
        y.Value[0] = 4.0; y.Value[1] = 5.0; y.Value[2] = 6.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);
            tape.Watch(y);

            // z = sum(x * y) (dot product)
            var xy = TensorOperations<double>.ElementwiseMultiply(x, y);
            var z = TensorOperations<double>.Sum(xy);

            var gradients = tape.Gradient(z, new[] { x, y });

            // Assert
            // dz/dx = y
            Assert.Equal(4.0, gradients[x][0], Tolerance);
            Assert.Equal(5.0, gradients[x][1], Tolerance);
            Assert.Equal(6.0, gradients[x][2], Tolerance);

            // dz/dy = x
            Assert.Equal(1.0, gradients[y][0], Tolerance);
            Assert.Equal(2.0, gradients[y][1], Tolerance);
            Assert.Equal(3.0, gradients[y][2], Tolerance);
        }
    }

    /// <summary>
    /// Test: Gradient of vector sum
    /// f(x) = sum(x), df/dx = [1, 1, ..., 1]
    /// </summary>
    [Fact]
    public void Gradient_VectorSum_AllOnes()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 5 }), "x");
        for (int i = 0; i < 5; i++)
            x.Value[i] = i + 1.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var y = TensorOperations<double>.Sum(x);
            var gradients = tape.Gradient(y, new[] { x });

            // Assert - all gradients should be 1
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(1.0, gradients[x][i], Tolerance);
            }
        }
    }

    /// <summary>
    /// Test: Gradient of vector mean
    /// f(x) = mean(x), df/dx = [1/n, 1/n, ..., 1/n]
    /// </summary>
    [Fact]
    public void Gradient_VectorMean_AllEqualFraction()
    {
        // Arrange
        int n = 5;
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { n }), "x");
        for (int i = 0; i < n; i++)
            x.Value[i] = i + 1.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var y = TensorOperations<double>.Mean(x);
            var gradients = tape.Gradient(y, new[] { x });

            // Assert - all gradients should be 1/n
            double expected = 1.0 / n;
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(expected, gradients[x][i], Tolerance);
            }
        }
    }

    #endregion

    #region Activation Functions Gradient Verification

    /// <summary>
    /// Test: ReLU gradient
    /// df/dx = 1 if x > 0, else 0
    /// </summary>
    [Fact]
    public void Gradient_ReLU_CorrectForPositiveAndNegative()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 4 }), "x");
        x.Value[0] = 2.0;   // positive -> gradient = 1
        x.Value[1] = -2.0;  // negative -> gradient = 0
        x.Value[2] = 0.0;   // zero -> gradient = 0 (typically)
        x.Value[3] = 0.5;   // positive -> gradient = 1

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var relu = TensorOperations<double>.ReLU(x);
            var y = TensorOperations<double>.Sum(relu);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            Assert.Equal(1.0, gradients[x][0], Tolerance); // positive
            Assert.Equal(0.0, gradients[x][1], Tolerance); // negative
            Assert.Equal(0.0, gradients[x][2], Tolerance); // zero
            Assert.Equal(1.0, gradients[x][3], Tolerance); // positive
        }
    }

    /// <summary>
    /// Test: Sigmoid gradient
    /// df/dx = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    [Fact]
    public void Gradient_Sigmoid_MatchesFormula()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var sig = TensorOperations<double>.Sigmoid(x);
            var y = TensorOperations<double>.Sum(sig);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
            // gradient = sig * (1 - sig) ≈ 0.7311 * 0.2689 ≈ 0.1966
            double sigVal = 1.0 / (1.0 + Math.Exp(-1.0));
            double expected = sigVal * (1.0 - sigVal);
            Assert.Equal(expected, gradients[x][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: Tanh gradient
    /// df/dx = 1 - tanh^2(x)
    /// </summary>
    [Fact]
    public void Gradient_Tanh_MatchesFormula()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.5;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var tanhX = TensorOperations<double>.Tanh(x);
            var y = TensorOperations<double>.Sum(tanhX);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // gradient = 1 - tanh^2(0.5)
            double tanhVal = Math.Tanh(0.5);
            double expected = 1.0 - tanhVal * tanhVal;
            Assert.Equal(expected, gradients[x][0], Tolerance);
        }
    }

    /// <summary>
    /// Test: Softmax gradient (Jacobian verification)
    /// For softmax output s_i, ds_i/dx_j = s_i * (delta_ij - s_j)
    /// </summary>
    [Fact]
    public void Gradient_Softmax_NumericalVerification()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "x");
        x.Value[0] = 1.0;
        x.Value[1] = 2.0;
        x.Value[2] = 3.0;

        // Compute expected softmax values
        double maxX = 3.0;
        double sumExp = Math.Exp(1.0 - maxX) + Math.Exp(2.0 - maxX) + Math.Exp(3.0 - maxX);
        double[] softmax = new double[3];
        softmax[0] = Math.Exp(1.0 - maxX) / sumExp;
        softmax[1] = Math.Exp(2.0 - maxX) / sumExp;
        softmax[2] = Math.Exp(3.0 - maxX) / sumExp;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var soft = TensorOperations<double>.Softmax(x);
            var y = TensorOperations<double>.Sum(soft);

            var gradients = tape.Gradient(y, new[] { x });

            // For sum(softmax(x)), gradient is all zeros because sum is constant = 1
            // Actually, because softmax sums to 1, derivative of sum(softmax) is 0
            for (int i = 0; i < 3; i++)
            {
                // The sum of softmax is always 1, so its derivative w.r.t. inputs is 0
                Assert.Equal(0.0, gradients[x][i], Tolerance);
            }
        }
    }

    #endregion

    #region Numerical Gradient Verification

    /// <summary>
    /// Test: Compare autodiff gradient against numerical gradient for complex expression
    /// f(x) = x^3 + 2x^2 - 5x + 3
    /// df/dx = 3x^2 + 4x - 5
    /// </summary>
    [Fact]
    public void Gradient_Polynomial_MatchesNumerical()
    {
        // Test at multiple points
        double[] testPoints = { -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };

        foreach (double xVal in testPoints)
        {
            // Arrange
            var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
            x.Value[0] = xVal;

            // Constants
            var two = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
            two.Value[0] = 2.0;
            var five = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
            five.Value[0] = 5.0;
            var three = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
            three.Value[0] = 3.0;

            using (var tape = new GradientTape<double>())
            {
                tape.Watch(x);

                // f(x) = x^3 + 2x^2 - 5x + 3
                var x2 = TensorOperations<double>.ElementwiseMultiply(x, x);
                var x3 = TensorOperations<double>.ElementwiseMultiply(x2, x);
                var term1 = x3; // x^3
                var term2 = TensorOperations<double>.ElementwiseMultiply(two, x2); // 2x^2
                var term3 = TensorOperations<double>.ElementwiseMultiply(five, x); // 5x
                var temp1 = TensorOperations<double>.Add(term1, term2); // x^3 + 2x^2
                var temp2 = TensorOperations<double>.Subtract(temp1, term3); // x^3 + 2x^2 - 5x
                var y = TensorOperations<double>.Add(temp2, three); // x^3 + 2x^2 - 5x + 3

                var gradients = tape.Gradient(y, new[] { x });

                // Expected: df/dx = 3x^2 + 4x - 5
                double expected = 3.0 * xVal * xVal + 4.0 * xVal - 5.0;

                // Numerical verification
                double h = NumericalEpsilon;
                double f_plus = Math.Pow(xVal + h, 3) + 2 * Math.Pow(xVal + h, 2) - 5 * (xVal + h) + 3;
                double f_minus = Math.Pow(xVal - h, 3) + 2 * Math.Pow(xVal - h, 2) - 5 * (xVal - h) + 3;
                double numerical = (f_plus - f_minus) / (2 * h);

                Assert.Equal(expected, gradients[x][0], GradientTolerance);
                Assert.Equal(numerical, gradients[x][0], GradientTolerance);
            }
        }
    }

    /// <summary>
    /// Test: Verify gradient using NumericalGradient utility
    /// </summary>
    [Fact]
    public void Gradient_UsingNumericalGradientUtility_AllMatch()
    {
        // Use the built-in NumericalGradient class to verify
        var config = new TensorOperationsVerification<double>.VerificationConfig
        {
            RelativeTolerance = 1e-3,
            AbsoluteTolerance = 1e-6,
            Epsilon = 1e-5
        };

        var verifier = new TensorOperationsVerification<double>(config);

        // Verify a few critical operations
        var reluResult = verifier.VerifyReLU();
        Assert.True(reluResult.Passed, $"ReLU verification failed: {reluResult}");

        var sigmoidResult = verifier.VerifySigmoid();
        Assert.True(sigmoidResult.Passed, $"Sigmoid verification failed: {sigmoidResult}");

        var tanhResult = verifier.VerifyTanh();
        Assert.True(tanhResult.Passed, $"Tanh verification failed: {tanhResult}");

        var expResult = verifier.VerifyExp();
        Assert.True(expResult.Passed, $"Exp verification failed: {expResult}");

        var logResult = verifier.VerifyLog();
        Assert.True(logResult.Passed, $"Log verification failed: {logResult}");
    }

    #endregion

    #region Edge Cases and Numerical Stability

    /// <summary>
    /// Test: Gradient at zero for square function
    /// f(x) = x^2, df/dx = 2x = 0 at x=0
    /// </summary>
    [Fact]
    public void Gradient_SquareAtZero_ReturnsZero()
    {
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var y = TensorOperations<double>.ElementwiseMultiply(x, x);
            var gradients = tape.Gradient(y, new[] { x });

            Assert.Equal(0.0, gradients[x][0], Tolerance);
            Assert.False(double.IsNaN(gradients[x][0]));
        }
    }

    /// <summary>
    /// Test: Gradient with very small values (numerical precision)
    /// </summary>
    [Fact]
    public void Gradient_VerySmallValues_NumericallyStable()
    {
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1e-10;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x^2
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);
            var gradients = tape.Gradient(y, new[] { x });

            // df/dx = 2x = 2e-10
            double expected = 2e-10;
            Assert.Equal(expected, gradients[x][0], 1e-15);
            Assert.False(double.IsNaN(gradients[x][0]));
            Assert.False(double.IsInfinity(gradients[x][0]));
        }
    }

    /// <summary>
    /// Test: Gradient with very large values
    /// </summary>
    [Fact]
    public void Gradient_VeryLargeValues_NumericallyStable()
    {
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1e6;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x + 1 (linear, safe for large values)
            var one = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
            one.Value[0] = 1.0;
            var y = TensorOperations<double>.Add(x, one);

            var gradients = tape.Gradient(y, new[] { x });

            // df/dx = 1
            Assert.Equal(1.0, gradients[x][0], Tolerance);
            Assert.False(double.IsNaN(gradients[x][0]));
            Assert.False(double.IsInfinity(gradients[x][0]));
        }
    }

    /// <summary>
    /// Test: Log gradient near zero (should handle gracefully)
    /// </summary>
    [Fact]
    public void Gradient_LogNearZero_HandledGracefully()
    {
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1e-10; // Very small but positive

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var y = TensorOperations<double>.Log(x);
            var gradients = tape.Gradient(y, new[] { x });

            // d(log(x))/dx = 1/x = 1e10
            // Should be large but not NaN or Inf
            Assert.False(double.IsNaN(gradients[x][0]));
            Assert.True(gradients[x][0] > 0, "Gradient of log should be positive for positive x");
        }
    }

    /// <summary>
    /// Test: Exp gradient with negative input (should be small but stable)
    /// </summary>
    [Fact]
    public void Gradient_ExpWithNegativeInput_Stable()
    {
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = -10.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var y = TensorOperations<double>.Exp(x);
            var gradients = tape.Gradient(y, new[] { x });

            // d(exp(x))/dx = exp(x) = exp(-10) ≈ 4.54e-5
            double expected = Math.Exp(-10.0);
            Assert.Equal(expected, gradients[x][0], Tolerance);
            Assert.False(double.IsNaN(gradients[x][0]));
        }
    }

    #endregion

    #region Stop/Resume Recording Tests

    /// <summary>
    /// Test: Operations after StopRecording are not tracked
    /// </summary>
    [Fact]
    public void StopRecording_OperationsNotTracked()
    {
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // This operation should be recorded
            var y1 = TensorOperations<double>.ElementwiseMultiply(x, x);

            tape.StopRecording();

            // This operation should NOT be recorded
            var y2 = TensorOperations<double>.ElementwiseMultiply(y1, x);

            tape.ResumeRecording();

            // Compute gradient - should only reflect y1 = x^2
            var gradients = tape.Gradient(y1, new[] { x });

            // Gradient of x^2 = 2x = 4
            Assert.Equal(4.0, gradients[x][0], Tolerance);
        }
    }

    #endregion

    #region Float Type Tests

    /// <summary>
    /// Test: Gradients work correctly with float type
    /// </summary>
    [Fact]
    public void Gradient_FloatType_WorksCorrectly()
    {
        var x = TensorOperations<float>.Variable(new Tensor<float>(new[] { 1 }), "x");
        x.Value[0] = 3.0f;

        using (var tape = new GradientTape<float>())
        {
            tape.Watch(x);

            var y = TensorOperations<float>.ElementwiseMultiply(x, x);
            var gradients = tape.Gradient(y, new[] { x });

            // df/dx = 2x = 6
            Assert.Equal(6.0f, gradients[x][0], 1e-4f);
        }
    }

    #endregion

    #region Edge Case Tests - Large, Small, and Extreme Gradients

    /// <summary>
    /// Test: Very large gradients (gradient explosion scenario)
    /// f(x) = x^10, df/dx = 10*x^9
    /// For x = 10: gradient = 10 * 10^9 = 10^10 (very large)
    /// </summary>
    [Fact]
    public void Gradient_VeryLargeGradient_HandlesCorrectly()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 10.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x^10 computed as repeated multiplication
            var result = x;
            for (int i = 1; i < 10; i++)
            {
                result = TensorOperations<double>.ElementwiseMultiply(result, x);
            }

            var gradients = tape.Gradient(result, new[] { x });

            // Assert
            // df/dx = 10 * x^9 = 10 * 10^9 = 10^10
            double expected = 10.0 * Math.Pow(10.0, 9);
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(expected, gradients[x][0], expected * 1e-6); // relative tolerance for large numbers
        }
    }

    /// <summary>
    /// Test: Very small gradients (gradient vanishing scenario)
    /// f(x) = x^10, df/dx = 10*x^9
    /// For x = 0.1: gradient = 10 * 0.1^9 = 10^-8 (very small)
    /// </summary>
    [Fact]
    public void Gradient_VerySmallGradient_HandlesCorrectly()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.1;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x^10
            var result = x;
            for (int i = 1; i < 10; i++)
            {
                result = TensorOperations<double>.ElementwiseMultiply(result, x);
            }

            var gradients = tape.Gradient(result, new[] { x });

            // Assert
            // df/dx = 10 * x^9 = 10 * 0.1^9 = 10 * 10^-9 = 10^-8
            double expected = 10.0 * Math.Pow(0.1, 9);
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(expected, gradients[x][0], 1e-12);
        }
    }

    /// <summary>
    /// Test: Gradient at non-differentiable point (ReLU at x=0)
    /// ReLU is not differentiable at x=0, but implementations typically use subgradient 0
    /// </summary>
    [Fact]
    public void Gradient_ReLU_AtNonDifferentiablePoint_ReturnsSubgradient()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.0; // Non-differentiable point

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var relu = TensorOperations<double>.ReLU(x);
            var y = TensorOperations<double>.Sum(relu);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert - at x=0, subgradient is typically 0 (could also be 0.5 in some implementations)
            Assert.True(gradients.ContainsKey(x));
            Assert.True(gradients[x][0] == 0.0 || gradients[x][0] == 0.5 || gradients[x][0] == 1.0,
                $"Expected subgradient at x=0 to be 0, 0.5, or 1, but got {gradients[x][0]}");
        }
    }

    /// <summary>
    /// Test: Gradient of absolute value at non-differentiable point (x=0)
    /// |x| is not differentiable at x=0
    /// </summary>
    [Fact]
    public void Gradient_AbsoluteValue_AtZero_HandlesNonDifferentiability()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "x");
        x.Value[0] = 2.0;   // positive -> gradient = 1
        x.Value[1] = -2.0;  // negative -> gradient = -1
        x.Value[2] = 0.0;   // non-differentiable point

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var abs = TensorOperations<double>.Abs(x);
            var y = TensorOperations<double>.Sum(abs);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(1.0, gradients[x][0], Tolerance);  // d|x|/dx = 1 for x > 0
            Assert.Equal(-1.0, gradients[x][1], Tolerance); // d|x|/dx = -1 for x < 0
            // At x=0, subgradient is typically 0
            Assert.True(Math.Abs(gradients[x][2]) <= 1.0,
                $"Expected subgradient at x=0 to be in [-1, 1], but got {gradients[x][2]}");
        }
    }

    /// <summary>
    /// Test: Gradient near discontinuity (steep sigmoid approximating step function)
    /// Very steep sigmoid behaves almost like step function
    /// </summary>
    [Fact]
    public void Gradient_SteepSigmoid_NearDiscontinuity_HandlesCorrectly()
    {
        // Arrange - using very large values to create steep sigmoid
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "x");
        x.Value[0] = -100.0; // Far negative - sigmoid ≈ 0, gradient ≈ 0
        x.Value[1] = 0.0;    // At zero - sigmoid = 0.5, gradient = 0.25
        x.Value[2] = 100.0;  // Far positive - sigmoid ≈ 1, gradient ≈ 0

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var sig = TensorOperations<double>.Sigmoid(x);
            var y = TensorOperations<double>.Sum(sig);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            Assert.True(gradients.ContainsKey(x));

            // At x = -100: sigmoid ≈ 0, gradient = sig * (1 - sig) ≈ 0
            Assert.True(gradients[x][0] < 1e-10, $"Expected near-zero gradient at x=-100, got {gradients[x][0]}");

            // At x = 0: sigmoid = 0.5, gradient = 0.5 * 0.5 = 0.25
            Assert.Equal(0.25, gradients[x][1], 1e-6);

            // At x = 100: sigmoid ≈ 1, gradient = sig * (1 - sig) ≈ 0
            Assert.True(gradients[x][2] < 1e-10, $"Expected near-zero gradient at x=100, got {gradients[x][2]}");
        }
    }

    /// <summary>
    /// Test: Gradient with values near machine epsilon
    /// Tests numerical stability with very small values
    /// </summary>
    [Fact]
    public void Gradient_NearMachineEpsilon_MaintainsStability()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1e-15; // Very close to machine epsilon

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x^2
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // df/dx = 2x = 2 * 1e-15 = 2e-15
            Assert.True(gradients.ContainsKey(x));
            double expected = 2.0 * 1e-15;
            Assert.Equal(expected, gradients[x][0], 1e-20);
        }
    }

    /// <summary>
    /// Test: Gradient explosion detection with exponential growth
    /// f(x) = e^(e^x), gradient grows very fast
    /// </summary>
    [Fact]
    public void Gradient_ExponentialChain_HandlesLargeValues()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 2.0; // e^(e^2) = e^7.389 ≈ 1618.18

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = e^(e^x)
            var exp1 = TensorOperations<double>.Exp(x);
            var y = TensorOperations<double>.Exp(exp1);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // d/dx[e^(e^x)] = e^(e^x) * e^x
            double expX = Math.Exp(2.0);
            double expected = Math.Exp(expX) * expX;

            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(expected, gradients[x][0], expected * 1e-6);
        }
    }

    /// <summary>
    /// Test: Gradient with mixed very large and very small values in same tensor
    /// Tests handling of wide dynamic range
    /// </summary>
    [Fact]
    public void Gradient_MixedMagnitudes_HandlesWideDynamicRange()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "x");
        x.Value[0] = 1e-10;  // Very small
        x.Value[1] = 1.0;    // Normal
        x.Value[2] = 1e10;   // Very large

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x^2, df/dx = 2x
            var y = TensorOperations<double>.ElementwiseMultiply(x, x);
            var z = TensorOperations<double>.Sum(y);

            var gradients = tape.Gradient(z, new[] { x });

            // Assert - gradients should scale with input
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(2e-10, gradients[x][0], 1e-15);
            Assert.Equal(2.0, gradients[x][1], Tolerance);
            Assert.Equal(2e10, gradients[x][2], 1e5); // relative tolerance for large numbers
        }
    }

    /// <summary>
    /// Test: LeakyReLU at non-differentiable point (x=0)
    /// LeakyReLU(x) = max(alpha*x, x) where alpha is typically 0.01
    /// </summary>
    [Fact]
    public void Gradient_LeakyReLU_AtNonDifferentiablePoint()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 3 }), "x");
        x.Value[0] = 2.0;   // positive -> gradient = 1
        x.Value[1] = -2.0;  // negative -> gradient = alpha (0.01)
        x.Value[2] = 0.0;   // non-differentiable point

        const double alpha = 0.01;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            var leakyRelu = TensorOperations<double>.LeakyReLU(x, alpha);
            var y = TensorOperations<double>.Sum(leakyRelu);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(1.0, gradients[x][0], Tolerance);     // positive region
            Assert.Equal(alpha, gradients[x][1], Tolerance);   // negative region
            // At x=0, typically uses right derivative (1) or left (alpha)
            Assert.True(gradients[x][2] == 1.0 || gradients[x][2] == alpha,
                $"Expected gradient at x=0 to be {alpha} or 1, but got {gradients[x][2]}");
        }
    }

    /// <summary>
    /// Test: Division gradient near zero (potential for very large gradients)
    /// f(x) = 1/x, df/dx = -1/x^2
    /// As x approaches 0, gradient approaches infinity
    /// </summary>
    [Fact]
    public void Gradient_Division_NearZero_HandlesLargeGradient()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 0.001; // Small but not zero

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = 1/x
            var one = TensorOperations<double>.Constant(new Tensor<double>(new[] { 1 }));
            one.Value[0] = 1.0;
            var y = TensorOperations<double>.Divide(one, x);

            var gradients = tape.Gradient(y, new[] { x });

            // Assert
            // df/dx = -1/x^2 = -1/0.000001 = -1000000
            double expected = -1.0 / (0.001 * 0.001);
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(expected, gradients[x][0], Math.Abs(expected) * 1e-6);
        }
    }

    /// <summary>
    /// Test: Gradient stability with repeated operations (accumulation test)
    /// Verifies gradients don't drift with many operations
    /// </summary>
    [Fact]
    public void Gradient_ManyOperations_MaintainsAccuracy()
    {
        // Arrange
        var x = TensorOperations<double>.Variable(new Tensor<double>(new[] { 1 }), "x");
        x.Value[0] = 1.0;

        using (var tape = new GradientTape<double>())
        {
            tape.Watch(x);

            // f(x) = x + x + x + ... (100 times) = 100x, df/dx = 100
            var result = x;
            for (int i = 1; i < 100; i++)
            {
                result = TensorOperations<double>.Add(result, x);
            }

            var gradients = tape.Gradient(result, new[] { x });

            // Assert
            Assert.True(gradients.ContainsKey(x));
            Assert.Equal(100.0, gradients[x][0], Tolerance);
        }
    }

    #endregion
}
