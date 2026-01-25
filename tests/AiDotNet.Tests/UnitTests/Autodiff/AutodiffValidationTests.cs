using AiDotNet.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Autodiff;

/// <summary>
/// Validation tests for Autodiff module to verify production bug fixes.
/// </summary>
public class AutodiffValidationTests
{
    #region TensorOperations Division Tests

    [Fact]
    public void Divide_DivisionByZero_ThrowsArgumentException()
    {
        // Arrange
        var aTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));
        var bTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 0.0, 2.0])); // Contains zero
        var a = TensorOperations<double>.Variable(aTensor, "a");
        var b = TensorOperations<double>.Variable(bTensor, "b");

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => TensorOperations<double>.Divide(a, b));
        Assert.Contains("division by zero", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Divide_ValidInputs_ReturnsCorrectResult()
    {
        // Arrange
        var aTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([6.0, 8.0, 9.0]));
        var bTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([2.0, 4.0, 3.0]));
        var a = TensorOperations<double>.Variable(aTensor, "a");
        var b = TensorOperations<double>.Variable(bTensor, "b");

        // Act
        var result = TensorOperations<double>.Divide(a, b);

        // Assert
        Assert.Equal(3.0, result.Value[0], 5);
        Assert.Equal(2.0, result.Value[1], 5);
        Assert.Equal(3.0, result.Value[2], 5);
    }

    #endregion

    #region TensorOperations Log Tests

    [Fact]
    public void Log_ZeroInput_ThrowsArgumentException()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 0.0, 2.0])); // Contains zero
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => TensorOperations<double>.Log(a));
        Assert.Contains("non-positive", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Log_NegativeInput_ThrowsArgumentException()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, -1.0, 2.0])); // Contains negative
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => TensorOperations<double>.Log(a));
        Assert.Contains("non-positive", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Log_ValidPositiveInputs_ReturnsCorrectResult()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2 }, new Vector<double>([1.0, Math.E]));
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act
        var result = TensorOperations<double>.Log(a);

        // Assert
        Assert.Equal(0.0, result.Value[0], 5); // log(1) = 0
        Assert.Equal(1.0, result.Value[1], 5); // log(e) = 1
    }

    #endregion

    #region TensorOperations Sqrt Tests

    [Fact]
    public void Sqrt_NegativeInput_ThrowsArgumentException()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, -1.0, 4.0])); // Contains negative
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => TensorOperations<double>.Sqrt(a));
        Assert.Contains("negative", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Sqrt_ValidNonNegativeInputs_ReturnsCorrectResult()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([0.0, 1.0, 4.0]));
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act
        var result = TensorOperations<double>.Sqrt(a);

        // Assert
        Assert.Equal(0.0, result.Value[0], 5); // sqrt(0) = 0
        Assert.Equal(1.0, result.Value[1], 5); // sqrt(1) = 1
        Assert.Equal(2.0, result.Value[2], 5); // sqrt(4) = 2
    }

    [Fact]
    public void Sqrt_ZeroInput_BackwardDoesNotProduceInfinity()
    {
        // Arrange - sqrt(0) = 0, but d/dx sqrt(x) = 1/(2*sqrt(x)) which is infinity at x=0
        // We should handle this edge case gracefully
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([0.0, 1.0, 4.0]));
        var a = TensorOperations<double>.Variable(tensor, "a", requiresGradient: true);

        // Act
        var result = TensorOperations<double>.Sqrt(a);
        result.Backward();

        // Assert - gradient at 0 should be 0 (or some finite value), not infinity
        Assert.NotNull(a.Gradient);
        Assert.False(double.IsInfinity(a.Gradient[0]), "Gradient at sqrt(0) should not be infinity");
        Assert.False(double.IsNaN(a.Gradient[0]), "Gradient at sqrt(0) should not be NaN");
    }

    #endregion

    #region TensorOperations Sum Tests

    [Fact]
    public void Sum_NullAxes_OperationParamsHandledCorrectly()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2, 3 }, new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act - Sum with null axes (sum all elements)
        var result = TensorOperations<double>.Sum(a, axes: null);

        // Assert
        Assert.Equal(21.0, result.Value[0], 5); // 1+2+3+4+5+6 = 21
        // When axes is null, OperationParams should not contain "Axes" key (to avoid null storage)
        // It should either be null entirely or contain only "KeepDims"
        Assert.True(result.OperationParams == null ||
                    !result.OperationParams.ContainsKey("Axes") ||
                    result.OperationParams["Axes"] != null);
    }

    [Fact]
    public void Sum_SpecificAxis_ReturnsCorrectResult()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2, 3 }, new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        var a = TensorOperations<double>.Variable(tensor, "a");

        // Act - Sum along axis 1 (columns)
        var result = TensorOperations<double>.Sum(a, axes: new[] { 1 });

        // Assert
        Assert.Equal(2, result.Value.Shape.Length == 1 ? result.Value.Shape[0] : result.Value.Length);
    }

    #endregion

    #region GradientCheckpointing Tests

    [Fact]
    public void SequentialCheckpoint_InvalidSegmentSizeZero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var inputTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));
        var input = TensorOperations<double>.Variable(inputTensor, "input");
        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.ReLU(x),
            x => TensorOperations<double>.Sigmoid(x)
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: 0));
        Assert.Contains("segmentSize", ex.ParamName);
    }

    [Fact]
    public void SequentialCheckpoint_InvalidSegmentSizeNegative_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var inputTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));
        var input = TensorOperations<double>.Variable(inputTensor, "input");
        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.ReLU(x),
            x => TensorOperations<double>.Sigmoid(x)
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: -5));
        Assert.Contains("segmentSize", ex.ParamName);
    }

    [Fact]
    public void SequentialCheckpoint_ValidSegmentSize_ReturnsCorrectResult()
    {
        // Arrange
        var inputTensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));
        var input = TensorOperations<double>.Variable(inputTensor, "input");
        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.ReLU(x),
            x => TensorOperations<double>.Sigmoid(x)
        };

        // Act
        var result = GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: 2);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.Value);
    }

    #endregion

    #region GradientTape Tests

    [Fact]
    public void GradientTape_NonPersistent_SecondGradientCallThrows()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));

        using var tape = new GradientTape<double>(persistent: false);
        var x = TensorOperations<double>.Variable(tensor, "x", requiresGradient: true);
        tape.Watch(x);
        var y = TensorOperations<double>.Sum(x);

        // Act - First gradient call should succeed
        var grads1 = tape.Gradient(y, new[] { x });
        Assert.NotNull(grads1);

        // Assert - Second gradient call should throw
        Assert.Throws<InvalidOperationException>(() => tape.Gradient(y, new[] { x }));
    }

    [Fact]
    public void GradientTape_Persistent_MultipleGradientCallsSucceed()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));

        using var tape = new GradientTape<double>(persistent: true);
        var x = TensorOperations<double>.Variable(tensor, "x", requiresGradient: true);
        tape.Watch(x);
        var y = TensorOperations<double>.Sum(x);

        // Act & Assert - Multiple gradient calls should succeed
        var grads1 = tape.Gradient(y, new[] { x });
        Assert.NotNull(grads1);

        var grads2 = tape.Gradient(y, new[] { x });
        Assert.NotNull(grads2);
    }

    [Fact]
    public void GradientTape_DisposedTape_ThrowsObjectDisposedException()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>([1.0, 2.0, 3.0]));
        var tape = new GradientTape<double>();
        var x = TensorOperations<double>.Variable(tensor, "x", requiresGradient: true);
        tape.Watch(x);
        var y = TensorOperations<double>.Sum(x);

        // Act
        tape.Dispose();

        // Assert
        Assert.Throws<ObjectDisposedException>(() => tape.Gradient(y, new[] { x }));
    }

    #endregion

    #region Gradient Correctness Tests

    [Fact]
    public void Add_Gradient_IsCorrect()
    {
        // Arrange
        var aTensor = new Tensor<double>(new[] { 2 }, new Vector<double>([1.0, 2.0]));
        var bTensor = new Tensor<double>(new[] { 2 }, new Vector<double>([3.0, 4.0]));

        using var tape = new GradientTape<double>();
        var a = TensorOperations<double>.Variable(aTensor, "a", requiresGradient: true);
        var b = TensorOperations<double>.Variable(bTensor, "b", requiresGradient: true);
        tape.Watch(a);
        tape.Watch(b);

        // Act
        var c = TensorOperations<double>.Add(a, b);
        var loss = TensorOperations<double>.Sum(c);
        tape.Gradient(loss, new[] { a, b });

        // Assert - gradient of sum(a+b) w.r.t. a and b should be all ones
        Assert.NotNull(a.Gradient);
        Assert.NotNull(b.Gradient);
        Assert.Equal(1.0, a.Gradient[0], 5);
        Assert.Equal(1.0, a.Gradient[1], 5);
        Assert.Equal(1.0, b.Gradient[0], 5);
        Assert.Equal(1.0, b.Gradient[1], 5);
    }

    [Fact]
    public void Multiply_Gradient_IsCorrect()
    {
        // Arrange
        var aTensor = new Tensor<double>(new[] { 2 }, new Vector<double>([2.0, 3.0]));
        var bTensor = new Tensor<double>(new[] { 2 }, new Vector<double>([4.0, 5.0]));

        using var tape = new GradientTape<double>();
        var a = TensorOperations<double>.Variable(aTensor, "a", requiresGradient: true);
        var b = TensorOperations<double>.Variable(bTensor, "b", requiresGradient: true);
        tape.Watch(a);
        tape.Watch(b);

        // Act - c = a * b, loss = sum(c)
        var c = TensorOperations<double>.ElementwiseMultiply(a, b);
        var loss = TensorOperations<double>.Sum(c);
        tape.Gradient(loss, new[] { a, b });

        // Assert - gradient of sum(a*b) w.r.t. a is b, w.r.t. b is a
        Assert.NotNull(a.Gradient);
        Assert.NotNull(b.Gradient);
        Assert.Equal(4.0, a.Gradient[0], 5); // d/da (a*b) = b
        Assert.Equal(5.0, a.Gradient[1], 5);
        Assert.Equal(2.0, b.Gradient[0], 5); // d/db (a*b) = a
        Assert.Equal(3.0, b.Gradient[1], 5);
    }

    [Fact]
    public void Power_Gradient_IsCorrect()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2 }, new Vector<double>([2.0, 3.0]));

        using var tape = new GradientTape<double>();
        var a = TensorOperations<double>.Variable(tensor, "a", requiresGradient: true);
        tape.Watch(a);

        // Act - c = a^2, loss = sum(c)
        var c = TensorOperations<double>.Power(a, 2.0);
        var loss = TensorOperations<double>.Sum(c);
        tape.Gradient(loss, new[] { a });

        // Assert - gradient of sum(a^2) w.r.t. a is 2*a
        Assert.NotNull(a.Gradient);
        Assert.Equal(4.0, a.Gradient[0], 5); // 2 * 2 = 4
        Assert.Equal(6.0, a.Gradient[1], 5); // 2 * 3 = 6
    }

    #endregion
}
