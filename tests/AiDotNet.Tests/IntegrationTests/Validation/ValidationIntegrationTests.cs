using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Validation;

/// <summary>
/// Integration tests for Validation utilities.
/// </summary>
public class ValidationIntegrationTests
{
    #region VectorValidator Tests

    [Fact]
    public void VectorValidator_ValidateLength_WithMatchingLength_Succeeds()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Should not throw
        VectorValidator.ValidateLength(vector, 3);
    }

    [Fact]
    public void VectorValidator_ValidateLength_WithMismatchedLength_ThrowsException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        Assert.ThrowsAny<Exception>(() => VectorValidator.ValidateLength(vector, 5));
    }

    [Fact]
    public void VectorValidator_ValidateLengthForShape_WithMatchingShape_Succeeds()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        var shape = new[] { 2, 3 }; // 2 * 3 = 6

        // Should not throw
        VectorValidator.ValidateLengthForShape(vector, shape);
    }

    [Fact]
    public void VectorValidator_ValidateLengthForShape_WithMismatchedShape_ThrowsException()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var shape = new[] { 2, 2 }; // 2 * 2 = 4, but vector has 3 elements

        Assert.ThrowsAny<Exception>(() => VectorValidator.ValidateLengthForShape(vector, shape));
    }

    [Fact]
    public void VectorValidator_Float_ValidateLength_Works()
    {
        var vector = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });

        // Should not throw
        VectorValidator.ValidateLength(vector, 3);
    }

    [Fact]
    public void VectorValidator_ValidateLength_WithDifferentSizes_Works()
    {
        var vector1 = new Vector<double>(new[] { 1.0 });
        var vector5 = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var vector10 = new Vector<double>(new double[10]);

        VectorValidator.ValidateLength(vector1, 1);
        VectorValidator.ValidateLength(vector5, 5);
        VectorValidator.ValidateLength(vector10, 10);
    }

    #endregion

    #region TensorValidator Tests

    [Fact]
    public void TensorValidator_ValidateShape_WithMatchingShape_Succeeds()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        var expectedShape = new[] { 2, 3, 4 };

        // Should not throw
        TensorValidator.ValidateShape(tensor, expectedShape);
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithMismatchedShape_ThrowsException()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        var expectedShape = new[] { 2, 4, 3 };

        Assert.ThrowsAny<Exception>(() => TensorValidator.ValidateShape(tensor, expectedShape));
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithDifferentDimensions_ThrowsException()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        var expectedShape = new[] { 2, 3, 4 };

        Assert.ThrowsAny<Exception>(() => TensorValidator.ValidateShape(tensor, expectedShape));
    }

    [Fact]
    public void TensorValidator_Float_ValidateShape_Works()
    {
        var tensor = new Tensor<float>(new[] { 3, 4 });
        var expectedShape = new[] { 3, 4 };

        // Should not throw
        TensorValidator.ValidateShape(tensor, expectedShape);
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithSingleDimension_Works()
    {
        var tensor = new Tensor<double>(new[] { 10 });
        var expectedShape = new[] { 10 };

        TensorValidator.ValidateShape(tensor, expectedShape);
    }

    [Fact]
    public void TensorValidator_ValidateShape_With4DDimension_Works()
    {
        var tensor = new Tensor<double>(new[] { 1, 3, 28, 28 }); // Batch, Channels, Height, Width
        var expectedShape = new[] { 1, 3, 28, 28 };

        TensorValidator.ValidateShape(tensor, expectedShape);
    }

    #endregion

    #region Cross-Validator Tests

    [Fact]
    public void AllValidators_WorkWithDifferentNumericTypes()
    {
        // Double
        var doubleVector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        VectorValidator.ValidateLength(doubleVector, 3);

        // Float
        var floatVector = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        VectorValidator.ValidateLength(floatVector, 3);

        // All should succeed without exceptions
        Assert.True(true);
    }

    [Fact]
    public void TensorAndVectorValidators_WorkTogether()
    {
        var vector = new Vector<double>(new double[12]);
        var tensor = new Tensor<double>(new[] { 3, 4 });

        // Vector length matches product of tensor shape
        VectorValidator.ValidateLengthForShape(vector, new[] { 3, 4 });
        TensorValidator.ValidateShape(tensor, new[] { 3, 4 });

        Assert.True(true);
    }

    [Fact]
    public void VectorValidator_ValidateLength_WithComponentAndOperation_Succeeds()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Should not throw even with component and operation parameters
        VectorValidator.ValidateLength(vector, 3, "TestComponent", "TestOperation");
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithComponentAndOperation_Succeeds()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        var expectedShape = new[] { 2, 3 };

        // Should not throw even with component and operation parameters
        TensorValidator.ValidateShape(tensor, expectedShape, "TestComponent", "TestOperation");
    }

    #endregion
}
