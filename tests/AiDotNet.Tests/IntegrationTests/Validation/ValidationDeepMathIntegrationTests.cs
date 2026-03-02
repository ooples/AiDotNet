using AiDotNet.Exceptions;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Validation;

/// <summary>
/// Deep integration tests for TensorValidator, VectorValidator, and Guard:
/// shape validation, rank validation, length validation, range checks,
/// null/empty guards, boundary conditions, and error message accuracy.
/// </summary>
public class ValidationDeepMathIntegrationTests
{
    // ============================
    // TensorValidator.ValidateShape Tests
    // ============================

    [Fact]
    public void ValidateShape_MatchingShape_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        TensorValidator.ValidateShape(tensor, new[] { 3, 4 });
    }

    [Fact]
    public void ValidateShape_1DMatch_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        TensorValidator.ValidateShape(tensor, new[] { 5 });
    }

    [Fact]
    public void ValidateShape_3DMatch_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        TensorValidator.ValidateShape(tensor, new[] { 2, 3, 4 });
    }

    [Fact]
    public void ValidateShape_4DMatch_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4, 5 });
        TensorValidator.ValidateShape(tensor, new[] { 2, 3, 4, 5 });
    }

    [Fact]
    public void ValidateShape_WrongDimensions_Throws()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        Assert.Throws<TensorShapeMismatchException>(() =>
            TensorValidator.ValidateShape(tensor, new[] { 3, 5 }));
    }

    [Fact]
    public void ValidateShape_WrongRank_Throws()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        Assert.Throws<TensorShapeMismatchException>(() =>
            TensorValidator.ValidateShape(tensor, new[] { 3, 4, 1 }));
    }

    [Fact]
    public void ValidateShape_SwappedDimensions_Throws()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        Assert.Throws<TensorShapeMismatchException>(() =>
            TensorValidator.ValidateShape(tensor, new[] { 4, 3 }));
    }

    // ============================
    // TensorValidator.ValidateShapesMatch Tests
    // ============================

    [Fact]
    public void ValidateShapesMatch_SameShape_DoesNotThrow()
    {
        var t1 = new Tensor<double>(new[] { 3, 4 });
        var t2 = new Tensor<double>(new[] { 3, 4 });
        TensorValidator.ValidateShapesMatch(t1, t2);
    }

    [Fact]
    public void ValidateShapesMatch_DifferentShape_Throws()
    {
        var t1 = new Tensor<double>(new[] { 3, 4 });
        var t2 = new Tensor<double>(new[] { 3, 5 });
        Assert.Throws<TensorShapeMismatchException>(() =>
            TensorValidator.ValidateShapesMatch(t1, t2));
    }

    [Fact]
    public void ValidateShapesMatch_DifferentRank_Throws()
    {
        var t1 = new Tensor<double>(new[] { 3, 4 });
        var t2 = new Tensor<double>(new[] { 12 });
        Assert.Throws<TensorShapeMismatchException>(() =>
            TensorValidator.ValidateShapesMatch(t1, t2));
    }

    // ============================
    // TensorValidator.ValidateRank Tests
    // ============================

    [Fact]
    public void ValidateRank_1D_Correct()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        TensorValidator.ValidateRank(tensor, 1);
    }

    [Fact]
    public void ValidateRank_2D_Correct()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        TensorValidator.ValidateRank(tensor, 2);
    }

    [Fact]
    public void ValidateRank_3D_Correct()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        TensorValidator.ValidateRank(tensor, 3);
    }

    [Fact]
    public void ValidateRank_4D_Correct()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4, 5 });
        TensorValidator.ValidateRank(tensor, 4);
    }

    [Fact]
    public void ValidateRank_WrongRank_Throws()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        Assert.Throws<TensorRankException>(() =>
            TensorValidator.ValidateRank(tensor, 3));
    }

    [Fact]
    public void ValidateRank_1DExpected2D_Throws()
    {
        var tensor = new Tensor<double>(new[] { 12 });
        Assert.Throws<TensorRankException>(() =>
            TensorValidator.ValidateRank(tensor, 2));
    }

    // ============================
    // TensorValidator.ValidateForwardPassPerformed Tests
    // ============================

    [Fact]
    public void ValidateForwardPass_NullInput_Throws()
    {
        Assert.Throws<ForwardPassRequiredException>(() =>
            TensorValidator.ValidateForwardPassPerformed<double>(null));
    }

    [Fact]
    public void ValidateForwardPass_ValidInput_DoesNotThrow()
    {
        var input = new Tensor<double>(new[] { 3 });
        TensorValidator.ValidateForwardPassPerformed(input);
    }

    [Fact]
    public void ValidateForwardPassForLayer_NullInput_Throws()
    {
        Assert.Throws<ForwardPassRequiredException>(() =>
            TensorValidator.ValidateForwardPassPerformedForLayer<double>(null));
    }

    [Fact]
    public void ValidateForwardPassForLayer_ValidInput_DoesNotThrow()
    {
        var input = new Tensor<double>(new[] { 3 });
        TensorValidator.ValidateForwardPassPerformedForLayer(input);
    }

    // ============================
    // VectorValidator.ValidateLength Tests
    // ============================

    [Fact]
    public void VectorValidateLength_CorrectLength_DoesNotThrow()
    {
        var v = new Vector<double>(5);
        VectorValidator.ValidateLength(v, 5);
    }

    [Fact]
    public void VectorValidateLength_WrongLength_Throws()
    {
        var v = new Vector<double>(5);
        Assert.Throws<VectorLengthMismatchException>(() =>
            VectorValidator.ValidateLength(v, 3));
    }

    [Fact]
    public void VectorValidateLength_SingleElement_Correct()
    {
        var v = new Vector<double>(1);
        VectorValidator.ValidateLength(v, 1);
    }

    // ============================
    // VectorValidator.ValidateLengthForShape Tests
    // ============================

    [Fact]
    public void VectorValidateLengthForShape_Matching_DoesNotThrow()
    {
        // Shape [3,4] => product = 12
        var v = new Vector<double>(12);
        VectorValidator.ValidateLengthForShape(v, new[] { 3, 4 });
    }

    [Fact]
    public void VectorValidateLengthForShape_3DMatching_DoesNotThrow()
    {
        // Shape [2,3,4] => product = 24
        var v = new Vector<double>(24);
        VectorValidator.ValidateLengthForShape(v, new[] { 2, 3, 4 });
    }

    [Fact]
    public void VectorValidateLengthForShape_Mismatched_Throws()
    {
        // Shape [3,4] => product = 12, but vector is length 10
        var v = new Vector<double>(10);
        Assert.Throws<VectorLengthMismatchException>(() =>
            VectorValidator.ValidateLengthForShape(v, new[] { 3, 4 }));
    }

    [Fact]
    public void VectorValidateLengthForShape_1DMatch()
    {
        var v = new Vector<double>(5);
        VectorValidator.ValidateLengthForShape(v, new[] { 5 });
    }

    // ============================
    // Guard.NotNull Tests
    // ============================

    [Fact]
    public void Guard_NotNull_ValidObject_DoesNotThrow()
    {
        var obj = "hello";
        Guard.NotNull(obj);
    }

    [Fact]
    public void Guard_NotNull_NullObject_ThrowsArgumentNull()
    {
        string? obj = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNull(obj));
    }

    // ============================
    // Guard.NotNullOrEmpty Tests
    // ============================

    [Fact]
    public void Guard_NotNullOrEmpty_ValidString_DoesNotThrow()
    {
        Guard.NotNullOrEmpty("hello");
    }

    [Fact]
    public void Guard_NotNullOrEmpty_NullString_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() => Guard.NotNullOrEmpty(null));
    }

    [Fact]
    public void Guard_NotNullOrEmpty_EmptyString_ThrowsArgument()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrEmpty(""));
    }

    // ============================
    // Guard.NotNullOrWhiteSpace Tests
    // ============================

    [Fact]
    public void Guard_NotNullOrWhiteSpace_ValidString_DoesNotThrow()
    {
        Guard.NotNullOrWhiteSpace("hello");
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_NullString_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() => Guard.NotNullOrWhiteSpace(null));
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_WhitespaceOnly_ThrowsArgument()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace("   "));
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_Tab_ThrowsArgument()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace("\t"));
    }

    // ============================
    // Guard.Positive (int) Tests
    // ============================

    [Fact]
    public void Guard_PositiveInt_PositiveValue_DoesNotThrow()
    {
        Guard.Positive(1);
        Guard.Positive(100);
        Guard.Positive(int.MaxValue);
    }

    [Fact]
    public void Guard_PositiveInt_Zero_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(0));
    }

    [Fact]
    public void Guard_PositiveInt_Negative_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(-1));
    }

    // ============================
    // Guard.Positive (double) Tests
    // ============================

    [Fact]
    public void Guard_PositiveDouble_PositiveValue_DoesNotThrow()
    {
        Guard.Positive(0.001);
        Guard.Positive(1.0);
        Guard.Positive(1e15);
    }

    [Fact]
    public void Guard_PositiveDouble_Zero_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(0.0));
    }

    [Fact]
    public void Guard_PositiveDouble_Negative_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(-0.001));
    }

    [Fact]
    public void Guard_PositiveDouble_NaN_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.NaN));
    }

    [Fact]
    public void Guard_PositiveDouble_PositiveInfinity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.PositiveInfinity));
    }

    [Fact]
    public void Guard_PositiveDouble_NegativeInfinity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.NegativeInfinity));
    }

    // ============================
    // Guard.NonNegative (int) Tests
    // ============================

    [Fact]
    public void Guard_NonNegativeInt_Zero_DoesNotThrow()
    {
        Guard.NonNegative(0);
    }

    [Fact]
    public void Guard_NonNegativeInt_Positive_DoesNotThrow()
    {
        Guard.NonNegative(1);
        Guard.NonNegative(100);
    }

    [Fact]
    public void Guard_NonNegativeInt_Negative_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(-1));
    }

    // ============================
    // Guard.NonNegative (double) Tests
    // ============================

    [Fact]
    public void Guard_NonNegativeDouble_Zero_DoesNotThrow()
    {
        Guard.NonNegative(0.0);
    }

    [Fact]
    public void Guard_NonNegativeDouble_Positive_DoesNotThrow()
    {
        Guard.NonNegative(0.001);
        Guard.NonNegative(1e15);
    }

    [Fact]
    public void Guard_NonNegativeDouble_Negative_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(-0.001));
    }

    [Fact]
    public void Guard_NonNegativeDouble_NaN_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.NaN));
    }

    [Fact]
    public void Guard_NonNegativeDouble_Infinity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.PositiveInfinity));
    }

    // ============================
    // Guard.InRange (int) Tests
    // ============================

    [Fact]
    public void Guard_InRangeInt_InRange_DoesNotThrow()
    {
        Guard.InRange(5, 1, 10);
    }

    [Fact]
    public void Guard_InRangeInt_AtMin_DoesNotThrow()
    {
        Guard.InRange(1, 1, 10);
    }

    [Fact]
    public void Guard_InRangeInt_AtMax_DoesNotThrow()
    {
        Guard.InRange(10, 1, 10);
    }

    [Fact]
    public void Guard_InRangeInt_BelowMin_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0, 1, 10));
    }

    [Fact]
    public void Guard_InRangeInt_AboveMax_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(11, 1, 10));
    }

    [Fact]
    public void Guard_InRangeInt_MinEqualsMax_InRange_DoesNotThrow()
    {
        Guard.InRange(5, 5, 5);
    }

    [Fact]
    public void Guard_InRangeInt_MinEqualsMax_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(6, 5, 5));
    }

    [Fact]
    public void Guard_InRangeInt_MinGreaterThanMax_Throws()
    {
        Assert.Throws<ArgumentException>(() => Guard.InRange(5, 10, 1));
    }

    // ============================
    // Guard.InRange (double) Tests
    // ============================

    [Fact]
    public void Guard_InRangeDouble_InRange_DoesNotThrow()
    {
        Guard.InRange(0.5, 0.0, 1.0);
    }

    [Fact]
    public void Guard_InRangeDouble_AtBoundaries_DoesNotThrow()
    {
        Guard.InRange(0.0, 0.0, 1.0);
        Guard.InRange(1.0, 0.0, 1.0);
    }

    [Fact]
    public void Guard_InRangeDouble_OutOfRange_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(1.1, 0.0, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(-0.1, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRangeDouble_NaN_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(double.NaN, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRangeDouble_Infinity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Guard.InRange(double.PositiveInfinity, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRangeDouble_NaNMin_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Guard.InRange(0.5, double.NaN, 1.0));
    }

    [Fact]
    public void Guard_InRangeDouble_InfinityMax_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Guard.InRange(0.5, 0.0, double.PositiveInfinity));
    }

    [Fact]
    public void Guard_InRangeDouble_MinGreaterThanMax_Throws()
    {
        Assert.Throws<ArgumentException>(() => Guard.InRange(0.5, 1.0, 0.0));
    }

    // ============================
    // Exception Type Tests
    // ============================

    [Fact]
    public void TensorShapeMismatchException_IsAiDotNetException()
    {
        var ex = new TensorShapeMismatchException(new[] { 3, 4 }, new[] { 3, 5 }, "Test", "Forward");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    [Fact]
    public void TensorShapeMismatchException_ContainsShapeInfo()
    {
        var ex = new TensorShapeMismatchException(new[] { 3, 4 }, new[] { 3, 5 }, "Test", "Forward");
        Assert.Contains("3", ex.Message);
    }

    [Fact]
    public void TensorRankException_IsAiDotNetException()
    {
        var ex = new TensorRankException(2, 3, "Test", "Forward");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    [Fact]
    public void VectorLengthMismatchException_IsAiDotNetException()
    {
        var ex = new VectorLengthMismatchException(5, 3, "Test", "Forward");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    [Fact]
    public void ForwardPassRequiredException_IsAiDotNetException()
    {
        var ex = new ForwardPassRequiredException("TestLayer", "Dense");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // Validator With Component Names
    // ============================

    [Fact]
    public void ValidateShape_WithComponentName_IncludesInException()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        var ex = Assert.Throws<TensorShapeMismatchException>(() =>
            TensorValidator.ValidateShape(tensor, new[] { 3, 5 }, "TestLayer", "Forward"));

        Assert.Contains("TestLayer", ex.Message);
    }

    [Fact]
    public void ValidateRank_WithComponentName_IncludesInException()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        var ex = Assert.Throws<TensorRankException>(() =>
            TensorValidator.ValidateRank(tensor, 3, "ConvLayer", "Forward"));

        Assert.Contains("ConvLayer", ex.Message);
    }

    [Fact]
    public void VectorValidateLength_WithComponentName_IncludesInException()
    {
        var v = new Vector<double>(5);
        var ex = Assert.Throws<VectorLengthMismatchException>(() =>
            VectorValidator.ValidateLength(v, 10, "DenseLayer", "Backward"));

        Assert.Contains("DenseLayer", ex.Message);
    }

    // ============================
    // Shape Product Math Tests
    // ============================

    [Fact]
    public void VectorLengthForShape_ProductComputation_HandVerified()
    {
        // Shape [2, 3, 4] => product = 2*3*4 = 24
        var v = new Vector<double>(24);
        VectorValidator.ValidateLengthForShape(v, new[] { 2, 3, 4 });

        // Wrong product: 2*3*5 = 30 != 24
        Assert.Throws<VectorLengthMismatchException>(() =>
            VectorValidator.ValidateLengthForShape(v, new[] { 2, 3, 5 }));
    }

    [Fact]
    public void VectorLengthForShape_LargeShape_HandVerified()
    {
        // Shape [8, 16, 32] => product = 8*16*32 = 4096
        var v = new Vector<double>(4096);
        VectorValidator.ValidateLengthForShape(v, new[] { 8, 16, 32 });
    }

    [Fact]
    public void VectorLengthForShape_SingleDimension()
    {
        var v = new Vector<double>(100);
        VectorValidator.ValidateLengthForShape(v, new[] { 100 });
    }

    // ============================
    // Edge Cases
    // ============================

    [Fact]
    public void ValidateShape_SingleElement_Match()
    {
        var tensor = new Tensor<double>(new[] { 1 });
        TensorValidator.ValidateShape(tensor, new[] { 1 });
    }

    [Fact]
    public void ValidateShapesMatch_1DIdenticalTensors()
    {
        var t1 = new Tensor<double>(new[] { 100 });
        var t2 = new Tensor<double>(new[] { 100 });
        TensorValidator.ValidateShapesMatch(t1, t2);
    }

    [Fact]
    public void ValidateRank_SingleElement_Rank1()
    {
        var tensor = new Tensor<double>(new[] { 1 });
        TensorValidator.ValidateRank(tensor, 1);
    }

    [Fact]
    public void Guard_InRange_NegativeRange()
    {
        Guard.InRange(-5, -10, -1);
    }

    [Fact]
    public void Guard_InRange_NegativeRange_OutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0, -10, -1));
    }
}
