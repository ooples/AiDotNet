using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Validation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Validation;

/// <summary>
/// Integration tests for Validation utilities:
/// Guard, VectorValidator, TensorValidator, ArchitectureValidator,
/// RegressionValidator, SerializationValidator, GNNBenchmarkValidator.
/// </summary>
public class ValidationIntegrationTests
{
    #region Guard - NotNull

    [Fact]
    public void Guard_NotNull_WithNonNullValue_DoesNotThrow()
    {
        var obj = new object();
        Guard.NotNull(obj);
    }

    [Fact]
    public void Guard_NotNull_WithNullValue_ThrowsArgumentNullException()
    {
        object? obj = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNull(obj));
    }

    [Fact]
    public void Guard_NotNull_WithString_DoesNotThrow()
    {
        string value = "hello";
        Guard.NotNull(value);
    }

    [Fact]
    public void Guard_NotNull_WithNullString_ThrowsArgumentNullException()
    {
        string? value = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNull(value));
    }

    #endregion

    #region Guard - NotNullOrEmpty

    [Fact]
    public void Guard_NotNullOrEmpty_WithValidString_DoesNotThrow()
    {
        Guard.NotNullOrEmpty("hello");
    }

    [Fact]
    public void Guard_NotNullOrEmpty_WithNullString_ThrowsArgumentNullException()
    {
        string? value = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNullOrEmpty(value));
    }

    [Fact]
    public void Guard_NotNullOrEmpty_WithEmptyString_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrEmpty(""));
    }

    [Fact]
    public void Guard_NotNullOrEmpty_WithWhitespaceString_DoesNotThrow()
    {
        // NotNullOrEmpty only checks for empty, not whitespace
        Guard.NotNullOrEmpty("   ");
    }

    #endregion

    #region Guard - NotNullOrWhiteSpace

    [Fact]
    public void Guard_NotNullOrWhiteSpace_WithValidString_DoesNotThrow()
    {
        Guard.NotNullOrWhiteSpace("hello");
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_WithNullString_ThrowsArgumentNullException()
    {
        string? value = null;
        Assert.Throws<ArgumentNullException>(() => Guard.NotNullOrWhiteSpace(value));
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_WithEmptyString_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace(""));
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_WithWhitespaceString_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace("   "));
    }

    [Fact]
    public void Guard_NotNullOrWhiteSpace_WithTabOnly_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.NotNullOrWhiteSpace("\t"));
    }

    #endregion

    #region Guard - Positive (int)

    [Fact]
    public void Guard_Positive_Int_WithPositiveValue_DoesNotThrow()
    {
        Guard.Positive(1);
        Guard.Positive(100);
        Guard.Positive(int.MaxValue);
    }

    [Fact]
    public void Guard_Positive_Int_WithZero_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(0));
    }

    [Fact]
    public void Guard_Positive_Int_WithNegative_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(-100));
    }

    #endregion

    #region Guard - Positive (double)

    [Fact]
    public void Guard_Positive_Double_WithPositiveValue_DoesNotThrow()
    {
        Guard.Positive(0.001);
        Guard.Positive(1.0);
        Guard.Positive(1e10);
    }

    [Fact]
    public void Guard_Positive_Double_WithZero_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(0.0));
    }

    [Fact]
    public void Guard_Positive_Double_WithNegative_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(-0.001));
    }

    [Fact]
    public void Guard_Positive_Double_WithNaN_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.NaN));
    }

    [Fact]
    public void Guard_Positive_Double_WithInfinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.Positive(double.NegativeInfinity));
    }

    #endregion

    #region Guard - NonNegative (int)

    [Fact]
    public void Guard_NonNegative_Int_WithPositive_DoesNotThrow()
    {
        Guard.NonNegative(1);
        Guard.NonNegative(100);
    }

    [Fact]
    public void Guard_NonNegative_Int_WithZero_DoesNotThrow()
    {
        Guard.NonNegative(0);
    }

    [Fact]
    public void Guard_NonNegative_Int_WithNegative_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(-1));
    }

    #endregion

    #region Guard - NonNegative (double)

    [Fact]
    public void Guard_NonNegative_Double_WithPositive_DoesNotThrow()
    {
        Guard.NonNegative(0.001);
        Guard.NonNegative(1.0);
    }

    [Fact]
    public void Guard_NonNegative_Double_WithZero_DoesNotThrow()
    {
        Guard.NonNegative(0.0);
    }

    [Fact]
    public void Guard_NonNegative_Double_WithNegative_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(-0.001));
    }

    [Fact]
    public void Guard_NonNegative_Double_WithNaN_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.NaN));
    }

    [Fact]
    public void Guard_NonNegative_Double_WithInfinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.NonNegative(double.NegativeInfinity));
    }

    #endregion

    #region Guard - InRange (int)

    [Fact]
    public void Guard_InRange_Int_WithinRange_DoesNotThrow()
    {
        Guard.InRange(5, 1, 10);
        Guard.InRange(1, 1, 10); // min boundary
        Guard.InRange(10, 1, 10); // max boundary
    }

    [Fact]
    public void Guard_InRange_Int_BelowMin_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0, 1, 10));
    }

    [Fact]
    public void Guard_InRange_Int_AboveMax_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(11, 1, 10));
    }

    [Fact]
    public void Guard_InRange_Int_MinGreaterThanMax_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.InRange(5, 10, 1));
    }

    [Fact]
    public void Guard_InRange_Int_SameMinMax_WorksForExactValue()
    {
        Guard.InRange(5, 5, 5);
    }

    #endregion

    #region Guard - InRange (double)

    [Fact]
    public void Guard_InRange_Double_WithinRange_DoesNotThrow()
    {
        Guard.InRange(0.5, 0.0, 1.0);
        Guard.InRange(0.0, 0.0, 1.0); // min boundary
        Guard.InRange(1.0, 0.0, 1.0); // max boundary
    }

    [Fact]
    public void Guard_InRange_Double_BelowMin_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(-0.1, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRange_Double_AboveMax_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(1.1, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRange_Double_NaN_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(double.NaN, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRange_Double_Infinity_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(double.PositiveInfinity, 0.0, 1.0));
    }

    [Fact]
    public void Guard_InRange_Double_MinGreaterThanMax_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Guard.InRange(0.5, 1.0, 0.0));
    }

    [Fact]
    public void Guard_InRange_Double_NaNMin_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0.5, double.NaN, 1.0));
    }

    [Fact]
    public void Guard_InRange_Double_InfinityMax_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Guard.InRange(0.5, 0.0, double.PositiveInfinity));
    }

    #endregion

    #region VectorValidator Tests

    [Fact]
    public void VectorValidator_ValidateLength_WithMatchingLength_Succeeds()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
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

    [Fact]
    public void VectorValidator_ValidateLength_WithComponentAndOperation_Succeeds()
    {
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        VectorValidator.ValidateLength(vector, 3, "TestComponent", "TestOperation");
    }

    [Fact]
    public void VectorValidator_ValidateLengthForShape_3DShape_Succeeds()
    {
        var vector = new Vector<double>(new double[24]); // 2 * 3 * 4 = 24
        VectorValidator.ValidateLengthForShape(vector, new[] { 2, 3, 4 });
    }

    [Fact]
    public void VectorValidator_ValidateLengthForShape_SingleDim_Succeeds()
    {
        var vector = new Vector<double>(new double[5]);
        VectorValidator.ValidateLengthForShape(vector, new[] { 5 });
    }

    #endregion

    #region TensorValidator Tests

    [Fact]
    public void TensorValidator_ValidateShape_WithMatchingShape_Succeeds()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        TensorValidator.ValidateShape(tensor, new[] { 2, 3, 4 });
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithMismatchedShape_ThrowsException()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        Assert.ThrowsAny<Exception>(() => TensorValidator.ValidateShape(tensor, new[] { 2, 4, 3 }));
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithDifferentDimensions_ThrowsException()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        Assert.ThrowsAny<Exception>(() => TensorValidator.ValidateShape(tensor, new[] { 2, 3, 4 }));
    }

    [Fact]
    public void TensorValidator_Float_ValidateShape_Works()
    {
        var tensor = new Tensor<float>(new[] { 3, 4 });
        TensorValidator.ValidateShape(tensor, new[] { 3, 4 });
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithSingleDimension_Works()
    {
        var tensor = new Tensor<double>(new[] { 10 });
        TensorValidator.ValidateShape(tensor, new[] { 10 });
    }

    [Fact]
    public void TensorValidator_ValidateShape_With4DDimension_Works()
    {
        var tensor = new Tensor<double>(new[] { 1, 3, 28, 28 });
        TensorValidator.ValidateShape(tensor, new[] { 1, 3, 28, 28 });
    }

    [Fact]
    public void TensorValidator_ValidateShape_WithComponentAndOperation_Succeeds()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        TensorValidator.ValidateShape(tensor, new[] { 2, 3 }, "TestComponent", "TestOperation");
    }

    [Fact]
    public void TensorValidator_ValidateForwardPassPerformed_WithTensor_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        TensorValidator.ValidateForwardPassPerformed<double>(tensor);
    }

    [Fact]
    public void TensorValidator_ValidateForwardPassPerformed_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            TensorValidator.ValidateForwardPassPerformed<double>(null));
    }

    [Fact]
    public void TensorValidator_ValidateForwardPassPerformed_WithComponentInfo_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            TensorValidator.ValidateForwardPassPerformed<double>(null, "DenseLayer", "Dense", "Backward"));
    }

    [Fact]
    public void TensorValidator_ValidateForwardPassPerformedForLayer_WithTensor_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 3 });
        TensorValidator.ValidateForwardPassPerformedForLayer<double>(tensor, "TestLayer", "Dense");
    }

    [Fact]
    public void TensorValidator_ValidateForwardPassPerformedForLayer_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            TensorValidator.ValidateForwardPassPerformedForLayer<double>(null, "TestLayer", "Dense"));
    }

    [Fact]
    public void TensorValidator_ValidateShapesMatch_MatchingShapes_DoesNotThrow()
    {
        var tensor1 = new Tensor<double>(new[] { 2, 3 });
        var tensor2 = new Tensor<double>(new[] { 2, 3 });
        TensorValidator.ValidateShapesMatch(tensor1, tensor2);
    }

    [Fact]
    public void TensorValidator_ValidateShapesMatch_DifferentShapes_Throws()
    {
        var tensor1 = new Tensor<double>(new[] { 2, 3 });
        var tensor2 = new Tensor<double>(new[] { 3, 2 });
        Assert.ThrowsAny<Exception>(() =>
            TensorValidator.ValidateShapesMatch(tensor1, tensor2));
    }

    [Fact]
    public void TensorValidator_ValidateShapesMatch_DifferentRanks_Throws()
    {
        var tensor1 = new Tensor<double>(new[] { 6 });
        var tensor2 = new Tensor<double>(new[] { 2, 3 });
        Assert.ThrowsAny<Exception>(() =>
            TensorValidator.ValidateShapesMatch(tensor1, tensor2));
    }

    [Fact]
    public void TensorValidator_ValidateShapesMatch_WithComponentInfo()
    {
        var tensor1 = new Tensor<double>(new[] { 2, 3 });
        var tensor2 = new Tensor<double>(new[] { 2, 3 });
        TensorValidator.ValidateShapesMatch(tensor1, tensor2, "TestComponent", "Addition");
    }

    [Fact]
    public void TensorValidator_ValidateRank_MatchingRank_DoesNotThrow()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        TensorValidator.ValidateRank(tensor, 3);
    }

    [Fact]
    public void TensorValidator_ValidateRank_MismatchedRank_Throws()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        Assert.ThrowsAny<Exception>(() => TensorValidator.ValidateRank(tensor, 3));
    }

    [Fact]
    public void TensorValidator_ValidateRank_1D_Works()
    {
        var tensor = new Tensor<double>(new[] { 10 });
        TensorValidator.ValidateRank(tensor, 1);
    }

    [Fact]
    public void TensorValidator_ValidateRank_4D_Works()
    {
        var tensor = new Tensor<double>(new[] { 1, 3, 28, 28 });
        TensorValidator.ValidateRank(tensor, 4);
    }

    [Fact]
    public void TensorValidator_ValidateRank_WithComponentInfo()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        TensorValidator.ValidateRank(tensor, 2, "ConvLayer", "Forward");
    }

    #endregion

    #region ArchitectureValidator Tests

    [Fact]
    public void ArchitectureValidator_ValidateInputType_MatchingType_DoesNotThrow()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        ArchitectureValidator.ValidateInputType(arch, InputType.OneDimensional, "TestNetwork");
    }

    [Fact]
    public void ArchitectureValidator_ValidateInputType_MismatchedType_ThrowsInvalidInputTypeException()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        Assert.Throws<InvalidInputTypeException>(() =>
            ArchitectureValidator.ValidateInputType(arch, InputType.TwoDimensional, "TestNetwork"));
    }

    [Fact]
    public void ArchitectureValidator_ValidateInputType_2DArchWith2DExpected_DoesNotThrow()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 28,
            inputWidth: 28,
            outputSize: 10);

        ArchitectureValidator.ValidateInputType(arch, InputType.TwoDimensional, "CNN");
    }

    [Fact]
    public void ArchitectureValidator_ValidateInputType_3DArchWith1DExpected_Throws()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        Assert.Throws<InvalidInputTypeException>(() =>
            ArchitectureValidator.ValidateInputType(arch, InputType.OneDimensional, "MLP"));
    }

    #endregion

    #region RegressionValidator Tests

    [Fact]
    public void RegressionValidator_ValidateFeatureCount_MatchingColumns_DoesNotThrow()
    {
        var x = new Matrix<double>(5, 3);
        RegressionValidator.ValidateFeatureCount(x, 3);
    }

    [Fact]
    public void RegressionValidator_ValidateFeatureCount_MismatchedColumns_Throws()
    {
        var x = new Matrix<double>(5, 3);
        Assert.ThrowsAny<Exception>(() =>
            RegressionValidator.ValidateFeatureCount(x, 4));
    }

    [Fact]
    public void RegressionValidator_ValidateFeatureCount_WithComponentInfo()
    {
        var x = new Matrix<double>(5, 3);
        RegressionValidator.ValidateFeatureCount(x, 3, "LinearRegression", "Predict");
    }

    [Fact]
    public void RegressionValidator_ValidateInputOutputDimensions_Matching_DoesNotThrow()
    {
        var x = new Matrix<double>(10, 3);
        var y = new Vector<double>(new double[10]);
        RegressionValidator.ValidateInputOutputDimensions(x, y);
    }

    [Fact]
    public void RegressionValidator_ValidateInputOutputDimensions_Mismatched_Throws()
    {
        var x = new Matrix<double>(10, 3);
        var y = new Vector<double>(new double[5]); // 5 != 10
        Assert.ThrowsAny<Exception>(() =>
            RegressionValidator.ValidateInputOutputDimensions(x, y));
    }

    [Fact]
    public void RegressionValidator_ValidateDataValues_ValidData_DoesNotThrow()
    {
        var x = new Matrix<double>(3, 2);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                x[i, j] = (i + 1) * (j + 1);

        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        RegressionValidator.ValidateDataValues(x, y);
    }

    [Fact]
    public void RegressionValidator_ValidateDataValues_NaNInMatrix_Throws()
    {
        var x = new Matrix<double>(3, 2);
        x[1, 0] = double.NaN;
        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        Assert.ThrowsAny<Exception>(() =>
            RegressionValidator.ValidateDataValues(x, y));
    }

    [Fact]
    public void RegressionValidator_ValidateDataValues_InfinityInMatrix_Throws()
    {
        var x = new Matrix<double>(3, 2);
        x[0, 1] = double.PositiveInfinity;
        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        Assert.ThrowsAny<Exception>(() =>
            RegressionValidator.ValidateDataValues(x, y));
    }

    [Fact]
    public void RegressionValidator_ValidateDataValues_NaNInVector_Throws()
    {
        var x = new Matrix<double>(3, 2);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                x[i, j] = 1.0;

        var y = new Vector<double>(new double[] { 1.0, double.NaN, 3.0 });

        Assert.ThrowsAny<Exception>(() =>
            RegressionValidator.ValidateDataValues(x, y));
    }

    [Fact]
    public void RegressionValidator_ValidateDataValues_InfinityInVector_Throws()
    {
        var x = new Matrix<double>(3, 2);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                x[i, j] = 1.0;

        var y = new Vector<double>(new double[] { 1.0, 2.0, double.NegativeInfinity });

        Assert.ThrowsAny<Exception>(() =>
            RegressionValidator.ValidateDataValues(x, y));
    }

    #endregion

    #region SerializationValidator Tests

    [Fact]
    public void SerializationValidator_ValidateWriter_WithValidWriter_DoesNotThrow()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        SerializationValidator.ValidateWriter(writer);
    }

    [Fact]
    public void SerializationValidator_ValidateWriter_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateWriter(null));
    }

    [Fact]
    public void SerializationValidator_ValidateWriter_WithComponentInfo()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        SerializationValidator.ValidateWriter(writer, "DenseLayer", "Serialize");
    }

    [Fact]
    public void SerializationValidator_ValidateReader_WithValidReader_DoesNotThrow()
    {
        using var stream = new MemoryStream(new byte[] { 1, 2, 3 });
        using var reader = new BinaryReader(stream);
        SerializationValidator.ValidateReader(reader);
    }

    [Fact]
    public void SerializationValidator_ValidateReader_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateReader(null));
    }

    [Fact]
    public void SerializationValidator_ValidateStream_WithValidReadStream_DoesNotThrow()
    {
        using var stream = new MemoryStream(new byte[] { 1, 2, 3 });
        SerializationValidator.ValidateStream(stream, requireRead: true);
    }

    [Fact]
    public void SerializationValidator_ValidateStream_WithValidWriteStream_DoesNotThrow()
    {
        using var stream = new MemoryStream();
        SerializationValidator.ValidateStream(stream, requireWrite: true);
    }

    [Fact]
    public void SerializationValidator_ValidateStream_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateStream(null));
    }

    [Fact]
    public void SerializationValidator_ValidateFilePath_WithValidPath_DoesNotThrow()
    {
        SerializationValidator.ValidateFilePath("/some/path/model.bin");
    }

    [Fact]
    public void SerializationValidator_ValidateFilePath_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateFilePath(null));
    }

    [Fact]
    public void SerializationValidator_ValidateFilePath_WithEmpty_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateFilePath(""));
    }

    [Fact]
    public void SerializationValidator_ValidateFilePath_WithWhitespace_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateFilePath("   "));
    }

    [Fact]
    public void SerializationValidator_ValidateVersion_MatchingVersion_DoesNotThrow()
    {
        SerializationValidator.ValidateVersion(1, 1);
        SerializationValidator.ValidateVersion(42, 42);
    }

    [Fact]
    public void SerializationValidator_ValidateVersion_MismatchedVersion_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateVersion(1, 2));
    }

    [Fact]
    public void SerializationValidator_ValidateLayerTypeName_WithValidName_DoesNotThrow()
    {
        SerializationValidator.ValidateLayerTypeName("DenseLayer");
    }

    [Fact]
    public void SerializationValidator_ValidateLayerTypeName_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateLayerTypeName(null));
    }

    [Fact]
    public void SerializationValidator_ValidateLayerTypeName_WithEmpty_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateLayerTypeName(""));
    }

    [Fact]
    public void SerializationValidator_ValidateLayerTypeExists_WithValidType_DoesNotThrow()
    {
        SerializationValidator.ValidateLayerTypeExists("System.String", typeof(string));
    }

    [Fact]
    public void SerializationValidator_ValidateLayerTypeExists_WithNull_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            SerializationValidator.ValidateLayerTypeExists("UnknownLayer", null));
    }

    #endregion

    #region GNNBenchmarkValidator Result Classes

    [Fact]
    public void NodeClassificationResults_DefaultProperties()
    {
        var results = new GNNBenchmarkValidator<double>.NodeClassificationResults();
        Assert.Equal(0.0, results.TestAccuracy);
        Assert.Equal(0.0, results.TrainAccuracy);
        Assert.False(results.PassedBaseline);
        Assert.Equal(0.0, results.ExpectedAccuracy);
        Assert.Equal(string.Empty, results.DatasetName);
    }

    [Fact]
    public void NodeClassificationResults_SetProperties()
    {
        var results = new GNNBenchmarkValidator<double>.NodeClassificationResults
        {
            TestAccuracy = 0.81,
            TrainAccuracy = 0.95,
            PassedBaseline = true,
            ExpectedAccuracy = 0.80,
            DatasetName = "Cora"
        };

        Assert.Equal(0.81, results.TestAccuracy);
        Assert.Equal(0.95, results.TrainAccuracy);
        Assert.True(results.PassedBaseline);
        Assert.Equal(0.80, results.ExpectedAccuracy);
        Assert.Equal("Cora", results.DatasetName);
    }

    [Fact]
    public void GraphClassificationResults_DefaultProperties()
    {
        var results = new GNNBenchmarkValidator<double>.GraphClassificationResults();
        Assert.Equal(0.0, results.TestAccuracy);
        Assert.False(results.PassedBaseline);
        Assert.Equal(0.0, results.ExpectedAccuracy);
        Assert.Equal(string.Empty, results.DatasetName);
    }

    [Fact]
    public void GraphClassificationResults_SetProperties()
    {
        var results = new GNNBenchmarkValidator<double>.GraphClassificationResults
        {
            TestAccuracy = 0.85,
            PassedBaseline = true,
            ExpectedAccuracy = 0.80,
            DatasetName = "ZINC"
        };

        Assert.Equal(0.85, results.TestAccuracy);
        Assert.True(results.PassedBaseline);
        Assert.Equal("ZINC", results.DatasetName);
    }

    [Fact]
    public void LinkPredictionResults_DefaultProperties()
    {
        var results = new GNNBenchmarkValidator<double>.LinkPredictionResults();
        Assert.Equal(0.0, results.AUC);
        Assert.False(results.PassedBaseline);
        Assert.Equal(0.0, results.ExpectedAUC);
        Assert.Equal(string.Empty, results.DatasetName);
    }

    [Fact]
    public void LinkPredictionResults_SetProperties()
    {
        var results = new GNNBenchmarkValidator<double>.LinkPredictionResults
        {
            AUC = 0.90,
            PassedBaseline = true,
            ExpectedAUC = 0.85,
            DatasetName = "Cora"
        };

        Assert.Equal(0.90, results.AUC);
        Assert.True(results.PassedBaseline);
        Assert.Equal(0.85, results.ExpectedAUC);
    }

    [Fact]
    public void GNNBenchmarkValidator_CanInstantiate()
    {
        var validator = new GNNBenchmarkValidator<double>();
        Assert.NotNull(validator);
    }

    #endregion

    #region Cross-Validator Tests

    [Fact]
    public void AllValidators_WorkWithDifferentNumericTypes()
    {
        var doubleVector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        VectorValidator.ValidateLength(doubleVector, 3);

        var floatVector = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        VectorValidator.ValidateLength(floatVector, 3);
    }

    [Fact]
    public void TensorAndVectorValidators_WorkTogether()
    {
        var vector = new Vector<double>(new double[12]);
        var tensor = new Tensor<double>(new[] { 3, 4 });

        VectorValidator.ValidateLengthForShape(vector, new[] { 3, 4 });
        TensorValidator.ValidateShape(tensor, new[] { 3, 4 });
    }

    #endregion
}
