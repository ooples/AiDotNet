using AiDotNet.Enums;
using AiDotNet.Exceptions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Exceptions;

/// <summary>
/// Integration tests for all AiDotNet custom exception classes.
/// Verifies construction, message formatting, property storage, and inheritance.
/// </summary>
public class ExceptionsIntegrationTests
{
    #region AiDotNetException

    [Fact]
    public void AiDotNetException_DefaultConstructor()
    {
        var ex = new AiDotNetException();
        Assert.NotNull(ex);
        Assert.IsAssignableFrom<Exception>(ex);
    }

    [Fact]
    public void AiDotNetException_MessageConstructor()
    {
        var ex = new AiDotNetException("test error");
        Assert.Equal("test error", ex.Message);
    }

    [Fact]
    public void AiDotNetException_MessageAndInnerException()
    {
        var inner = new InvalidOperationException("inner");
        var ex = new AiDotNetException("outer", inner);
        Assert.Equal("outer", ex.Message);
        Assert.Same(inner, ex.InnerException);
    }

    #endregion

    #region TensorShapeMismatchException

    [Fact]
    public void TensorShapeMismatch_DefaultConstructor_SetsDefaults()
    {
        var ex = new TensorShapeMismatchException();
        Assert.Empty(ex.ExpectedShape);
        Assert.Empty(ex.ActualShape);
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_MessageConstructor_SetsDefaults()
    {
        var ex = new TensorShapeMismatchException("shape error");
        Assert.Contains("shape error", ex.Message);
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_WithInnerException()
    {
        var inner = new Exception("root cause");
        var ex = new TensorShapeMismatchException("wrapper", inner);
        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void TensorShapeMismatch_FullConstructor_StoresProperties()
    {
        var expected = new[] { 1, 3, 224, 224 };
        var actual = new[] { 1, 3, 112, 112 };
        var ex = new TensorShapeMismatchException(expected, actual, "ConvLayer", "Forward");

        Assert.Equal(expected, ex.ExpectedShape);
        Assert.Equal(actual, ex.ActualShape);
        Assert.Equal("ConvLayer", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_FullConstructor_FormatsMessage()
    {
        var ex = new TensorShapeMismatchException(new[] { 2, 3 }, new[] { 4, 5 }, "DenseLayer", "MatMul");
        Assert.Contains("DenseLayer", ex.Message);
        Assert.Contains("MatMul", ex.Message);
        Assert.Contains("2, 3", ex.Message);
        Assert.Contains("4, 5", ex.Message);
    }

    [Fact]
    public void TensorShapeMismatch_ThreeArgConstructor_SetsOperationToUnknown()
    {
        var ex = new TensorShapeMismatchException(new[] { 1 }, new[] { 2 }, "MyLayer");
        Assert.Equal("MyLayer", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_WithInnerAndShapes()
    {
        var inner = new Exception("root");
        var ex = new TensorShapeMismatchException(new[] { 1 }, new[] { 2 }, "Comp", "Op", inner);
        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Comp", ex.Component);
        Assert.Equal("Op", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_InheritsFromAiDotNetException()
    {
        var ex = new TensorShapeMismatchException();
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region TensorRankException

    [Fact]
    public void TensorRankException_StoresProperties()
    {
        var ex = new TensorRankException(2, 3, "DenseLayer", "Forward");
        Assert.Equal(2, ex.ExpectedRank);
        Assert.Equal(3, ex.ActualRank);
        Assert.Equal("DenseLayer", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void TensorRankException_FormatsMessage()
    {
        var ex = new TensorRankException(2, 4, "Conv2D", "Forward");
        Assert.Contains("Conv2D", ex.Message);
        Assert.Contains("Forward", ex.Message);
        Assert.Contains("2", ex.Message);
        Assert.Contains("4", ex.Message);
    }

    [Fact]
    public void TensorRankException_InheritsFromAiDotNetException()
    {
        var ex = new TensorRankException(1, 2, "Test", "Test");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region TensorDimensionException

    [Fact]
    public void TensorDimensionException_StoresProperties()
    {
        var ex = new TensorDimensionException(1, 224, 112, "ConvLayer", "Forward");
        Assert.Equal(1, ex.DimensionIndex);
        Assert.Equal(224, ex.ExpectedValue);
        Assert.Equal(112, ex.ActualValue);
        Assert.Equal("ConvLayer", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void TensorDimensionException_FormatsMessage()
    {
        var ex = new TensorDimensionException(0, 10, 20, "Pool", "Compute");
        Assert.Contains("Pool", ex.Message);
        Assert.Contains("Compute", ex.Message);
        Assert.Contains("0", ex.Message);
        Assert.Contains("10", ex.Message);
        Assert.Contains("20", ex.Message);
    }

    [Fact]
    public void TensorDimensionException_InheritsFromAiDotNetException()
    {
        var ex = new TensorDimensionException(0, 1, 2, "T", "O");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region VectorLengthMismatchException

    [Fact]
    public void VectorLengthMismatch_StoresProperties()
    {
        var ex = new VectorLengthMismatchException(10, 20, "DotProduct", "Compute");
        Assert.Equal(10, ex.ExpectedLength);
        Assert.Equal(20, ex.ActualLength);
        Assert.Equal("DotProduct", ex.Component);
        Assert.Equal("Compute", ex.Operation);
    }

    [Fact]
    public void VectorLengthMismatch_FormatsMessage()
    {
        var ex = new VectorLengthMismatchException(5, 10, "Add", "ElementWise");
        Assert.Contains("Add", ex.Message);
        Assert.Contains("ElementWise", ex.Message);
        Assert.Contains("5", ex.Message);
        Assert.Contains("10", ex.Message);
    }

    [Fact]
    public void VectorLengthMismatch_InheritsFromAiDotNetException()
    {
        var ex = new VectorLengthMismatchException(1, 2, "T", "O");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region ForwardPassRequiredException

    [Fact]
    public void ForwardPassRequired_TwoArgConstructor()
    {
        var ex = new ForwardPassRequiredException("layer1", "DenseLayer");
        Assert.Equal("layer1", ex.ComponentName);
        Assert.Equal("DenseLayer", ex.ComponentType);
        Assert.Equal("backward pass", ex.Operation);
        Assert.Contains("layer1", ex.Message);
        Assert.Contains("DenseLayer", ex.Message);
    }

    [Fact]
    public void ForwardPassRequired_ThreeArgConstructor()
    {
        var ex = new ForwardPassRequiredException("bn1", "BatchNorm", "gradient computation");
        Assert.Equal("bn1", ex.ComponentName);
        Assert.Equal("BatchNorm", ex.ComponentType);
        Assert.Equal("gradient computation", ex.Operation);
        Assert.Contains("gradient computation", ex.Message);
    }

    [Fact]
    public void ForwardPassRequired_InheritsFromAiDotNetException()
    {
        var ex = new ForwardPassRequiredException("x", "y");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region InvalidDataValueException

    [Fact]
    public void InvalidDataValue_DefaultConstructor()
    {
        var ex = new InvalidDataValueException();
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void InvalidDataValue_MessageConstructor()
    {
        var ex = new InvalidDataValueException("NaN detected");
        Assert.Contains("NaN detected", ex.Message);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void InvalidDataValue_WithInnerException()
    {
        var inner = new ArithmeticException("overflow");
        var ex = new InvalidDataValueException("bad value", inner);
        Assert.Same(inner, ex.InnerException);
    }

    [Fact]
    public void InvalidDataValue_WithContext()
    {
        var ex = new InvalidDataValueException("NaN found", "Normalizer", "Scale");
        Assert.Equal("Normalizer", ex.Component);
        Assert.Equal("Scale", ex.Operation);
        Assert.Contains("Normalizer", ex.Message);
        Assert.Contains("Scale", ex.Message);
        Assert.Contains("NaN found", ex.Message);
    }

    [Fact]
    public void InvalidDataValue_WithContextAndInner()
    {
        var inner = new Exception("root");
        var ex = new InvalidDataValueException("inf", "Layer", "Forward", inner);
        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Layer", ex.Component);
    }

    [Fact]
    public void InvalidDataValue_InheritsFromAiDotNetException()
    {
        var ex = new InvalidDataValueException();
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region InvalidInputDimensionException

    [Fact]
    public void InvalidInputDimension_DefaultConstructor()
    {
        var ex = new InvalidInputDimensionException();
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void InvalidInputDimension_MessageConstructor()
    {
        var ex = new InvalidInputDimensionException("wrong dimensions");
        Assert.Contains("wrong dimensions", ex.Message);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void InvalidInputDimension_WithInnerException()
    {
        var inner = new Exception("source");
        var ex = new InvalidInputDimensionException("msg", inner);
        Assert.Same(inner, ex.InnerException);
    }

    [Fact]
    public void InvalidInputDimension_WithContext()
    {
        var ex = new InvalidInputDimensionException("need 2D", "LSTM", "Forward");
        Assert.Equal("LSTM", ex.Component);
        Assert.Equal("Forward", ex.Operation);
        Assert.Contains("LSTM", ex.Message);
        Assert.Contains("Forward", ex.Message);
        Assert.Contains("need 2D", ex.Message);
    }

    [Fact]
    public void InvalidInputDimension_WithContextAndInner()
    {
        var inner = new Exception("root");
        var ex = new InvalidInputDimensionException("bad dim", "RNN", "Init", inner);
        Assert.Same(inner, ex.InnerException);
        Assert.Equal("RNN", ex.Component);
        Assert.Equal("Init", ex.Operation);
    }

    [Fact]
    public void InvalidInputDimension_InheritsFromAiDotNetException()
    {
        var ex = new InvalidInputDimensionException();
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region InvalidInputTypeException

    [Fact]
    public void InvalidInputType_StoresProperties()
    {
        var ex = new InvalidInputTypeException(InputType.TwoDimensional, InputType.OneDimensional, "ConvNet");
        Assert.Equal(InputType.TwoDimensional, ex.ExpectedInputType);
        Assert.Equal(InputType.OneDimensional, ex.ActualInputType);
        Assert.Equal("ConvNet", ex.NetworkType);
    }

    [Fact]
    public void InvalidInputType_FormatsMessage()
    {
        var ex = new InvalidInputTypeException(InputType.ThreeDimensional, InputType.OneDimensional, "ResNet");
        Assert.Contains("ResNet", ex.Message);
        Assert.Contains(InputType.ThreeDimensional.ToString(), ex.Message);
        Assert.Contains(InputType.OneDimensional.ToString(), ex.Message);
    }

    [Fact]
    public void InvalidInputType_InheritsFromAiDotNetException()
    {
        var ex = new InvalidInputTypeException(InputType.OneDimensional, InputType.TwoDimensional, "Net");
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region ModelTrainingException

    [Fact]
    public void ModelTrainingException_DefaultConstructor()
    {
        var ex = new ModelTrainingException();
        Assert.NotNull(ex);
    }

    [Fact]
    public void ModelTrainingException_MessageConstructor()
    {
        var ex = new ModelTrainingException("training failed");
        Assert.Equal("training failed", ex.Message);
    }

    [Fact]
    public void ModelTrainingException_WithInnerException()
    {
        var inner = new InvalidOperationException("diverged");
        var ex = new ModelTrainingException("loss exploded", inner);
        Assert.Equal("loss exploded", ex.Message);
        Assert.Same(inner, ex.InnerException);
    }

    [Fact]
    public void ModelTrainingException_InheritsFromAiDotNetException()
    {
        var ex = new ModelTrainingException();
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region SerializationException

    [Fact]
    public void SerializationException_DefaultConstructor()
    {
        var ex = new AiDotNet.Exceptions.SerializationException();
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void SerializationException_MessageConstructor()
    {
        var ex = new AiDotNet.Exceptions.SerializationException("corrupt data");
        Assert.Contains("corrupt data", ex.Message);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void SerializationException_WithInnerException()
    {
        var inner = new IOException("stream closed");
        var ex = new AiDotNet.Exceptions.SerializationException("read failed", inner);
        Assert.Same(inner, ex.InnerException);
    }

    [Fact]
    public void SerializationException_WithContext()
    {
        var ex = new AiDotNet.Exceptions.SerializationException("version mismatch", "ModelSaver", "Save");
        Assert.Equal("ModelSaver", ex.Component);
        Assert.Equal("Save", ex.Operation);
        Assert.Contains("ModelSaver", ex.Message);
        Assert.Contains("Save", ex.Message);
    }

    [Fact]
    public void SerializationException_WithContextAndInner()
    {
        var inner = new Exception("root");
        var ex = new AiDotNet.Exceptions.SerializationException("fail", "Loader", "Load", inner);
        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Loader", ex.Component);
        Assert.Equal("Load", ex.Operation);
    }

    [Fact]
    public void SerializationException_InheritsFromAiDotNetException()
    {
        var ex = new AiDotNet.Exceptions.SerializationException();
        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    #endregion

    #region Cross-Exception - All Derive from AiDotNetException

    [Fact]
    public void AllExceptions_DeriveFromAiDotNetException()
    {
        // Verify the entire exception hierarchy
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(TensorShapeMismatchException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(TensorRankException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(TensorDimensionException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(VectorLengthMismatchException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(ForwardPassRequiredException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(InvalidDataValueException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(InvalidInputDimensionException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(InvalidInputTypeException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(ModelTrainingException)));
        Assert.True(typeof(AiDotNetException).IsAssignableFrom(typeof(AiDotNet.Exceptions.SerializationException)));
    }

    [Fact]
    public void AllExceptions_DeriveFromSystemException()
    {
        Assert.True(typeof(Exception).IsAssignableFrom(typeof(AiDotNetException)));
        Assert.True(typeof(Exception).IsAssignableFrom(typeof(TensorShapeMismatchException)));
    }

    #endregion
}
