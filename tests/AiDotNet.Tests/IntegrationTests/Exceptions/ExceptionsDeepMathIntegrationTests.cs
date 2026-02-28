using AiDotNet.Enums;
using AiDotNet.Exceptions;
using Xunit;

using AiDotNetSerializationException = AiDotNet.Exceptions.SerializationException;

namespace AiDotNet.Tests.IntegrationTests.Exceptions;

/// <summary>
/// Deep integration tests for the AiDotNet exception hierarchy:
/// constructors, property assignments, FormatMessage output, inheritance,
/// inner exception chaining, and default value handling.
/// </summary>
public class ExceptionsDeepMathIntegrationTests
{
    // ============================
    // AiDotNetException Tests
    // ============================

    [Fact]
    public void AiDotNetException_DefaultConstructor_IsException()
    {
        var ex = new AiDotNetException();

        Assert.IsAssignableFrom<Exception>(ex);
        Assert.IsType<AiDotNetException>(ex);
    }

    [Fact]
    public void AiDotNetException_MessageConstructor_StoresMessage()
    {
        var ex = new AiDotNetException("test error");

        Assert.Equal("test error", ex.Message);
    }

    [Fact]
    public void AiDotNetException_InnerExceptionConstructor_ChainsCorrectly()
    {
        var inner = new InvalidOperationException("inner");
        var ex = new AiDotNetException("outer", inner);

        Assert.Equal("outer", ex.Message);
        Assert.Same(inner, ex.InnerException);
    }

    // ============================
    // TensorShapeMismatchException Tests
    // ============================

    [Fact]
    public void TensorShapeMismatch_DefaultConstructor_HasEmptyShapes()
    {
        var ex = new TensorShapeMismatchException();

        Assert.Empty(ex.ExpectedShape);
        Assert.Empty(ex.ActualShape);
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_MessageConstructor_HasEmptyShapes()
    {
        var ex = new TensorShapeMismatchException("custom message");

        Assert.Equal("custom message", ex.Message);
        Assert.Empty(ex.ExpectedShape);
        Assert.Empty(ex.ActualShape);
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_FullConstructor_StoresAllProperties()
    {
        var expected = new[] { 3, 4, 5 };
        var actual = new[] { 3, 4, 6 };

        var ex = new TensorShapeMismatchException(expected, actual, "DenseLayer", "Forward");

        Assert.Equal(expected, ex.ExpectedShape);
        Assert.Equal(actual, ex.ActualShape);
        Assert.Equal("DenseLayer", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_FormatMessage_ContainsExpectedAndActualShapes()
    {
        var expected = new[] { 10, 20 };
        var actual = new[] { 10, 30 };

        var ex = new TensorShapeMismatchException(expected, actual, "Conv2D", "Forward");

        // FormatMessage: "Shape mismatch in Conv2D during Forward: Expected shape [10, 20], but got [10, 30]."
        Assert.Contains("Shape mismatch", ex.Message);
        Assert.Contains("Conv2D", ex.Message);
        Assert.Contains("Forward", ex.Message);
        Assert.Contains("[10, 20]", ex.Message);
        Assert.Contains("[10, 30]", ex.Message);
    }

    [Fact]
    public void TensorShapeMismatch_FormatMessage_HandVerified()
    {
        var expected = new[] { 2, 3 };
        var actual = new[] { 4, 5 };

        var ex = new TensorShapeMismatchException(expected, actual, "MyComp", "MyOp");

        var expectedMessage = "Shape mismatch in MyComp during MyOp: Expected shape [2, 3], but got [4, 5].";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void TensorShapeMismatch_ContextConstructor_UsesUnknownOperation()
    {
        var expected = new[] { 5 };
        var actual = new[] { 10 };

        var ex = new TensorShapeMismatchException(expected, actual, "TestContext");

        Assert.Equal("TestContext", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_InnerExceptionConstructor_ChainsCorrectly()
    {
        var inner = new ArgumentException("bad arg");
        var expected = new[] { 1, 2, 3 };
        var actual = new[] { 1, 2, 4 };

        var ex = new TensorShapeMismatchException(expected, actual, "Layer1", "Backward", inner);

        Assert.Same(inner, ex.InnerException);
        Assert.Equal(expected, ex.ExpectedShape);
        Assert.Equal(actual, ex.ActualShape);
        Assert.Equal("Layer1", ex.Component);
        Assert.Equal("Backward", ex.Operation);
    }

    [Fact]
    public void TensorShapeMismatch_InheritsAiDotNetException()
    {
        var ex = new TensorShapeMismatchException();

        Assert.IsAssignableFrom<AiDotNetException>(ex);
        Assert.IsAssignableFrom<Exception>(ex);
    }

    [Fact]
    public void TensorShapeMismatch_EmptyShapes_FormatMessageHandlesGracefully()
    {
        var expected = Array.Empty<int>();
        var actual = Array.Empty<int>();

        var ex = new TensorShapeMismatchException(expected, actual, "Test", "Op");

        // Empty shapes should produce "[]"
        Assert.Contains("[]", ex.Message);
    }

    [Fact]
    public void TensorShapeMismatch_SingleDimShapes_FormatMessageCorrect()
    {
        var expected = new[] { 100 };
        var actual = new[] { 200 };

        var ex = new TensorShapeMismatchException(expected, actual, "FC", "Forward");

        Assert.Contains("[100]", ex.Message);
        Assert.Contains("[200]", ex.Message);
    }

    // ============================
    // TensorRankException Tests
    // ============================

    [Fact]
    public void TensorRank_StoresExpectedAndActualRank()
    {
        var ex = new TensorRankException(3, 2, "Conv3D", "Forward");

        Assert.Equal(3, ex.ExpectedRank);
        Assert.Equal(2, ex.ActualRank);
        Assert.Equal("Conv3D", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void TensorRank_FormatMessage_HandVerified()
    {
        var ex = new TensorRankException(4, 2, "BatchNorm", "Validate");

        var expectedMessage = "Rank mismatch in BatchNorm during Validate: Expected rank 4, but got 2.";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void TensorRank_InheritsAiDotNetException()
    {
        var ex = new TensorRankException(1, 2, "Test", "Op");

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    [Fact]
    public void TensorRank_SameRank_StillCreatesException()
    {
        // Edge case: expected == actual (someone might construct this)
        var ex = new TensorRankException(3, 3, "Test", "Op");

        Assert.Equal(3, ex.ExpectedRank);
        Assert.Equal(3, ex.ActualRank);
        Assert.Contains("Expected rank 3, but got 3", ex.Message);
    }

    [Fact]
    public void TensorRank_ZeroRank_HandledCorrectly()
    {
        var ex = new TensorRankException(0, 1, "Scalar", "Validate");

        Assert.Equal(0, ex.ExpectedRank);
        Assert.Equal(1, ex.ActualRank);
        Assert.Contains("Expected rank 0", ex.Message);
    }

    // ============================
    // TensorDimensionException Tests
    // ============================

    [Fact]
    public void TensorDimension_StoresAllProperties()
    {
        var ex = new TensorDimensionException(2, 100, 200, "Linear", "Forward");

        Assert.Equal(2, ex.DimensionIndex);
        Assert.Equal(100, ex.ExpectedValue);
        Assert.Equal(200, ex.ActualValue);
        Assert.Equal("Linear", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void TensorDimension_FormatMessage_HandVerified()
    {
        var ex = new TensorDimensionException(0, 32, 64, "Conv2D", "Forward");

        var expectedMessage = "Dimension mismatch in Conv2D during Forward: Expected dimension 0 to be 32, but got 64.";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void TensorDimension_InheritsAiDotNetException()
    {
        var ex = new TensorDimensionException(0, 1, 2, "Test", "Op");

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    [Fact]
    public void TensorDimension_HighDimensionIndex_HandledCorrectly()
    {
        var ex = new TensorDimensionException(5, 10, 20, "HighDim", "Check");

        Assert.Equal(5, ex.DimensionIndex);
        Assert.Contains("dimension 5", ex.Message);
    }

    // ============================
    // VectorLengthMismatchException Tests
    // ============================

    [Fact]
    public void VectorLength_StoresExpectedAndActualLength()
    {
        var ex = new VectorLengthMismatchException(100, 50, "DotProduct", "Compute");

        Assert.Equal(100, ex.ExpectedLength);
        Assert.Equal(50, ex.ActualLength);
        Assert.Equal("DotProduct", ex.Component);
        Assert.Equal("Compute", ex.Operation);
    }

    [Fact]
    public void VectorLength_FormatMessage_HandVerified()
    {
        var ex = new VectorLengthMismatchException(10, 20, "Embedding", "Lookup");

        var expectedMessage = "Vector length mismatch in Embedding during Lookup: Expected length 10, but got 20.";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void VectorLength_InheritsAiDotNetException()
    {
        var ex = new VectorLengthMismatchException(1, 2, "Test", "Op");

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    [Fact]
    public void VectorLength_ZeroLength_HandledCorrectly()
    {
        var ex = new VectorLengthMismatchException(0, 5, "EmptyCheck", "Validate");

        Assert.Equal(0, ex.ExpectedLength);
        Assert.Contains("Expected length 0", ex.Message);
    }

    // ============================
    // ForwardPassRequiredException Tests
    // ============================

    [Fact]
    public void ForwardPass_TwoArgConstructor_StoresProperties()
    {
        var ex = new ForwardPassRequiredException("hidden1", "Dense");

        Assert.Equal("hidden1", ex.ComponentName);
        Assert.Equal("Dense", ex.ComponentType);
        Assert.Equal("backward pass", ex.Operation);
    }

    [Fact]
    public void ForwardPass_TwoArgConstructor_FormatMessage_HandVerified()
    {
        var ex = new ForwardPassRequiredException("layer1", "Conv2D");

        var expectedMessage = "Forward pass must be called before backward pass in layer 'layer1' of type Conv2D.";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void ForwardPass_ThreeArgConstructor_StoresProperties()
    {
        var ex = new ForwardPassRequiredException("encoder", "Transformer", "attention computation");

        Assert.Equal("encoder", ex.ComponentName);
        Assert.Equal("Transformer", ex.ComponentType);
        Assert.Equal("attention computation", ex.Operation);
    }

    [Fact]
    public void ForwardPass_ThreeArgConstructor_FormatMessage_HandVerified()
    {
        var ex = new ForwardPassRequiredException("decoder", "LSTM", "gradient calc");

        var expectedMessage = "Forward pass must be called before gradient calc in LSTM 'decoder'.";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void ForwardPass_InheritsAiDotNetException()
    {
        var ex = new ForwardPassRequiredException("test", "Dense");

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // InvalidDataValueException Tests
    // ============================

    [Fact]
    public void InvalidDataValue_DefaultConstructor_HasUnknownComponents()
    {
        var ex = new InvalidDataValueException();

        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void InvalidDataValue_MessageConstructor_HasUnknownComponents()
    {
        var ex = new InvalidDataValueException("NaN detected");

        Assert.Equal("NaN detected", ex.Message);
        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void InvalidDataValue_ContextConstructor_StoresProperties()
    {
        var ex = new InvalidDataValueException("Infinity found", "Normalizer", "Scale");

        Assert.Equal("Normalizer", ex.Component);
        Assert.Equal("Scale", ex.Operation);
    }

    [Fact]
    public void InvalidDataValue_ContextConstructor_FormatMessage_HandVerified()
    {
        var ex = new InvalidDataValueException("NaN detected", "Loss", "Compute");

        var expectedMessage = "Invalid data value in Loss during Compute: NaN detected";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void InvalidDataValue_InnerExceptionConstructor_ChainsCorrectly()
    {
        var inner = new DivideByZeroException();
        var ex = new InvalidDataValueException("divide by zero", inner);

        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void InvalidDataValue_FullConstructor_ChainsInnerException()
    {
        var inner = new OverflowException("overflow");
        var ex = new InvalidDataValueException("value overflow", "Activator", "ReLU", inner);

        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Activator", ex.Component);
        Assert.Equal("ReLU", ex.Operation);
        Assert.Contains("Invalid data value in Activator during ReLU: value overflow", ex.Message);
    }

    [Fact]
    public void InvalidDataValue_InheritsAiDotNetException()
    {
        var ex = new InvalidDataValueException();

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // InvalidInputDimensionException Tests
    // ============================

    [Fact]
    public void InvalidInputDimension_DefaultConstructor_HasUnknownComponents()
    {
        var ex = new InvalidInputDimensionException();

        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void InvalidInputDimension_MessageConstructor_HasUnknownComponents()
    {
        var ex = new InvalidInputDimensionException("wrong dim");

        Assert.Equal("wrong dim", ex.Message);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void InvalidInputDimension_ContextConstructor_FormatMessage_HandVerified()
    {
        var ex = new InvalidInputDimensionException("Expected 3D input", "Conv1D", "Forward");

        var expectedMessage = "Dimension error in Conv1D during Forward: Expected 3D input";
        Assert.Equal(expectedMessage, ex.Message);
        Assert.Equal("Conv1D", ex.Component);
        Assert.Equal("Forward", ex.Operation);
    }

    [Fact]
    public void InvalidInputDimension_FullConstructor_ChainsInnerException()
    {
        var inner = new ArgumentException("bad");
        var ex = new InvalidInputDimensionException("2D required", "Pool", "Check", inner);

        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Pool", ex.Component);
        Assert.Equal("Check", ex.Operation);
    }

    [Fact]
    public void InvalidInputDimension_InheritsAiDotNetException()
    {
        var ex = new InvalidInputDimensionException();

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // InvalidInputTypeException Tests
    // ============================

    [Fact]
    public void InvalidInputType_StoresExpectedAndActualTypes()
    {
        var ex = new InvalidInputTypeException(InputType.TwoDimensional, InputType.OneDimensional, "CNN");

        Assert.Equal(InputType.TwoDimensional, ex.ExpectedInputType);
        Assert.Equal(InputType.OneDimensional, ex.ActualInputType);
        Assert.Equal("CNN", ex.NetworkType);
    }

    [Fact]
    public void InvalidInputType_FormatMessage_HandVerified()
    {
        var ex = new InvalidInputTypeException(InputType.ThreeDimensional, InputType.OneDimensional, "RNN");

        var expectedMessage = $"RNN requires {InputType.ThreeDimensional} input, but received {InputType.OneDimensional} input.";
        Assert.Equal(expectedMessage, ex.Message);
    }

    [Fact]
    public void InvalidInputType_InheritsAiDotNetException()
    {
        var ex = new InvalidInputTypeException(InputType.OneDimensional, InputType.TwoDimensional, "Test");

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // SerializationException Tests
    // ============================

    [Fact]
    public void Serialization_DefaultConstructor_HasUnknownComponents()
    {
        var ex = new AiDotNetSerializationException();

        Assert.Equal("Unknown", ex.Component);
        Assert.Equal("Unknown", ex.Operation);
    }

    [Fact]
    public void Serialization_MessageConstructor_HasUnknownComponents()
    {
        var ex = new AiDotNetSerializationException("bad format");

        Assert.Equal("bad format", ex.Message);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void Serialization_ContextConstructor_FormatMessage_HandVerified()
    {
        var ex = new AiDotNetSerializationException("invalid json", "ModelSaver", "Save");

        var expectedMessage = "Serialization error in ModelSaver during Save: invalid json";
        Assert.Equal(expectedMessage, ex.Message);
        Assert.Equal("ModelSaver", ex.Component);
        Assert.Equal("Save", ex.Operation);
    }

    [Fact]
    public void Serialization_InnerExceptionConstructor_ChainsCorrectly()
    {
        var inner = new FormatException("bad");
        var ex = new AiDotNetSerializationException("parse error", inner);

        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Unknown", ex.Component);
    }

    [Fact]
    public void Serialization_FullConstructor_ChainsInnerException()
    {
        var inner = new IOException("disk full");
        var ex = new AiDotNetSerializationException("write failed", "Checkpoint", "Export", inner);

        Assert.Same(inner, ex.InnerException);
        Assert.Equal("Checkpoint", ex.Component);
        Assert.Equal("Export", ex.Operation);
    }

    [Fact]
    public void Serialization_InheritsAiDotNetException()
    {
        var ex = new AiDotNetSerializationException();

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // ModelTrainingException Tests
    // ============================

    [Fact]
    public void ModelTraining_DefaultConstructor_IsValid()
    {
        var ex = new ModelTrainingException();

        Assert.IsType<ModelTrainingException>(ex);
    }

    [Fact]
    public void ModelTraining_MessageConstructor_StoresMessage()
    {
        var ex = new ModelTrainingException("training diverged");

        Assert.Equal("training diverged", ex.Message);
    }

    [Fact]
    public void ModelTraining_InnerExceptionConstructor_ChainsCorrectly()
    {
        var inner = new ArithmeticException("NaN loss");
        var ex = new ModelTrainingException("training failed", inner);

        Assert.Equal("training failed", ex.Message);
        Assert.Same(inner, ex.InnerException);
    }

    [Fact]
    public void ModelTraining_InheritsAiDotNetException()
    {
        var ex = new ModelTrainingException();

        Assert.IsAssignableFrom<AiDotNetException>(ex);
    }

    // ============================
    // Cross-Exception Hierarchy Tests
    // ============================

    [Fact]
    public void AllCustomExceptions_InheritFromAiDotNetException()
    {
        // Verify the entire exception hierarchy
        Assert.IsAssignableFrom<AiDotNetException>(new TensorShapeMismatchException());
        Assert.IsAssignableFrom<AiDotNetException>(new TensorRankException(1, 2, "T", "O"));
        Assert.IsAssignableFrom<AiDotNetException>(new TensorDimensionException(0, 1, 2, "T", "O"));
        Assert.IsAssignableFrom<AiDotNetException>(new VectorLengthMismatchException(1, 2, "T", "O"));
        Assert.IsAssignableFrom<AiDotNetException>(new ForwardPassRequiredException("n", "t"));
        Assert.IsAssignableFrom<AiDotNetException>(new InvalidDataValueException());
        Assert.IsAssignableFrom<AiDotNetException>(new InvalidInputDimensionException());
        Assert.IsAssignableFrom<AiDotNetException>(new InvalidInputTypeException(InputType.OneDimensional, InputType.TwoDimensional, "T"));
        Assert.IsAssignableFrom<AiDotNetException>(new AiDotNetSerializationException());
        Assert.IsAssignableFrom<AiDotNetException>(new ModelTrainingException());
    }

    [Fact]
    public void AllCustomExceptions_AreCatchableAsException()
    {
        // All should be catchable as System.Exception
        Exception caught;

        try { throw new TensorShapeMismatchException(new[] { 1 }, new[] { 2 }, "T", "O"); }
        catch (Exception ex) { caught = ex; Assert.IsType<TensorShapeMismatchException>(caught); }

        try { throw new TensorRankException(1, 2, "T", "O"); }
        catch (Exception ex) { caught = ex; Assert.IsType<TensorRankException>(caught); }

        try { throw new VectorLengthMismatchException(1, 2, "T", "O"); }
        catch (Exception ex) { caught = ex; Assert.IsType<VectorLengthMismatchException>(caught); }
    }

    [Fact]
    public void AllCustomExceptions_CatchableAsAiDotNetException()
    {
        AiDotNetException caught;

        try { throw new TensorDimensionException(0, 10, 20, "T", "O"); }
        catch (AiDotNetException ex) { caught = ex; Assert.IsType<TensorDimensionException>(caught); }

        try { throw new ForwardPassRequiredException("n", "t"); }
        catch (AiDotNetException ex) { caught = ex; Assert.IsType<ForwardPassRequiredException>(caught); }

        try { throw new ModelTrainingException("msg"); }
        catch (AiDotNetException ex) { caught = ex; Assert.IsType<ModelTrainingException>(caught); }
    }

    // ============================
    // Exception Message Format Consistency Tests
    // ============================

    [Fact]
    public void ShapeMismatch_MessageContainsComponentAndOperation()
    {
        var ex = new TensorShapeMismatchException(new[] { 1 }, new[] { 2 }, "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    [Fact]
    public void RankMismatch_MessageContainsComponentAndOperation()
    {
        var ex = new TensorRankException(3, 4, "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    [Fact]
    public void DimensionMismatch_MessageContainsComponentAndOperation()
    {
        var ex = new TensorDimensionException(1, 10, 20, "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    [Fact]
    public void VectorLengthMismatch_MessageContainsComponentAndOperation()
    {
        var ex = new VectorLengthMismatchException(5, 10, "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    [Fact]
    public void InvalidDataValue_ContextMessage_ContainsComponentAndOperation()
    {
        var ex = new InvalidDataValueException("msg", "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    [Fact]
    public void InvalidInputDimension_ContextMessage_ContainsComponentAndOperation()
    {
        var ex = new InvalidInputDimensionException("msg", "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    [Fact]
    public void SerializationException_ContextMessage_ContainsComponentAndOperation()
    {
        var ex = new AiDotNetSerializationException("msg", "MyComp", "MyOp");

        Assert.Contains("MyComp", ex.Message);
        Assert.Contains("MyOp", ex.Message);
    }

    // ============================
    // Large Shape/Dimension Tests
    // ============================

    [Fact]
    public void TensorShapeMismatch_4DShape_FormatCorrectly()
    {
        // Common 4D tensor shape: [batch, channels, height, width]
        var expected = new[] { 32, 3, 224, 224 };
        var actual = new[] { 32, 3, 112, 112 };

        var ex = new TensorShapeMismatchException(expected, actual, "ResNet", "Forward");

        Assert.Contains("[32, 3, 224, 224]", ex.Message);
        Assert.Contains("[32, 3, 112, 112]", ex.Message);
    }

    [Fact]
    public void TensorShapeMismatch_HighRankShape_FormatCorrectly()
    {
        // 6D tensor: [batch, time, heads, seq, depth, features]
        var expected = new[] { 8, 10, 12, 64, 32, 16 };
        var actual = new[] { 8, 10, 12, 64, 32, 8 };

        var ex = new TensorShapeMismatchException(expected, actual, "Attention", "Compute");

        Assert.Contains("[8, 10, 12, 64, 32, 16]", ex.Message);
        Assert.Contains("[8, 10, 12, 64, 32, 8]", ex.Message);
    }

    [Fact]
    public void TensorRank_HighRank_FormatCorrectly()
    {
        var ex = new TensorRankException(6, 4, "MultiHead", "Reshape");

        Assert.Contains("Expected rank 6", ex.Message);
        Assert.Contains("but got 4", ex.Message);
    }

    // ============================
    // Inner Exception Chain Depth Tests
    // ============================

    [Fact]
    public void Exception_MultiLevelChain_PreservesAll()
    {
        var root = new DivideByZeroException("root cause");
        var mid = new InvalidDataValueException("intermediate", root);
        var outer = new ModelTrainingException("training failed", mid);

        Assert.Same(mid, outer.InnerException);
        Assert.IsType<InvalidDataValueException>(outer.InnerException);
        Assert.Same(root, outer.InnerException?.InnerException);
        Assert.IsType<DivideByZeroException>(outer.InnerException?.InnerException);
    }

    [Fact]
    public void TensorShapeMismatch_DeepChain_PreservesAll()
    {
        var root = new ArgumentException("root");
        var mid = new AiDotNetException("mid", root);
        var outer = new TensorShapeMismatchException("outer", mid);

        Assert.Same(mid, outer.InnerException);
        Assert.Same(root, outer.InnerException?.InnerException);
    }
}
