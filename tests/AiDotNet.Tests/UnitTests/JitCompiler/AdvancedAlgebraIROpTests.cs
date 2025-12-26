using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using Xunit;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Unit tests for advanced algebra IR operations.
/// </summary>
public class AdvancedAlgebraIROpTests
{
    #region Octonion Operation Tests

    [Fact]
    public void OctonionMultiplyOp_ValidInputs_PassesValidation()
    {
        // Arrange
        var op = new OctonionMultiplyOp
        {
            OutputId = 0,
            InputIds = new int[16], // 8 components each for 2 octonions
            OutputShape = new[] { 8 },
            OutputType = IRType.Float64
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void OctonionMultiplyOp_InvalidInputCount_FailsValidation()
    {
        // Arrange
        var op = new OctonionMultiplyOp
        {
            OutputId = 0,
            InputIds = new int[10], // Wrong count
            OutputShape = new[] { 8 },
            OutputType = IRType.Float64
        };

        // Act & Assert
        Assert.False(op.Validate());
    }

    [Fact]
    public void OctonionMatMulOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new OctonionMatMulOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 32, 64 },
            OutputType = IRType.Float64,
            BatchSize = 32,
            InputFeatures = 128,
            OutputFeatures = 64
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void OctonionMatMulOp_InvalidBatchSize_FailsValidation()
    {
        // Arrange
        var op = new OctonionMatMulOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 32, 64 },
            OutputType = IRType.Float64,
            BatchSize = 0, // Invalid
            InputFeatures = 128,
            OutputFeatures = 64
        };

        // Act & Assert
        Assert.False(op.Validate());
    }

    #endregion

    #region Geometric Algebra Operation Tests

    [Fact]
    public void GeometricProductOp_ValidConfiguration_PassesValidation()
    {
        // Arrange: G(3,0,0) - 3D Euclidean space
        var op = new GeometricProductOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 8 }, // 2^3 = 8 basis elements
            OutputType = IRType.Float64,
            PositiveSignature = 3,
            NegativeSignature = 0,
            ZeroSignature = 0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void GeometricProductOp_NegativeSignature_FailsValidation()
    {
        // Arrange
        var op = new GeometricProductOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 8 },
            OutputType = IRType.Float64,
            PositiveSignature = -1, // Invalid
            NegativeSignature = 0,
            ZeroSignature = 0
        };

        // Act & Assert
        Assert.False(op.Validate());
    }

    [Fact]
    public void WedgeProductOp_ValidConfiguration_PassesValidation()
    {
        // Arrange: G(2,0,0) - 2D space
        var op = new WedgeProductOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 4 }, // 2^2 = 4 basis elements
            OutputType = IRType.Float64,
            PositiveSignature = 2,
            NegativeSignature = 0,
            ZeroSignature = 0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    #endregion

    #region Hyperbolic Operation Tests

    [Fact]
    public void MobiusAddOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new MobiusAddOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 }, // 10-dimensional hyperbolic embeddings
            OutputType = IRType.Float64,
            Curvature = -1.0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void MobiusAddOp_PositiveCurvature_FailsValidation()
    {
        // Arrange: Positive curvature is invalid for hyperbolic space
        var op = new MobiusAddOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 },
            OutputType = IRType.Float64,
            Curvature = 1.0 // Invalid for hyperbolic
        };

        // Act & Assert
        Assert.False(op.Validate());
    }

    [Fact]
    public void PoincareExpMapOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new PoincareExpMapOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 }, // base point, tangent vector
            OutputShape = new[] { 10 },
            OutputType = IRType.Float64,
            Curvature = -1.0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void PoincareLogMapOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new PoincareLogMapOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1 }, // base point, target point
            OutputShape = new[] { 10 },
            OutputType = IRType.Float64,
            Curvature = -1.0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    #endregion

    #region Sparse Operation Tests

    [Fact]
    public void SpMVOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new SpMVOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1, 2, 3 }, // row_ptr, col_idx, values, vector
            OutputShape = new[] { 100 },
            OutputType = IRType.Float64,
            Rows = 100,
            Columns = 50,
            NonZeroCount = 500
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void SpMVOp_InvalidDimensions_FailsValidation()
    {
        // Arrange
        var op = new SpMVOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1, 2, 3 },
            OutputShape = new[] { 100 },
            OutputType = IRType.Float64,
            Rows = 0, // Invalid
            Columns = 50,
            NonZeroCount = 500
        };

        // Act & Assert
        Assert.False(op.Validate());
    }

    [Fact]
    public void SpMMOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new SpMMOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1, 2, 3 }, // row_ptr, col_idx, values, dense matrix
            OutputShape = new[] { 100, 64 },
            OutputType = IRType.Float64,
            SparseRows = 100,
            SparseColumns = 50,
            DenseColumns = 64,
            NonZeroCount = 500
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    #endregion

    #region Gradient Operation Tests

    [Fact]
    public void GradOctonionMultiplyOp_ValidInputs_PassesValidation()
    {
        // Arrange
        var op = new GradOctonionMultiplyOp
        {
            OutputIds = new[] { 0, 1 }, // d_a, d_b (gradient outputs)
            InputIds = new[] { 0, 1, 2 }, // a, b, grad_output
            OutputShape = new[] { 8 },
            OutputType = IRType.Float64
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void GradGeometricProductOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new GradGeometricProductOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1, 2 },
            OutputShape = new[] { 8 },
            OutputType = IRType.Float64,
            PositiveSignature = 3,
            NegativeSignature = 0,
            ZeroSignature = 0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void GradMobiusAddOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new GradMobiusAddOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1, 2 },
            OutputShape = new[] { 10 },
            OutputType = IRType.Float64,
            Curvature = -1.0
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    [Fact]
    public void GradSpMVOp_ValidConfiguration_PassesValidation()
    {
        // Arrange
        var op = new GradSpMVOp
        {
            OutputId = 0,
            InputIds = new[] { 0, 1, 2, 3, 4 }, // sparse components, vector, grad_output
            OutputShape = new[] { 50 },
            OutputType = IRType.Float64,
            Rows = 100,
            Columns = 50
        };

        // Act & Assert
        Assert.True(op.Validate());
    }

    #endregion

    #region OpType Tests

    [Fact]
    public void AllOps_HaveCorrectOpTypeNames()
    {
        // Verify OpType naming convention (removes "Op" suffix)
        Assert.Equal("OctonionMultiply", new OctonionMultiplyOp().OpType);
        Assert.Equal("OctonionMatMul", new OctonionMatMulOp().OpType);
        Assert.Equal("GeometricProduct", new GeometricProductOp().OpType);
        Assert.Equal("WedgeProduct", new WedgeProductOp().OpType);
        Assert.Equal("MobiusAdd", new MobiusAddOp().OpType);
        Assert.Equal("PoincareExpMap", new PoincareExpMapOp().OpType);
        Assert.Equal("PoincareLogMap", new PoincareLogMapOp().OpType);
        Assert.Equal("SpMV", new SpMVOp().OpType);
        Assert.Equal("SpMM", new SpMMOp().OpType);
    }

    #endregion
}
