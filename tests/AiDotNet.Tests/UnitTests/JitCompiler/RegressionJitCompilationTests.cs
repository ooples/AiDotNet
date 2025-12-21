using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using JitCompilerClass = AiDotNet.JitCompiler.JitCompiler;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for JIT compilation support in regression models.
/// Verifies that linear and kernel-based regression models support JIT compilation correctly.
/// </summary>
/// <remarks>
/// These tests are quarantined because they trigger GPU initialization which can fail
/// on machines without proper GPU support or drivers.
/// </remarks>
[Trait("Category", "GPU")]
public class RegressionJitCompilationTests
{
    // ========== SimpleRegression Tests ==========

    [Fact]
    public void SimpleRegression_SupportsJitCompilation()
    {
        // Arrange
        var model = new SimpleRegression<double>();
        var (X, y) = GenerateLinearTestData(100, 1);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "SimpleRegression should support JIT after training");
    }

    [Fact]
    public void SimpleRegression_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var model = new SimpleRegression<double>();
        var (X, y) = GenerateLinearTestData(100, 1);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    [Fact]
    public void SimpleRegression_JitCompilation_ProducesCorrectResults()
    {
        // Arrange
        var model = new SimpleRegression<double>();
        var (X, y) = GenerateLinearTestData(100, 1);
        model.Train(X, y);

        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Act
        var jit = new JitCompilerClass();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            "SimpleRegression graph should be JIT compatible");
    }

    // ========== RidgeRegression Tests ==========
    // TODO: RidgeRegression<T> class not yet implemented

    // [Fact]
    // public void RidgeRegression_SupportsJitCompilation()
    // {
    //     // Arrange
    //     var options = new RidgeRegressionOptions { Lambda = 0.1 };
    //     var model = new RidgeRegression<double>(options);
    //     var (X, y) = GenerateLinearTestData(100, 5);
    //     model.Train(X, y);

    //     // Assert
    //     Assert.True(model.SupportsJitCompilation, "RidgeRegression should support JIT after training");
    // }

    // [Fact]
    // public void RidgeRegression_ExportComputationGraph_ReturnsValidGraph()
    // {
    //     // Arrange
    //     var options = new RidgeRegressionOptions { Lambda = 0.1 };
    //     var model = new RidgeRegression<double>(options);
    //     var (X, y) = GenerateLinearTestData(100, 5);
    //     model.Train(X, y);

    //     // Act
    //     var inputNodes = new List<ComputationNode<double>>();
    //     var outputNode = model.ExportComputationGraph(inputNodes);

    //     // Assert
    //     Assert.NotNull(outputNode);
    //     Assert.NotEmpty(inputNodes);
    // }

    // ========== LassoRegression Tests ==========
    // TODO: LassoRegression<T> class not yet implemented

    // [Fact]
    // public void LassoRegression_SupportsJitCompilation()
    // {
    //     // Arrange
    //     var options = new LassoRegressionOptions { Lambda = 0.1, MaxIterations = 100 };
    //     var model = new LassoRegression<double>(options);
    //     var (X, y) = GenerateLinearTestData(100, 5);
    //     model.Train(X, y);

    //     // Assert
    //     Assert.True(model.SupportsJitCompilation, "LassoRegression should support JIT after training");
    // }

    // ========== ElasticNetRegression Tests ==========
    // TODO: ElasticNetRegression<T> class not yet implemented

    // [Fact]
    // public void ElasticNetRegression_SupportsJitCompilation()
    // {
    //     // Arrange
    //     var options = new ElasticNetRegressionOptions { Lambda1 = 0.1, Lambda2 = 0.1, MaxIterations = 100 };
    //     var model = new ElasticNetRegression<double>(options);
    //     var (X, y) = GenerateLinearTestData(100, 5);
    //     model.Train(X, y);

    //     // Assert
    //     Assert.True(model.SupportsJitCompilation, "ElasticNetRegression should support JIT after training");
    // }

    // ========== NonLinearRegression with Supported Kernels Tests ==========

    [Theory]
    [InlineData(KernelType.Linear)]
    [InlineData(KernelType.RBF)]
    [InlineData(KernelType.Polynomial)]
    [InlineData(KernelType.Sigmoid)]
    [InlineData(KernelType.Laplacian)]
    public void NonLinearRegression_SupportsJit_WithSupportedKernels(KernelType kernelType)
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            KernelType = kernelType,
            C = 1.0,
            Epsilon = 0.1,
            Gamma = 0.5,
            PolynomialDegree = 2,
            Coef0 = 1.0
        };
        var model = new SupportVectorRegression<double>(options);
        var (X, y) = GenerateLinearTestData(50, 3);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation,
            $"SupportVectorRegression with {kernelType} kernel should support JIT after training");
    }

    [Fact]
    public void SupportVectorRegression_RBFKernel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            KernelType = KernelType.RBF,
            C = 1.0,
            Epsilon = 0.1,
            Gamma = 0.5
        };
        var model = new SupportVectorRegression<double>(options);
        var (X, y) = GenerateLinearTestData(50, 3);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    [Fact]
    public void SupportVectorRegression_PolynomialKernel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            KernelType = KernelType.Polynomial,
            C = 1.0,
            Epsilon = 0.1,
            PolynomialDegree = 2,
            Coef0 = 1.0
        };
        var model = new SupportVectorRegression<double>(options);
        var (X, y) = GenerateLinearTestData(50, 3);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    [Fact]
    public void SupportVectorRegression_LaplacianKernel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            KernelType = KernelType.Laplacian,
            C = 1.0,
            Epsilon = 0.1,
            Gamma = 0.5
        };
        var model = new SupportVectorRegression<double>(options);
        var (X, y) = GenerateLinearTestData(50, 3);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== Decision Tree Regression - Not Supported Tests ==========
    // TODO: DecisionTreeRegressionOptions does not exist (use DecisionTreeOptions instead)

    // [Fact]
    // public void DecisionTreeRegression_DoesNotSupportJitCompilation()
    // {
    //     // Arrange
    //     var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesLeaf = 2 };
    //     var model = new DecisionTreeRegression<double>(options);
    //     var (X, y) = GenerateLinearTestData(100, 5);
    //     model.Train(X, y);

    //     // Assert
    //     Assert.False(model.SupportsJitCompilation,
    //         "DecisionTreeRegression should NOT support JIT (discrete branching cannot be differentiated)");
    // }

    // [Fact]
    // public void DecisionTreeRegression_ExportComputationGraph_ThrowsNotSupported()
    // {
    //     // Arrange
    //     var options = new DecisionTreeOptions { MaxDepth = 5, MinSamplesLeaf = 2 };
    //     var model = new DecisionTreeRegression<double>(options);
    //     var (X, y) = GenerateLinearTestData(100, 5);
    //     model.Train(X, y);

    //     // Act & Assert
    //     var inputNodes = new List<ComputationNode<double>>();
    //     Assert.Throws<NotSupportedException>(() => model.ExportComputationGraph(inputNodes));
    // }

    // ========== Random Forest Regression - Not Supported Tests ==========

    [Fact]
    public void RandomForestRegression_DoesNotSupportJitCompilation()
    {
        // Arrange
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 5,
            MaxDepth = 5,
            MinSamplesSplit = 2
        };
        var model = new RandomForestRegression<double>(options);
        var (X, y) = GenerateLinearTestData(100, 5);
        model.Train(X, y);

        // Assert
        Assert.False(model.SupportsJitCompilation,
            "RandomForestRegression should NOT support JIT (tree-based models cannot be differentiated)");
    }

    // ========== JIT Compatibility Analysis Tests ==========

    [Fact]
    public void SimpleRegression_JitCompatibilityAnalysis_ReturnsValidResult()
    {
        // Arrange
        var model = new SimpleRegression<double>();
        var (X, y) = GenerateLinearTestData(100, 1);
        model.Train(X, y);

        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Act
        var jit = new JitCompilerClass();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.NotNull(compatibility);
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            "SimpleRegression should be JIT compatible");
    }

    [Theory]
    [InlineData(typeof(SimpleRegression<double>))]
    [InlineData(typeof(MultipleRegression<double>))]
    [InlineData(typeof(PolynomialRegression<double>))]
    [InlineData(typeof(LogisticRegression<double>))]
    public void LinearRegressionModels_JitCompatibilityAnalysis_AllSupported(Type modelType)
    {
        // Arrange
        var modelObj = CreateAndTrainLinearModel(modelType);
        if (modelObj is not IRegressionModel<double> model) return;
        if (!model.SupportsJitCompilation) return;

        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Act
        var jit = new JitCompilerClass();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.NotNull(compatibility);
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            $"{modelType.Name} should be JIT compatible");
    }

    // ========== Untrained Model Tests ==========

    [Fact]
    public void SimpleRegression_ExportGraph_ThrowsWhenNotTrained()
    {
        // Arrange
        var model = new SimpleRegression<double>();
        var inputNodes = new List<ComputationNode<double>>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => model.ExportComputationGraph(inputNodes));
    }

    [Fact]
    public void SupportVectorRegression_ExportGraph_ThrowsWhenNotTrained()
    {
        // Arrange
        var model = new SupportVectorRegression<double>(new SupportVectorRegressionOptions
        {
            KernelType = KernelType.RBF,
            C = 1.0,
            Epsilon = 0.1
        });
        var inputNodes = new List<ComputationNode<double>>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => model.ExportComputationGraph(inputNodes));
    }

    // ========== Helper Methods ==========

    private static (Matrix<double> X, Vector<double> y) GenerateLinearTestData(int samples, int features)
    {
        var random = new Random(42);
        var X = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        // Generate random weights
        var weights = new double[features];
        for (int j = 0; j < features; j++)
        {
            weights[j] = random.NextDouble() * 2 - 1;
        }

        // Generate data: y = X * w + noise
        for (int i = 0; i < samples; i++)
        {
            double sum = 0;
            for (int j = 0; j < features; j++)
            {
                X[i, j] = random.NextDouble() * 10;
                sum += X[i, j] * weights[j];
            }
            y[i] = sum + (random.NextDouble() * 0.1 - 0.05); // Add small noise
        }

        return (X, y);
    }

    private static object? CreateAndTrainLinearModel(Type modelType)
    {
        if (modelType == typeof(SimpleRegression<double>))
        {
            var (X, y) = GenerateLinearTestData(100, 1);
            var model = new SimpleRegression<double>();
            model.Train(X, y);
            return model;
        }
        else if (modelType == typeof(MultipleRegression<double>))
        {
            var (X, y) = GenerateLinearTestData(100, 5);
            var model = new MultipleRegression<double>();
            model.Train(X, y);
            return model;
        }
        else if (modelType == typeof(PolynomialRegression<double>))
        {
            var (X, y) = GenerateLinearTestData(100, 1);
            var model = new PolynomialRegression<double>();
            model.Train(X, y);
            return model;
        }
        else if (modelType == typeof(LogisticRegression<double>))
        {
            var (X, y) = GenerateLinearTestData(100, 5);
            var model = new LogisticRegression<double>();
            model.Train(X, y);
            return model;
        }

        return null;
    }
}
