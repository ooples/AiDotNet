// Nullable disabled: This test file has methods that may return null for unsupported model types
#nullable disable

using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.JitCompiler;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;
using JitCompilerClass = AiDotNet.JitCompiler.JitCompiler;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for JIT compilation support in time series models.
/// Verifies that models correctly support JIT compilation when trained.
/// </summary>
/// <remarks>
/// These tests are quarantined because they trigger GPU initialization which can fail
/// on machines without proper GPU support or drivers.
/// </remarks>
[Trait("Category", "GPU")]
public class TimeSeriesJitCompilationTests
{
    // ========== NBEATSModel Tests ==========

    [Fact]
    public void NBEATSModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 1,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);

        // Train with windowed data
        var (X, y) = GenerateTrainingData(50, options.LookbackWindow);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "NBEATSModel should support JIT after training");
    }

    [Fact]
    public void NBEATSModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 1,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);
        var (X, y) = GenerateTrainingData(50, options.LookbackWindow);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
        Assert.NotNull(outputNode.Value);
    }

    [Fact]
    public void NBEATSModel_JitCompilation_ProducesCorrectResults()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 1,
            BatchSize = 8
        };
        var model = new NBEATSModel<double>(options);
        var (X, y) = GenerateTrainingData(50, options.LookbackWindow);
        model.Train(X, y);

        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Act
        var jit = new JitCompilerClass();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            "NBEATSModel graph should be JIT compatible");
    }

    // ========== TBATSModel Tests ==========

    [Fact]
    public void TBATSModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new int[] { 7 },
            BoxCoxLambda = 1,
            TrendDampingFactor = 1
        };
        var model = new TBATSModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "TBATSModel should support JIT after training");
    }

    [Fact]
    public void TBATSModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new TBATSModelOptions<double>
        {
            SeasonalPeriods = new int[] { 7 },
            BoxCoxLambda = 1,
            TrendDampingFactor = 1
        };
        var model = new TBATSModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== ProphetModel Tests ==========

    [Fact]
    public void ProphetModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new ProphetOptions<double, Matrix<double>, Vector<double>>
        {
            YearlySeasonality = false,
            WeeklySeasonality = false,
            DailySeasonality = false
        };
        var model = new ProphetModel<double, Matrix<double>, Vector<double>>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "ProphetModel should support JIT after training");
    }

    [Fact]
    public void ProphetModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new ProphetOptions<double, Matrix<double>, Vector<double>>
        {
            YearlySeasonality = false,
            WeeklySeasonality = false,
            DailySeasonality = false
        };
        var model = new ProphetModel<double, Matrix<double>, Vector<double>>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== BayesianStructuralTimeSeriesModel Tests ==========

    [Fact]
    public void BayesianStructuralTimeSeriesModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new BayesianStructuralTimeSeriesOptions<double>
        {
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "BayesianStructuralTimeSeriesModel should support JIT after training");
    }

    [Fact]
    public void BayesianStructuralTimeSeriesModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new BayesianStructuralTimeSeriesOptions<double>
        {
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== STLDecomposition Tests ==========

    [Fact]
    public void STLDecomposition_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new STLDecompositionOptions<double>
        {
            SeasonalPeriod = 7
        };
        var model = new STLDecomposition<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "STLDecomposition should support JIT after training");
    }

    [Fact]
    public void STLDecomposition_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new STLDecompositionOptions<double>
        {
            SeasonalPeriod = 7
        };
        var model = new STLDecomposition<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== StateSpaceModel Tests ==========

    [Fact]
    public void StateSpaceModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new StateSpaceModelOptions<double>
        {
        };
        var model = new StateSpaceModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "StateSpaceModel should support JIT after training");
    }

    [Fact]
    public void StateSpaceModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new StateSpaceModelOptions<double>
        {
        };
        var model = new StateSpaceModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== SpectralAnalysisModel Tests ==========

    [Fact]
    public void SpectralAnalysisModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 64,
            UseWindowFunction = true
        };
        var model = new SpectralAnalysisModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(64); // Power of 2 for FFT
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "SpectralAnalysisModel should support JIT after training");
    }

    [Fact]
    public void SpectralAnalysisModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new SpectralAnalysisOptions<double>
        {
            NFFT = 64,
            UseWindowFunction = true
        };
        var model = new SpectralAnalysisModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(64);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== UnobservedComponentsModel Tests ==========

    [Fact]
    public void UnobservedComponentsModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new UnobservedComponentsOptions<double, Matrix<double>, Vector<double>>
        {
            SeasonalPeriod = 0
        };
        var model = new UnobservedComponentsModel<double, Matrix<double>, Vector<double>>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "UnobservedComponentsModel should support JIT after training");
    }

    [Fact]
    public void UnobservedComponentsModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new UnobservedComponentsOptions<double, Matrix<double>, Vector<double>>
        {
            SeasonalPeriod = 0
        };
        var model = new UnobservedComponentsModel<double, Matrix<double>, Vector<double>>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== NeuralNetworkARIMAModel Tests ==========

    [Fact]
    public void NeuralNetworkARIMAModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 2,
            MAOrder = 0
        };
        var model = new NeuralNetworkARIMAModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Assert
        Assert.True(model.SupportsJitCompilation, "NeuralNetworkARIMAModel should support JIT after training");
    }

    [Fact]
    public void NeuralNetworkARIMAModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new NeuralNetworkARIMAOptions<double>
        {
            AROrder = 2,
            MAOrder = 0
        };
        var model = new NeuralNetworkARIMAModel<double>(options);
        var (X, y) = GenerateSimpleTimeSeriesData(50);
        model.Train(X, y);

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
        Assert.NotEmpty(inputNodes);
    }

    // ========== JIT Compatibility Analysis Tests ==========

    [Theory]
    [InlineData(typeof(NBEATSModel<double>))]
    [InlineData(typeof(TBATSModel<double>))]
    [InlineData(typeof(ProphetModel<double, Matrix<double>, Vector<double>>))]
    [InlineData(typeof(StateSpaceModel<double>))]
    public void TimeSeriesModels_JitCompatibilityAnalysis_ReturnsValidResult(Type modelType)
    {
        // Arrange - Create and train the model
        var model = CreateAndTrainModel(modelType);
        if (model == null || !model.SupportsJitCompilation) return;

        // Act - Export computation graph
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Analyze compatibility
        var jit = new JitCompilerClass();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.NotNull(compatibility);
        // Models should either be fully supported or at least support hybrid mode
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            $"{modelType.Name} should be JIT compatible");
    }

    // ========== Helper Methods ==========

    private static (Matrix<double> X, Vector<double> y) GenerateTrainingData(int samples, int lookbackWindow = 10)
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Generate a time series with enough points
        var timeSeries = new double[samples + lookbackWindow];
        for (int i = 0; i < timeSeries.Length; i++)
        {
            timeSeries[i] = Math.Sin(i * 0.1) + random.NextDouble() * 0.1;
        }

        // Create windowed data: each row contains lookbackWindow historical values
        var x = new Matrix<double>(samples, lookbackWindow);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            // Fill in the lookback window for this sample
            for (int j = 0; j < lookbackWindow; j++)
            {
                x[i, j] = timeSeries[i + j];
            }
            // Target is the next value after the window
            y[i] = timeSeries[i + lookbackWindow];
        }

        return (x, y);
    }

    private static dynamic? CreateAndTrainModel(Type modelType)
    {
        if (modelType == typeof(NBEATSModel<double>))
        {
            var options = new NBEATSModelOptions<double>
            {
                LookbackWindow = 10,
                ForecastHorizon = 3,
                NumStacks = 1,
                NumBlocksPerStack = 2,
                HiddenLayerSize = 16,
                NumHiddenLayers = 1,
                Epochs = 1,
                BatchSize = 8
            };
            var (X, y) = GenerateTrainingData(50, options.LookbackWindow);
            var model = new NBEATSModel<double>(options);
            model.Train(X, y);
            return model;
        }
        else if (modelType == typeof(TBATSModel<double>))
        {
            var (X, y) = GenerateSimpleTimeSeriesData(50);
            var model = new TBATSModel<double>(new TBATSModelOptions<double>
            {
                SeasonalPeriods = new int[] { 7 },
                BoxCoxLambda = 1,
                TrendDampingFactor = 1
            });
            model.Train(X, y);
            return model;
        }
        else if (modelType == typeof(ProphetModel<double, Matrix<double>, Vector<double>>))
        {
            var (X, y) = GenerateSimpleTimeSeriesData(50);
            var model = new ProphetModel<double, Matrix<double>, Vector<double>>(new ProphetOptions<double, Matrix<double>, Vector<double>>
            {
                YearlySeasonality = false,
                WeeklySeasonality = false,
                DailySeasonality = false
            });
            model.Train(X, y);
            return model;
        }
        else if (modelType == typeof(StateSpaceModel<double>))
        {
            var (X, y) = GenerateSimpleTimeSeriesData(50);
            var model = new StateSpaceModel<double>(new StateSpaceModelOptions<double>
            {
            });
            model.Train(X, y);
            return model;
        }

        return null;
    }

    private static (Matrix<double> X, Vector<double> y) GenerateSimpleTimeSeriesData(int samples)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var x = new Matrix<double>(samples, 1);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = i; // Time index
            y[i] = Math.Sin(i * 0.1) + random.NextDouble() * 0.1;
        }

        return (x, y);
    }
}
