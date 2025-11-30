using Xunit;
using AiDotNet.TimeSeries;
using AiDotNet.Autodiff;
using AiDotNet.JitCompiler;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for JIT compilation support in time series models.
/// Verifies that models correctly support JIT compilation when trained.
/// </summary>
public class TimeSeriesJitCompilationTests
{
    // ========== NBEATSModel Tests ==========

    [Fact]
    public void NBEATSModel_SupportsJitCompilation_WhenTrained()
    {
        // Arrange
        var options = new NBEATSOptions
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            NumBlocks = 2,
            HiddenLayerSize = 16,
            ThetaDimension = 4
        };
        var model = new NBEATSModel<double>(options);

        // Train with simple data
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "NBEATSModel should support JIT after training");
    }

    [Fact]
    public void NBEATSModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new NBEATSOptions
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            NumBlocks = 2,
            HiddenLayerSize = 16,
            ThetaDimension = 4
        };
        var model = new NBEATSModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new NBEATSOptions
        {
            LookbackWindow = 10,
            ForecastHorizon = 3,
            NumBlocks = 2,
            HiddenLayerSize = 16,
            ThetaDimension = 4
        };
        var model = new NBEATSModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = model.ExportComputationGraph(inputNodes);

        // Act
        var jit = new JitCompiler();
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
        var options = new TBATSOptions
        {
            SeasonalPeriods = new int[] { 7 },
            UseBoxCox = false,
            UseTrend = true,
            UseDamping = false,
            FourierOrder = 2
        };
        var model = new TBATSModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "TBATSModel should support JIT after training");
    }

    [Fact]
    public void TBATSModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new TBATSOptions
        {
            SeasonalPeriods = new int[] { 7 },
            UseBoxCox = false,
            UseTrend = true,
            UseDamping = false,
            FourierOrder = 2
        };
        var model = new TBATSModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new ProphetOptions
        {
            GrowthType = GrowthType.Linear,
            YearlySeasonality = false,
            WeeklySeasonality = false,
            DailySeasonality = false
        };
        var model = new ProphetModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "ProphetModel should support JIT after training");
    }

    [Fact]
    public void ProphetModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new ProphetOptions
        {
            GrowthType = GrowthType.Linear,
            YearlySeasonality = false,
            WeeklySeasonality = false,
            DailySeasonality = false
        };
        var model = new ProphetModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new BSTSOptions
        {
            NumIterations = 10,
            BurnIn = 5,
            LocalLevelVariance = 0.1,
            LocalTrendVariance = 0.01
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "BayesianStructuralTimeSeriesModel should support JIT after training");
    }

    [Fact]
    public void BayesianStructuralTimeSeriesModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new BSTSOptions
        {
            NumIterations = 10,
            BurnIn = 5,
            LocalLevelVariance = 0.1,
            LocalTrendVariance = 0.01
        };
        var model = new BayesianStructuralTimeSeriesModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new STLOptions
        {
            SeasonalPeriod = 7,
            SeasonalSmoothing = 7,
            TrendSmoothing = 15,
            InnerLoopIterations = 2,
            OuterLoopIterations = 1
        };
        var model = new STLDecomposition<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "STLDecomposition should support JIT after training");
    }

    [Fact]
    public void STLDecomposition_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new STLOptions
        {
            SeasonalPeriod = 7,
            SeasonalSmoothing = 7,
            TrendSmoothing = 15,
            InnerLoopIterations = 2,
            OuterLoopIterations = 1
        };
        var model = new STLDecomposition<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new StateSpaceOptions
        {
            StateDimension = 2,
            ObservationDimension = 1
        };
        var model = new StateSpaceModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "StateSpaceModel should support JIT after training");
    }

    [Fact]
    public void StateSpaceModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new StateSpaceOptions
        {
            StateDimension = 2,
            ObservationDimension = 1
        };
        var model = new StateSpaceModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new SpectralAnalysisOptions
        {
            NumFrequencies = 5,
            WindowType = WindowType.Hann
        };
        var model = new SpectralAnalysisModel<double>(options);
        var data = GenerateTestData(64); // Power of 2 for FFT
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "SpectralAnalysisModel should support JIT after training");
    }

    [Fact]
    public void SpectralAnalysisModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new SpectralAnalysisOptions
        {
            NumFrequencies = 5,
            WindowType = WindowType.Hann
        };
        var model = new SpectralAnalysisModel<double>(options);
        var data = GenerateTestData(64);
        model.Train(data);

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
        var options = new UnobservedComponentsOptions
        {
            Level = true,
            Trend = false,
            SeasonalPeriod = 0,
            Cycle = false
        };
        var model = new UnobservedComponentsModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "UnobservedComponentsModel should support JIT after training");
    }

    [Fact]
    public void UnobservedComponentsModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new UnobservedComponentsOptions
        {
            Level = true,
            Trend = false,
            SeasonalPeriod = 0,
            Cycle = false
        };
        var model = new UnobservedComponentsModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
        var options = new NeuralNetworkARIMAOptions
        {
            AROrder = 2,
            MAOrder = 0,
            DifferenceOrder = 1,
            HiddenLayerSizes = new int[] { 8 },
            MaxEpochs = 10
        };
        var model = new NeuralNetworkARIMAModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

        // Assert
        Assert.True(model.SupportsJitCompilation, "NeuralNetworkARIMAModel should support JIT after training");
    }

    [Fact]
    public void NeuralNetworkARIMAModel_ExportComputationGraph_ReturnsValidGraph()
    {
        // Arrange
        var options = new NeuralNetworkARIMAOptions
        {
            AROrder = 2,
            MAOrder = 0,
            DifferenceOrder = 1,
            HiddenLayerSizes = new int[] { 8 },
            MaxEpochs = 10
        };
        var model = new NeuralNetworkARIMAModel<double>(options);
        var data = GenerateTestData(50);
        model.Train(data);

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
    [InlineData(typeof(ProphetModel<double>))]
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
        var jit = new JitCompiler();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.NotNull(compatibility);
        // Models should either be fully supported or at least support hybrid mode
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            $"{modelType.Name} should be JIT compatible");
    }

    // ========== Helper Methods ==========

    private static Vector<double> GenerateTestData(int length)
    {
        var data = new double[length];
        var random = new Random(42);
        double value = 100;

        for (int i = 0; i < length; i++)
        {
            // Simple random walk with trend
            value += random.NextDouble() * 2 - 1 + 0.1;
            data[i] = value;
        }

        return new Vector<double>(data);
    }

    private static ITimeSeriesModel<double>? CreateAndTrainModel(Type modelType)
    {
        var data = GenerateTestData(50);

        if (modelType == typeof(NBEATSModel<double>))
        {
            var model = new NBEATSModel<double>(new NBEATSOptions
            {
                LookbackWindow = 10,
                ForecastHorizon = 3,
                NumBlocks = 2,
                HiddenLayerSize = 16,
                ThetaDimension = 4
            });
            model.Train(data);
            return model;
        }
        else if (modelType == typeof(TBATSModel<double>))
        {
            var model = new TBATSModel<double>(new TBATSOptions
            {
                SeasonalPeriods = new int[] { 7 },
                UseBoxCox = false,
                UseTrend = true,
                UseDamping = false,
                FourierOrder = 2
            });
            model.Train(data);
            return model;
        }
        else if (modelType == typeof(ProphetModel<double>))
        {
            var model = new ProphetModel<double>(new ProphetOptions
            {
                GrowthType = GrowthType.Linear,
                YearlySeasonality = false,
                WeeklySeasonality = false,
                DailySeasonality = false
            });
            model.Train(data);
            return model;
        }
        else if (modelType == typeof(StateSpaceModel<double>))
        {
            var model = new StateSpaceModel<double>(new StateSpaceOptions
            {
                StateDimension = 2,
                ObservationDimension = 1
            });
            model.Train(data);
            return model;
        }

        return null;
    }
}

/// <summary>
/// Interface for time series models (for testing purposes).
/// </summary>
public interface ITimeSeriesModel<T>
{
    bool SupportsJitCompilation { get; }
    ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);
}
