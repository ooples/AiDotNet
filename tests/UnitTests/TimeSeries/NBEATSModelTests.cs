using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;

namespace AiDotNetTests.UnitTests.TimeSeries;

/// <summary>
/// Unit tests for the N-BEATS (Neural Basis Expansion Analysis for Time Series) model.
/// </summary>
public class NBEATSModelTests
{
    [Fact]
    public void Constructor_WithDefaultOptions_CreatesModel()
    {
        // Arrange & Act
        var model = new NBEATSModel<double>();

        // Assert
        Assert.NotNull(model);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void Constructor_WithCustomOptions_CreatesModel()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 7,
            ForecastHorizon = 3,
            NumStacks = 2,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 64,
            NumHiddenLayers = 2,
            UseInterpretableBasis = true,
            PolynomialDegree = 2
        };

        // Act
        var model = new NBEATSModel<double>(options);

        // Assert
        Assert.NotNull(model);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void Constructor_WithInvalidLookbackWindow_ThrowsArgumentException()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NBEATSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithInvalidForecastHorizon_ThrowsArgumentException()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            ForecastHorizon = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NBEATSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithInvalidNumStacks_ThrowsArgumentException()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            NumStacks = 0
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NBEATSModel<double>(options));
    }

    [Fact]
    public void Constructor_WithInvalidHiddenLayerSize_ThrowsArgumentException()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            HiddenLayerSize = -1
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NBEATSModel<double>(options));
    }

    [Fact]
    public void Train_WithValidData_CompletesSuccessfully()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5,
            ForecastHorizon = 2,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 32,
            NumHiddenLayers = 2,
            Epochs = 2,
            BatchSize = 4
        };
        var model = new NBEATSModel<double>(options);

        // Create simple synthetic time series data
        var x = new Matrix<double>(10, 5);
        var y = new Vector<double>(10);

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                x[i, j] = i + j;
            }
            y[i] = i + 5; // Next value in sequence
        }

        // Act
        model.Train(x, y);

        // Assert - model should be trained without throwing exceptions
        Assert.True(true);
    }

    [Fact]
    public void PredictSingle_WithValidInput_ReturnsPrediction()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 1
        };
        var model = new NBEATSModel<double>(options);

        // Create training data
        var x = new Matrix<double>(5, 5);
        var y = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                x[i, j] = i + j;
            }
            y[i] = i + 5;
        }

        model.Train(x, y);

        // Create test input
        var input = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            input[i] = i + 1.0;
        }

        // Act
        double prediction = model.PredictSingle(input);

        // Assert
        Assert.True(!double.IsNaN(prediction));
        Assert.True(!double.IsInfinity(prediction));
    }

    [Fact]
    public void PredictSingle_WithInvalidInputLength_ThrowsArgumentException()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5
        };
        var model = new NBEATSModel<double>(options);

        var invalidInput = new Vector<double>(3); // Wrong length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => model.PredictSingle(invalidInput));
    }

    [Fact]
    public void ForecastHorizon_WithValidInput_ReturnsMultiplePredictions()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1
        };
        var model = new NBEATSModel<double>(options);

        var input = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            input[i] = i + 1.0;
        }

        // Act
        var forecast = model.ForecastHorizon(input);

        // Assert
        Assert.NotNull(forecast);
        Assert.Equal(3, forecast.Length);

        for (int i = 0; i < forecast.Length; i++)
        {
            Assert.True(!double.IsNaN(forecast[i]));
            Assert.True(!double.IsInfinity(forecast[i]));
        }
    }

    [Fact]
    public void Predict_WithValidMatrix_ReturnsCorrectNumberOfPredictions()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 4,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            Epochs = 1
        };
        var model = new NBEATSModel<double>(options);

        var xTrain = new Matrix<double>(5, 4);
        var yTrain = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                xTrain[i, j] = i + j;
            }
            yTrain[i] = i + 4;
        }
        model.Train(xTrain, yTrain);

        var xTest = new Matrix<double>(3, 4);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                xTest[i, j] = i + j + 10;
            }
        }

        // Act
        var predictions = model.Predict(xTest);

        // Assert
        Assert.NotNull(predictions);
        Assert.Equal(3, predictions.Length);
    }

    [Fact]
    public void GetParameters_ReturnsCorrectParameterCount()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 3,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1
        };
        var model = new NBEATSModel<double>(options);

        // Act
        var parameters = model.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.Equal(model.ParameterCount, parameters.Length);
        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void SetParameters_WithValidParameters_UpdatesModel()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 3,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1
        };
        var model = new NBEATSModel<double>(options);

        var originalParams = model.GetParameters();
        var newParams = new Vector<double>(originalParams.Length);
        for (int i = 0; i < newParams.Length; i++)
        {
            newParams[i] = 0.5; // Set all parameters to 0.5
        }

        // Act
        model.SetParameters(newParams);
        var updatedParams = model.GetParameters();

        // Assert
        for (int i = 0; i < updatedParams.Length; i++)
        {
            Assert.Equal(0.5, updatedParams[i]);
        }
    }

    [Fact]
    public void SetParameters_WithInvalidParameterCount_ThrowsArgumentException()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 3,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1
        };
        var model = new NBEATSModel<double>(options);

        var invalidParams = new Vector<double>(10); // Wrong count

        // Act & Assert
        Assert.Throws<ArgumentException>(() => model.SetParameters(invalidParams));
    }

    [Fact]
    public void Serialize_Deserialize_PreservesModel()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 3,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1
        };
        var originalModel = new NBEATSModel<double>(options);

        var input = new Vector<double>(3);
        for (int i = 0; i < 3; i++)
        {
            input[i] = i + 1.0;
        }

        var originalPrediction = originalModel.PredictSingle(input);

        // Act
        byte[] serialized = originalModel.Serialize();
        var deserializedModel = new NBEATSModel<double>(options);
        deserializedModel.Deserialize(serialized);

        var deserializedPrediction = deserializedModel.PredictSingle(input);

        // Assert
        Assert.Equal(originalPrediction, deserializedPrediction, 5);
    }

    [Fact]
    public void SaveModel_LoadModel_PreservesModel()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 3,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1
        };
        var originalModel = new NBEATSModel<double>(options);

        var input = new Vector<double>(3);
        for (int i = 0; i < 3; i++)
        {
            input[i] = i + 1.0;
        }

        var originalPrediction = originalModel.PredictSingle(input);

        string tempFile = Path.Combine(Path.GetTempPath(), "nbeats_test_model.bin");

        try
        {
            // Act
            originalModel.SaveModel(tempFile);

            var loadedModel = new NBEATSModel<double>(options);
            loadedModel.LoadModel(tempFile);

            var loadedPrediction = loadedModel.PredictSingle(input);

            // Assert
            Assert.Equal(originalPrediction, loadedPrediction, 5);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5,
            ForecastHorizon = 3,
            NumStacks = 2,
            NumBlocksPerStack = 1
        };
        var model = new NBEATSModel<double>(options);

        // Act
        var metadata = model.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("N-BEATS", metadata.ModelName);
        Assert.Equal("Time Series Forecasting", metadata.ModelType);
        Assert.Equal(5, metadata.InputDimension);
        Assert.Equal(3, metadata.OutputDimension);
        Assert.True(metadata.ParameterCount > 0);
        Assert.NotNull(metadata.Hyperparameters);
    }

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 3,
            ForecastHorizon = 1,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 8,
            NumHiddenLayers = 1
        };
        var originalModel = new NBEATSModel<double>(options);

        // Act
        var clonedModel = (NBEATSModel<double>)originalModel.Clone();

        // Modify cloned model parameters
        var clonedParams = clonedModel.GetParameters();
        for (int i = 0; i < clonedParams.Length; i++)
        {
            clonedParams[i] = 1.0;
        }
        clonedModel.SetParameters(clonedParams);

        // Assert - Original model should remain unchanged
        var originalParams = originalModel.GetParameters();
        bool anyDifferent = false;
        for (int i = 0; i < originalParams.Length; i++)
        {
            if (Math.Abs(originalParams[i] - 1.0) > 0.001)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Original model parameters should not be affected by clone modification");
    }

    [Fact]
    public void InterpretableBasis_ProducesValidForecast()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            UseInterpretableBasis = true,
            PolynomialDegree = 2
        };
        var model = new NBEATSModel<double>(options);

        var input = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            input[i] = i + 1.0;
        }

        // Act
        var forecast = model.ForecastHorizon(input);

        // Assert
        Assert.NotNull(forecast);
        Assert.Equal(3, forecast.Length);

        for (int i = 0; i < forecast.Length; i++)
        {
            Assert.True(!double.IsNaN(forecast[i]));
            Assert.True(!double.IsInfinity(forecast[i]));
        }
    }

    [Fact]
    public void GenericBasis_ProducesValidForecast()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 5,
            ForecastHorizon = 3,
            NumStacks = 1,
            NumBlocksPerStack = 1,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            UseInterpretableBasis = false
        };
        var model = new NBEATSModel<double>(options);

        var input = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            input[i] = i + 1.0;
        }

        // Act
        var forecast = model.ForecastHorizon(input);

        // Assert
        Assert.NotNull(forecast);
        Assert.Equal(3, forecast.Length);

        for (int i = 0; i < forecast.Length; i++)
        {
            Assert.True(!double.IsNaN(forecast[i]));
            Assert.True(!double.IsInfinity(forecast[i]));
        }
    }

    [Fact]
    public void MultipleStacks_WorksCorrectly()
    {
        // Arrange
        var options = new NBEATSModelOptions<double>
        {
            LookbackWindow = 4,
            ForecastHorizon = 2,
            NumStacks = 3,
            NumBlocksPerStack = 2,
            HiddenLayerSize = 16,
            NumHiddenLayers = 2,
            Epochs = 1
        };
        var model = new NBEATSModel<double>(options);

        var xTrain = new Matrix<double>(5, 4);
        var yTrain = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                xTrain[i, j] = i + j;
            }
            yTrain[i] = i + 4;
        }

        // Act
        model.Train(xTrain, yTrain);

        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            input[i] = i + 1.0;
        }

        var prediction = model.PredictSingle(input);

        // Assert
        Assert.True(!double.IsNaN(prediction));
        Assert.True(!double.IsInfinity(prediction));
    }
}
