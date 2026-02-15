using AiDotNet.Diffusion;
using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Comprehensive unit tests for <see cref="DDPMModel{T}"/>.
/// </summary>
public class DDPMModelTests
{
    #region Construction Tests

    [Fact]
    public void Constructor_WithDefaultParameters_CreatesModel()
    {
        // Arrange & Act
        var model = new DDPMModel<double>();

        // Assert
        Assert.NotNull(model);
        Assert.NotNull(model.Scheduler);
        Assert.Equal(0, model.ParameterCount); // No neural network, so no parameters
    }

    [Fact]
    public void Constructor_WithScheduler_UsesProvidedScheduler()
    {
        // Arrange
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new DDIMScheduler<double>(config);

        // Act
        var model = new DDPMModel<double>(scheduler);

        // Assert
        Assert.Same(scheduler, model.Scheduler);
    }

    [Fact]
    public void Constructor_WithNullScheduler_UsesDefaultScheduler()
    {
        // Arrange & Act
        // Scheduler is nullable and defaults to DDIM scheduler if null
        var model = new DDPMModel<double>(options: null, scheduler: null);

        // Assert - model should use a default scheduler
        Assert.NotNull(model.Scheduler);
        Assert.IsType<DDIMScheduler<double>>(model.Scheduler);
    }

    [Fact]
    public void Constructor_WithNoisePredictor_UsesProvidedPredictor()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        bool predictorCalled = false;
        Func<Tensor<double>, int, Tensor<double>> predictor = (input, timestep) =>
        {
            predictorCalled = true;
            return new Tensor<double>(input.Shape, new Vector<double>(input.ToVector().Length));
        };

        // Act
        var model = new DDPMModel<double>(scheduler, predictor);
        var sample = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }));
        model.PredictNoise(sample, 500);

        // Assert
        Assert.True(predictorCalled);
    }

    [Fact]
    public void Constructor_WithSeed_ProducesReproducibleResults()
    {
        // Arrange
        int seed = 42;
        var model1 = new DDPMModel<double>(seed: seed);
        var model2 = new DDPMModel<double>(seed: seed);
        var shape = new[] { 1, 4 };

        // Act
        var result1 = model1.Generate(shape, numInferenceSteps: 5, seed: seed);
        var result2 = model2.Generate(shape, numInferenceSteps: 5, seed: seed);

        // Assert - Same seed should produce same results
        var vec1 = result1.ToVector();
        var vec2 = result2.ToVector();
        for (int i = 0; i < vec1.Length; i++)
        {
            Assert.Equal(vec1[i], vec2[i], precision: 10);
        }
    }

    #endregion

    #region PredictNoise Tests

    [Fact]
    public void PredictNoise_WithoutPredictor_ReturnsZeros()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var shape = new[] { 2, 3 };
        var sample = new Tensor<double>(shape, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));

        // Act
        var result = model.PredictNoise(sample, 500);

        // Assert - Default predictor returns zeros
        var resultVec = result.ToVector();
        Assert.Equal(sample.ToVector().Length, resultVec.Length);
        foreach (var v in resultVec)
        {
            Assert.Equal(0.0, v);
        }
    }

    [Fact]
    public void PredictNoise_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.PredictNoise(null!, 500));
    }

    [Fact]
    public void PredictNoise_WithCustomPredictor_CallsPredictor()
    {
        // Arrange
        int capturedTimestep = -1;
        Tensor<double>? capturedInput = null;

        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        var model = new DDPMModel<double>(scheduler, (input, timestep) =>
        {
            capturedInput = input;
            capturedTimestep = timestep;
            // Return input scaled by 0.5
            var vec = input.ToVector();
            var result = new Vector<double>(vec.Length);
            for (int i = 0; i < vec.Length; i++)
                result[i] = vec[i] * 0.5;
            return new Tensor<double>(input.Shape, result);
        });

        var sample = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));

        // Act
        var result = model.PredictNoise(sample, 750);

        // Assert
        Assert.Equal(750, capturedTimestep);
        Assert.NotNull(capturedInput);

        var resultVec = result.ToVector();
        Assert.Equal(0.5, resultVec[0]);
        Assert.Equal(1.0, resultVec[1]);
        Assert.Equal(1.5, resultVec[2]);
        Assert.Equal(2.0, resultVec[3]);
    }

    #endregion

    #region Generate Tests

    [Fact]
    public void Generate_WithValidShape_ReturnsCorrectShape()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var shape = new[] { 1, 3, 8, 8 };

        // Act
        var result = model.Generate(shape, numInferenceSteps: 10);

        // Assert
        Assert.Equal(shape, result.Shape);
    }

    [Fact]
    public void Generate_ReturnsFiniteValues()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var shape = new[] { 1, 16 };

        // Act
        var result = model.Generate(shape, numInferenceSteps: 10, seed: 42);

        // Assert
        var vec = result.ToVector();
        foreach (var v in vec)
        {
            Assert.False(double.IsNaN(v), "Generated sample contains NaN");
            Assert.False(double.IsInfinity(v), "Generated sample contains Infinity");
        }
    }

    [Fact]
    public void Generate_WithNullShape_ThrowsArgumentException()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => model.Generate(null!));
    }

    [Fact]
    public void Generate_WithEmptyShape_ThrowsArgumentException()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => model.Generate(Array.Empty<int>()));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Generate_WithInvalidInferenceSteps_ThrowsArgumentOutOfRangeException(int invalidSteps)
    {
        // Arrange
        var model = new DDPMModel<double>();
        var shape = new[] { 1, 4 };

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            model.Generate(shape, numInferenceSteps: invalidSteps));
    }

    [Theory]
    [InlineData(5)]
    [InlineData(20)]
    [InlineData(50)]
    public void Generate_WithVariousInferenceSteps_Succeeds(int steps)
    {
        // Arrange
        var model = new DDPMModel<double>();
        var shape = new[] { 1, 8 };

        // Act
        var result = model.Generate(shape, numInferenceSteps: steps, seed: 123);

        // Assert
        Assert.Equal(shape, result.Shape);
        var vec = result.ToVector();
        foreach (var v in vec)
        {
            Assert.False(double.IsNaN(v));
            Assert.False(double.IsInfinity(v));
        }
    }

    #endregion

    #region ComputeLoss Tests

    [Fact]
    public void ComputeLoss_WithValidInputs_ReturnsFiniteLoss()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var shape = new[] { 2, 4 };
        var cleanSamples = new Tensor<double>(shape, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }));
        var noise = new Tensor<double>(shape, new Vector<double>(new double[] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 }));
        var timesteps = new[] { 500 };

        // Act
        var loss = model.ComputeLoss(cleanSamples, noise, timesteps);

        // Assert
        Assert.False(double.IsNaN(loss));
        Assert.False(double.IsInfinity(loss));
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Fact]
    public void ComputeLoss_WithNullCleanSamples_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var noise = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }));
        var timesteps = new[] { 500 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.ComputeLoss(null!, noise, timesteps));
    }

    [Fact]
    public void ComputeLoss_WithNullNoise_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var cleanSamples = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }));
        var timesteps = new[] { 500 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.ComputeLoss(cleanSamples, null!, timesteps));
    }

    [Fact]
    public void ComputeLoss_WithEmptyTimesteps_ThrowsArgumentException()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var cleanSamples = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }));
        var noise = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }));

        // Act & Assert
        Assert.Throws<ArgumentException>(() => model.ComputeLoss(cleanSamples, noise, Array.Empty<int>()));
    }

    #endregion

    #region Parameter Management Tests

    [Fact]
    public void GetParameters_ReturnsEmptyVector_WhenNoNeuralNetwork()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act
        var parameters = model.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.Equal(0, parameters.Length);
    }

    [Fact]
    public void SetParameters_ThenGetParameters_ReturnsSetParameters()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var parameters = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 });

        // Act
        model.SetParameters(parameters);
        var retrieved = model.GetParameters();

        // Assert
        Assert.Equal(parameters.Length, retrieved.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i], retrieved[i]);
        }
    }

    [Fact]
    public void SetParameters_WithNull_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.SetParameters(null!));
    }

    [Fact]
    public void GetParameters_ReturnsCopy_NotReference()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        model.SetParameters(parameters);

        // Act
        var retrieved1 = model.GetParameters();
        var retrieved2 = model.GetParameters();

        // Assert - Modifications to one copy should not affect the other
        retrieved1[0] = 999.0;
        Assert.NotEqual(retrieved1[0], retrieved2[0]);
    }

    #endregion

    #region State Persistence Tests

    [Fact]
    public void SaveState_ThenLoadState_RestoresModel()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var parameters = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        model.SetParameters(parameters);

        using var stream = new MemoryStream();

        // Act
        model.SaveState(stream);
        stream.Position = 0;

        var loadedModel = new DDPMModel<double>();
        loadedModel.LoadState(stream);

        // Assert
        var loadedParams = loadedModel.GetParameters();
        Assert.Equal(parameters.Length, loadedParams.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i], loadedParams[i], precision: 10);
        }
    }

    [Fact]
    public void SaveState_WithNullStream_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.SaveState(null!));
    }

    [Fact]
    public void LoadState_WithNullStream_ThrowsArgumentNullException()
    {
        // Arrange
        var model = new DDPMModel<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => model.LoadState(null!));
    }

    [Fact]
    public void SaveState_WithNonWritableStream_ThrowsArgumentException()
    {
        // Arrange
        var model = new DDPMModel<double>();
        using var stream = new MemoryStream();
        stream.Close(); // Closed stream is not writable

        // Act & Assert
        Assert.Throws<ArgumentException>(() => model.SaveState(stream));
    }

    #endregion

    #region Clone Tests

    [Fact]
    public void Clone_ReturnsNewInstance()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        model.SetParameters(parameters);

        // Act
        var clone = model.Clone();

        // Assert
        Assert.NotSame(model, clone);
    }

    [Fact]
    public void Clone_CopiesParameters()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        model.SetParameters(parameters);

        // Act
        var clone = model.Clone();

        // Assert
        var cloneParams = clone.GetParameters();
        Assert.Equal(parameters.Length, cloneParams.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i], cloneParams[i]);
        }
    }

    [Fact]
    public void Clone_IsIndependent()
    {
        // Arrange
        var model = new DDPMModel<double>();
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        model.SetParameters(parameters);

        // Act
        var clone = model.Clone();
        clone.SetParameters(new Vector<double>(new double[] { 9.0, 9.0, 9.0 }));

        // Assert - Original should be unchanged
        var originalParams = model.GetParameters();
        Assert.Equal(1.0, originalParams[0]);
        Assert.Equal(2.0, originalParams[1]);
        Assert.Equal(3.0, originalParams[2]);
    }

    #endregion

    #region Factory Method Tests

    [Fact]
    public void Create_WithDefaultConfig_ReturnsWorkingModel()
    {
        // Arrange & Act
        var model = DDPMModel<double>.Create(SchedulerConfig<double>.CreateDefault());

        // Assert
        Assert.NotNull(model);
        Assert.NotNull(model.Scheduler);
    }

    [Fact]
    public void Create_WithStableDiffusionConfig_ReturnsWorkingModel()
    {
        // Arrange & Act
        var model = DDPMModel<double>.Create(SchedulerConfig<double>.CreateStableDiffusion());

        // Assert
        Assert.NotNull(model);
        var result = model.Generate(new[] { 1, 4 }, numInferenceSteps: 5, seed: 42);
        Assert.Equal(4, result.ToVector().Length);
    }

    [Fact]
    public void Create_WithCustomPredictor_UsesPredictor()
    {
        // Arrange
        bool predictorCalled = false;
        Func<Tensor<double>, int, Tensor<double>> predictor = (input, timestep) =>
        {
            predictorCalled = true;
            return new Tensor<double>(input.Shape, new Vector<double>(input.ToVector().Length));
        };

        // Act
        var model = DDPMModel<double>.Create(SchedulerConfig<double>.CreateDefault(), predictor);
        var sample = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 }));
        model.PredictNoise(sample, 500);

        // Assert
        Assert.True(predictorCalled);
    }

    #endregion

    #region Integration with Different Schedulers

    [Fact]
    public void Generate_WithDDIMScheduler_Succeeds()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        var model = new DDPMModel<double>(scheduler);
        var shape = new[] { 1, 8 };

        // Act
        var result = model.Generate(shape, numInferenceSteps: 10, seed: 42);

        // Assert
        Assert.Equal(shape, result.Shape);
    }

    [Fact]
    public void Generate_WithPNDMScheduler_Succeeds()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        var model = new DDPMModel<double>(scheduler);
        var shape = new[] { 1, 8 };

        // Act
        var result = model.Generate(shape, numInferenceSteps: 20, seed: 42);

        // Assert
        Assert.Equal(shape, result.Shape);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void DDPMModel_WithFloatType_Works()
    {
        // Arrange & Act
        var model = new DDPMModel<float>();
        var result = model.Generate(new[] { 1, 4 }, numInferenceSteps: 5, seed: 42);

        // Assert
        Assert.Equal(4, result.ToVector().Length);
        var vec = result.ToVector();
        foreach (var v in vec)
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
        }
    }

    #endregion
}
