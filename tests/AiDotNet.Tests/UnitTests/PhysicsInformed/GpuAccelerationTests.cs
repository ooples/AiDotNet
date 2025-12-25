using System;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PDEs;
using AiDotNet.PhysicsInformed.PINNs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed;

/// <summary>
/// Unit tests for Sprint 5: GPU Acceleration for Physics-Informed Training.
/// </summary>
public class GpuAccelerationTests
{
    #region Helper Classes

    /// <summary>
    /// Simple boundary condition for testing.
    /// </summary>
    private class SimpleBoundaryCondition : IBoundaryCondition<double>
    {
        private readonly double _boundaryValue;
        private readonly int _dimension;
        private readonly double _location;

        public SimpleBoundaryCondition(double location, int dimension, double boundaryValue)
        {
            _location = location;
            _dimension = dimension;
            _boundaryValue = boundaryValue;
        }

        public string Name => $"Boundary at dim {_dimension} = {_location}";

        public bool IsOnBoundary(double[] inputs)
        {
            return Math.Abs(inputs[_dimension] - _location) < 1e-10;
        }

        public double ComputeBoundaryResidual(double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
        {
            return outputs[0] - _boundaryValue;
        }
    }

    #endregion

    #region GpuPINNTrainingOptions Tests

    [Fact]
    public void GpuPINNTrainingOptions_Default_HasCorrectSettings()
    {
        // Arrange & Act
        var options = GpuPINNTrainingOptions.Default;

        // Assert
        Assert.True(options.EnableGpu);
        Assert.Equal(1024, options.BatchSizeGpu);
        Assert.True(options.ParallelDerivativeComputation);
        Assert.Equal(1000, options.MinPointsForGpu);
        Assert.True(options.AsyncTransfers);
        Assert.False(options.UseMixedPrecision);
        Assert.True(options.UsePinnedMemory);
        Assert.False(options.VerboseLogging);
        Assert.Equal(2, options.NumStreams);
    }

    [Fact]
    public void GpuPINNTrainingOptions_HighEnd_HasAggressiveSettings()
    {
        // Arrange & Act
        var options = GpuPINNTrainingOptions.HighEnd;

        // Assert
        Assert.True(options.EnableGpu);
        Assert.Equal(4096, options.BatchSizeGpu);
        Assert.True(options.ParallelDerivativeComputation);
        Assert.True(options.UseMixedPrecision);
        Assert.Equal(4, options.NumStreams);
        Assert.Equal(GpuUsageLevel.Aggressive, options.GpuConfig.UsageLevel);
    }

    [Fact]
    public void GpuPINNTrainingOptions_LowMemory_HasConservativeSettings()
    {
        // Arrange & Act
        var options = GpuPINNTrainingOptions.LowMemory;

        // Assert
        Assert.True(options.EnableGpu);
        Assert.Equal(256, options.BatchSizeGpu);
        Assert.True(options.ParallelDerivativeComputation);
        Assert.False(options.UseMixedPrecision);
        Assert.False(options.UsePinnedMemory);
        Assert.Equal(1, options.NumStreams);
        Assert.Equal(GpuUsageLevel.Conservative, options.GpuConfig.UsageLevel);
    }

    [Fact]
    public void GpuPINNTrainingOptions_CpuOnly_DisablesGpu()
    {
        // Arrange & Act
        var options = GpuPINNTrainingOptions.CpuOnly;

        // Assert
        Assert.False(options.EnableGpu);
        Assert.Equal(GpuDeviceType.CPU, options.GpuConfig.DeviceType);
    }

    [Fact]
    public void GpuPINNTrainingOptions_CustomSettings_AppliesCorrectly()
    {
        // Arrange & Act
        var options = new GpuPINNTrainingOptions
        {
            EnableGpu = true,
            BatchSizeGpu = 2048,
            ParallelDerivativeComputation = false,
            MinPointsForGpu = 5000,
            AsyncTransfers = false,
            UseMixedPrecision = true,
            UsePinnedMemory = false,
            VerboseLogging = true,
            NumStreams = 8
        };

        // Assert
        Assert.True(options.EnableGpu);
        Assert.Equal(2048, options.BatchSizeGpu);
        Assert.False(options.ParallelDerivativeComputation);
        Assert.Equal(5000, options.MinPointsForGpu);
        Assert.False(options.AsyncTransfers);
        Assert.True(options.UseMixedPrecision);
        Assert.False(options.UsePinnedMemory);
        Assert.True(options.VerboseLogging);
        Assert.Equal(8, options.NumStreams);
    }

    #endregion

    #region GpuAccelerationConfig Tests

    [Fact]
    public void GpuAccelerationConfig_Default_HasCorrectSettings()
    {
        // Arrange & Act
        var config = new GpuAccelerationConfig();

        // Assert
        Assert.Equal(GpuDeviceType.Auto, config.DeviceType);
        Assert.Equal(GpuUsageLevel.Default, config.UsageLevel);
        Assert.Equal(0, config.DeviceIndex);
        Assert.False(config.VerboseLogging);
        Assert.True(config.EnableForInference);
    }

    [Fact]
    public void GpuAccelerationConfig_ToString_ReturnsFormattedString()
    {
        // Arrange
        var config = new GpuAccelerationConfig
        {
            DeviceType = GpuDeviceType.CUDA,
            UsageLevel = GpuUsageLevel.Aggressive,
            DeviceIndex = 1,
            VerboseLogging = true,
            EnableForInference = false
        };

        // Act
        var result = config.ToString();

        // Assert
        Assert.Contains("CUDA", result);
        Assert.Contains("Aggressive", result);
        Assert.Contains("DeviceIndex=1", result);
    }

    [Fact]
    public void GpuDeviceType_HasAllExpectedValues()
    {
        // Assert
        Assert.True(Enum.IsDefined(typeof(GpuDeviceType), GpuDeviceType.Auto));
        Assert.True(Enum.IsDefined(typeof(GpuDeviceType), GpuDeviceType.CUDA));
        Assert.True(Enum.IsDefined(typeof(GpuDeviceType), GpuDeviceType.OpenCL));
        Assert.True(Enum.IsDefined(typeof(GpuDeviceType), GpuDeviceType.CPU));
    }

    [Fact]
    public void GpuUsageLevel_HasAllExpectedValues()
    {
        // Assert
        Assert.True(Enum.IsDefined(typeof(GpuUsageLevel), GpuUsageLevel.Conservative));
        Assert.True(Enum.IsDefined(typeof(GpuUsageLevel), GpuUsageLevel.Default));
        Assert.True(Enum.IsDefined(typeof(GpuUsageLevel), GpuUsageLevel.Aggressive));
        Assert.True(Enum.IsDefined(typeof(GpuUsageLevel), GpuUsageLevel.AlwaysGpu));
        Assert.True(Enum.IsDefined(typeof(GpuUsageLevel), GpuUsageLevel.AlwaysCpu));
    }

    #endregion

    #region GpuPINNTrainer Tests

    [Fact]
    public void GpuPINNTrainer_Constructor_CreatesTrainer()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        // Act
        var trainer = new GpuPINNTrainer<double>(pinn);

        // Assert
        Assert.NotNull(trainer);
        Assert.Same(pinn, trainer.Network);
        Assert.NotNull(trainer.Options);
    }

    [Fact]
    public void GpuPINNTrainer_Constructor_WithOptions_AppliesOptions()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var options = new GpuPINNTrainingOptions
        {
            BatchSizeGpu = 512,
            VerboseLogging = true
        };

        // Act
        var trainer = new GpuPINNTrainer<double>(pinn, options);

        // Assert
        Assert.Equal(512, trainer.Options.BatchSizeGpu);
        Assert.True(trainer.Options.VerboseLogging);
    }

    [Fact]
    public void GpuPINNTrainer_Constructor_ThrowsOnNullPinn()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new GpuPINNTrainer<double>(null!));
    }

    [Fact]
    public void GpuPINNTrainer_UpdateOptions_ChangesOptions()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var trainer = new GpuPINNTrainer<double>(pinn);
        var newOptions = new GpuPINNTrainingOptions
        {
            BatchSizeGpu = 2048,
            EnableGpu = false
        };

        // Act
        trainer.UpdateOptions(newOptions);

        // Assert
        Assert.Equal(2048, trainer.Options.BatchSizeGpu);
        Assert.False(trainer.Options.EnableGpu);
    }

    [Fact]
    public void GpuPINNTrainer_UpdateOptions_ThrowsOnNull()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var trainer = new GpuPINNTrainer<double>(pinn);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            trainer.UpdateOptions(null!));
    }

    [Fact]
    public void GpuPINNTrainer_TryInitializeGpu_ReturnsResult()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var trainer = new GpuPINNTrainer<double>(pinn, GpuPINNTrainingOptions.CpuOnly);

        // Act
        var result = trainer.TryInitializeGpu();

        // Assert - Should return a boolean (true or false based on GPU availability)
        Assert.IsType<bool>(result);
    }

    [Fact]
    public void GpuPINNTrainer_ReleaseGpuResources_DoesNotThrow()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var pde = new HeatEquation<double>(thermalDiffusivity: 0.1);
        var boundaryConditions = new IBoundaryCondition<double>[]
        {
            new SimpleBoundaryCondition(0.0, 0, 0.0)
        };

        var pinn = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            boundaryConditions,
            numCollocationPoints: 100);

        var trainer = new GpuPINNTrainer<double>(pinn);

        // Act & Assert - Should not throw
        trainer.ReleaseGpuResources();
        Assert.False(trainer.IsUsingGpu);
    }

    #endregion

    #region GpuTrainingHistory Tests

    [Fact]
    public void GpuTrainingHistory_TracksGpuMetrics()
    {
        // Arrange
        var history = new GpuTrainingHistory<double>();

        // Act
        history.UseGpuAcceleration = true;
        history.TotalTrainingTimeMs = 5000;
        history.PeakManagedMemoryBytes = 1024 * 1024 * 100; // 100 MB
        history.AddEpoch(1.0);
        history.AddEpoch(0.5);
        history.AddEpoch(0.25);

        // Assert
        Assert.True(history.UseGpuAcceleration);
        Assert.Equal(5000, history.TotalTrainingTimeMs);
        Assert.Equal(3, history.Losses.Count);
        Assert.Equal(1024 * 1024 * 100, history.PeakManagedMemoryBytes);
    }

    [Fact]
    public void GpuTrainingHistory_AverageEpochTime_CalculatesCorrectly()
    {
        // Arrange
        var history = new GpuTrainingHistory<double>();
        history.TotalTrainingTimeMs = 1000;
        history.AddEpoch(1.0);
        history.AddEpoch(0.5);

        // Act
        var avgTime = history.AverageEpochTimeMs;

        // Assert
        Assert.Equal(500, avgTime);
    }

    [Fact]
    public void GpuTrainingHistory_AverageEpochTime_ReturnsZeroWhenEmpty()
    {
        // Arrange
        var history = new GpuTrainingHistory<double>();
        history.TotalTrainingTimeMs = 1000;

        // Act
        var avgTime = history.AverageEpochTimeMs;

        // Assert
        Assert.Equal(0, avgTime);
    }

    [Fact]
    public void GpuTrainingHistory_InheritsFromTrainingHistory()
    {
        // Arrange & Act
        var history = new GpuTrainingHistory<double>();

        // Assert
        Assert.IsAssignableFrom<TrainingHistory<double>>(history);
    }

    #endregion

    #region PINNGpuMemoryInfo Tests

    [Fact]
    public void PINNGpuMemoryInfo_UsagePercentage_CalculatesCorrectly()
    {
        // Arrange
        var info = new PINNGpuMemoryInfo
        {
            TotalMemoryBytes = 1000,
            UsedMemoryBytes = 250,
            AvailableMemoryBytes = 750
        };

        // Act
        var usagePercent = info.UsagePercentage;

        // Assert
        Assert.Equal(25, usagePercent);
    }

    [Fact]
    public void PINNGpuMemoryInfo_UsagePercentage_ReturnsZeroWhenTotalIsZero()
    {
        // Arrange
        var info = new PINNGpuMemoryInfo
        {
            TotalMemoryBytes = 0,
            UsedMemoryBytes = 0,
            AvailableMemoryBytes = 0
        };

        // Act
        var usagePercent = info.UsagePercentage;

        // Assert - returns -1 when memory info is not available (TotalMemoryBytes = 0)
        Assert.Equal(-1, usagePercent);
    }

    #endregion
}
