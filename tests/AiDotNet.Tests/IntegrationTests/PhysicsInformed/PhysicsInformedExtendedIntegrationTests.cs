using AiDotNet.Engines;
using AiDotNet.PhysicsInformed;
using AiDotNet.PhysicsInformed.Benchmarks;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.PhysicsInformed.PDEs;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.PhysicsInformed;

/// <summary>
/// Extended integration tests for PhysicsInformed module covering benchmark classes,
/// training histories, GPU options, PDE equations, and Options classes.
/// </summary>
public class PhysicsInformedExtendedIntegrationTests
{
    private const double Tolerance = 1e-8;

    #region OperatorBenchmarkOptions

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_DefaultValues()
    {
        var opts = new OperatorBenchmarkOptions();

        Assert.Equal(64, opts.SpatialPoints);
        Assert.Equal(32, opts.SampleCount);
        Assert.Equal(3, opts.MaxFrequency);
        Assert.Equal(5, opts.SmoothingWindow);
        Assert.Equal(42, opts.Seed);
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_CustomValues()
    {
        var opts = new OperatorBenchmarkOptions
        {
            SpatialPoints = 128,
            SampleCount = 64,
            MaxFrequency = 5,
            SmoothingWindow = 7,
            Seed = 123
        };

        Assert.Equal(128, opts.SpatialPoints);
        Assert.Equal(64, opts.SampleCount);
        Assert.Equal(5, opts.MaxFrequency);
        Assert.Equal(7, opts.SmoothingWindow);
        Assert.Equal(123, opts.Seed);
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_Validate_ValidOptions()
    {
        var opts = new OperatorBenchmarkOptions();
        opts.Validate(); // Should not throw
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_Validate_SpatialPointsTooSmall_Throws()
    {
        var opts = new OperatorBenchmarkOptions { SpatialPoints = 3 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_Validate_SampleCountZero_Throws()
    {
        var opts = new OperatorBenchmarkOptions { SampleCount = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_Validate_MaxFrequencyZero_Throws()
    {
        var opts = new OperatorBenchmarkOptions { MaxFrequency = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkOptions_Validate_SmoothingWindowZero_Throws()
    {
        var opts = new OperatorBenchmarkOptions { SmoothingWindow = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    #endregion

    #region OperatorBenchmarkResult

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkResult_DefaultValues()
    {
        var result = new OperatorBenchmarkResult();

        Assert.Equal(string.Empty, result.OperatorName);
        Assert.Equal(0, result.SpatialPoints);
        Assert.Equal(0, result.SampleCount);
        Assert.Equal(0.0, result.Mse);
        Assert.Equal(0.0, result.L2Error);
        Assert.Equal(0.0, result.RelativeL2Error);
        Assert.Equal(0.0, result.MaxError);
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorBenchmarkResult_CustomValues()
    {
        var result = new OperatorBenchmarkResult
        {
            OperatorName = "TestOp",
            SpatialPoints = 64,
            SampleCount = 32,
            Mse = 0.001,
            L2Error = 0.01,
            RelativeL2Error = 0.05,
            MaxError = 0.1
        };

        Assert.Equal("TestOp", result.OperatorName);
        Assert.Equal(64, result.SpatialPoints);
        Assert.Equal(32, result.SampleCount);
        Assert.Equal(0.001, result.Mse);
        Assert.Equal(0.01, result.L2Error);
        Assert.Equal(0.05, result.RelativeL2Error);
        Assert.Equal(0.1, result.MaxError);
    }

    #endregion

    #region OperatorDataset2D

    [Fact(Timeout = 120000)]
    public async Task OperatorDataset2D_DefaultValues()
    {
        var dataset = new OperatorDataset2D();

        Assert.Equal(string.Empty, dataset.OperatorName);
        Assert.Equal(0, dataset.GridSize);
        Assert.Equal(0, dataset.SampleCount);
        Assert.NotNull(dataset.Inputs);
        Assert.NotNull(dataset.Outputs);
    }

    [Fact(Timeout = 120000)]
    public async Task OperatorDataset2D_CustomValues()
    {
        var inputs = new double[2, 4, 4];
        var outputs = new double[2, 4, 4];
        var dataset = new OperatorDataset2D
        {
            OperatorName = "Darcy",
            GridSize = 4,
            SampleCount = 2,
            Inputs = inputs,
            Outputs = outputs
        };

        Assert.Equal("Darcy", dataset.OperatorName);
        Assert.Equal(4, dataset.GridSize);
        Assert.Equal(2, dataset.SampleCount);
        Assert.Same(inputs, dataset.Inputs);
        Assert.Same(outputs, dataset.Outputs);
    }

    #endregion

    #region PdeBenchmarkOptions

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkOptions_DefaultValues()
    {
        var opts = new PdeBenchmarkOptions();

        Assert.Equal(64, opts.SpatialPoints);
        Assert.Equal(200, opts.TimeSteps);
        Assert.Equal(-1.0, opts.DomainStart);
        Assert.Equal(1.0, opts.DomainEnd);
        Assert.Equal(1.0, opts.FinalTime);
    }

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkOptions_Validate_ValidOptions()
    {
        var opts = new PdeBenchmarkOptions();
        opts.Validate(); // Should not throw
    }

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkOptions_Validate_SpatialPointsTooSmall_Throws()
    {
        var opts = new PdeBenchmarkOptions { SpatialPoints = 2 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkOptions_Validate_TimeStepsZero_Throws()
    {
        var opts = new PdeBenchmarkOptions { TimeSteps = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkOptions_Validate_DomainEndLessThanStart_Throws()
    {
        var opts = new PdeBenchmarkOptions { DomainStart = 1.0, DomainEnd = -1.0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkOptions_Validate_FinalTimeZero_Throws()
    {
        var opts = new PdeBenchmarkOptions { FinalTime = 0.0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    #endregion

    #region BurgersBenchmarkOptions

    [Fact(Timeout = 120000)]
    public async Task BurgersBenchmarkOptions_DefaultValues()
    {
        var opts = new BurgersBenchmarkOptions();

        Assert.Equal(0.01, opts.Viscosity);
        Assert.NotNull(opts.InitialCondition);
    }

    [Fact(Timeout = 120000)]
    public async Task BurgersBenchmarkOptions_Validate_ValidOptions()
    {
        var opts = new BurgersBenchmarkOptions();
        opts.Validate(); // Should not throw
    }

    [Fact(Timeout = 120000)]
    public async Task BurgersBenchmarkOptions_Validate_NegativeViscosity_Throws()
    {
        var opts = new BurgersBenchmarkOptions { Viscosity = -0.01 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task BurgersBenchmarkOptions_InitialCondition_EvaluatesCorrectly()
    {
        var opts = new BurgersBenchmarkOptions();
        double result = opts.InitialCondition(0.0);
        Assert.Equal(0.0, result, Tolerance); // -sin(0) = 0
    }

    #endregion

    #region AllenCahnBenchmarkOptions

    [Fact(Timeout = 120000)]
    public async Task AllenCahnBenchmarkOptions_DefaultValues()
    {
        var opts = new AllenCahnBenchmarkOptions();

        Assert.Equal(0.01, opts.Epsilon);
        Assert.NotNull(opts.InitialCondition);
    }

    [Fact(Timeout = 120000)]
    public async Task AllenCahnBenchmarkOptions_Validate_ValidOptions()
    {
        var opts = new AllenCahnBenchmarkOptions();
        opts.Validate(); // Should not throw
    }

    [Fact(Timeout = 120000)]
    public async Task AllenCahnBenchmarkOptions_Validate_EpsilonZero_Throws()
    {
        var opts = new AllenCahnBenchmarkOptions { Epsilon = 0.0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task AllenCahnBenchmarkOptions_InitialCondition_EvaluatesCorrectly()
    {
        var opts = new AllenCahnBenchmarkOptions();
        double result = opts.InitialCondition(0.0);
        Assert.Equal(0.0, result, Tolerance); // 0^2 * cos(0) = 0
    }

    #endregion

    #region DarcyOperatorBenchmarkOptions

    [Fact(Timeout = 120000)]
    public async Task DarcyOperatorBenchmarkOptions_DefaultValues()
    {
        var opts = new DarcyOperatorBenchmarkOptions();

        Assert.Equal(64, opts.GridSize);
        Assert.Equal(16, opts.SampleCount);
        Assert.Equal(4, opts.MaxFrequency);
        Assert.Equal(3000, opts.MaxIterations);
        Assert.Equal(1e-6, opts.Tolerance);
        Assert.Equal(1.0, opts.ForcingValue);
        Assert.Equal(0.5, opts.LogPermeabilityScale);
        Assert.Equal(42, opts.Seed);
    }

    [Fact(Timeout = 120000)]
    public async Task DarcyOperatorBenchmarkOptions_Validate_ValidOptions()
    {
        var opts = new DarcyOperatorBenchmarkOptions();
        opts.Validate(); // Should not throw
    }

    [Fact(Timeout = 120000)]
    public async Task DarcyOperatorBenchmarkOptions_Validate_GridSizeTooSmall_Throws()
    {
        var opts = new DarcyOperatorBenchmarkOptions { GridSize = 3 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task DarcyOperatorBenchmarkOptions_Validate_ToleranceZero_Throws()
    {
        var opts = new DarcyOperatorBenchmarkOptions { Tolerance = 0.0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task DarcyOperatorBenchmarkOptions_Validate_LogPermeabilityScaleZero_Throws()
    {
        var opts = new DarcyOperatorBenchmarkOptions { LogPermeabilityScale = 0.0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    #endregion

    #region PoissonOperatorBenchmarkOptions

    [Fact(Timeout = 120000)]
    public async Task PoissonOperatorBenchmarkOptions_DefaultValues()
    {
        var opts = new PoissonOperatorBenchmarkOptions();

        Assert.Equal(64, opts.GridSize);
        Assert.Equal(16, opts.SampleCount);
        Assert.Equal(4, opts.MaxFrequency);
        Assert.Equal(2000, opts.MaxIterations);
        Assert.Equal(1e-6, opts.Tolerance);
        Assert.Equal(42, opts.Seed);
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonOperatorBenchmarkOptions_Validate_ValidOptions()
    {
        var opts = new PoissonOperatorBenchmarkOptions();
        opts.Validate(); // Should not throw
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonOperatorBenchmarkOptions_Validate_SampleCountZero_Throws()
    {
        var opts = new PoissonOperatorBenchmarkOptions { SampleCount = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonOperatorBenchmarkOptions_Validate_MaxIterationsZero_Throws()
    {
        var opts = new PoissonOperatorBenchmarkOptions { MaxIterations = 0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => opts.Validate());
    }

    #endregion

    #region PdeBenchmarkResult

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkResult_DefaultValues()
    {
        var result = new PdeBenchmarkResult();

        Assert.Equal(string.Empty, result.EquationName);
        Assert.Equal(0, result.SpatialPoints);
        Assert.Equal(0, result.TimeSteps);
        Assert.Equal(0.0, result.FinalTime);
        Assert.Equal(0.0, result.L2Error);
        Assert.Equal(0.0, result.MaxError);
    }

    [Fact(Timeout = 120000)]
    public async Task PdeBenchmarkResult_CustomValues()
    {
        var result = new PdeBenchmarkResult
        {
            EquationName = "Burgers",
            SpatialPoints = 128,
            TimeSteps = 500,
            FinalTime = 2.0,
            L2Error = 0.005,
            MaxError = 0.02
        };

        Assert.Equal("Burgers", result.EquationName);
        Assert.Equal(128, result.SpatialPoints);
        Assert.Equal(500, result.TimeSteps);
        Assert.Equal(2.0, result.FinalTime);
        Assert.Equal(0.005, result.L2Error);
        Assert.Equal(0.02, result.MaxError);
    }

    #endregion

    #region TrainingHistory

    [Fact(Timeout = 120000)]
    public async Task TrainingHistory_Construction()
    {
        var history = new TrainingHistory<double>();
        Assert.NotNull(history.Losses);
        Assert.Empty(history.Losses);
    }

    [Fact(Timeout = 120000)]
    public async Task TrainingHistory_AddEpoch_MultipleLosses()
    {
        var history = new TrainingHistory<double>();
        history.AddEpoch(1.0);
        history.AddEpoch(0.5);
        history.AddEpoch(0.25);

        Assert.Equal(3, history.Losses.Count);
        Assert.Equal(1.0, history.Losses[0]);
        Assert.Equal(0.5, history.Losses[1]);
        Assert.Equal(0.25, history.Losses[2]);
    }

    #endregion

    #region DomainDecompositionTrainingHistory

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionTrainingHistory_Construction()
    {
        var history = new DomainDecompositionTrainingHistory<double>(3);

        Assert.Equal(3, history.SubdomainCount);
        Assert.Empty(history.Losses);
        Assert.Empty(history.SubdomainLosses);
        Assert.Empty(history.InterfaceLosses);
        Assert.Empty(history.PhysicsLosses);
    }

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionTrainingHistory_ZeroSubdomains_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DomainDecompositionTrainingHistory<double>(0));
    }

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionTrainingHistory_AddEpoch_TracksAllMetrics()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);
        var subLosses = new List<double> { 0.1, 0.2 };

        history.AddEpoch(0.5, subLosses, 0.05, 0.3);

        Assert.Single(history.Losses);
        Assert.Equal(0.5, history.Losses[0]);
        Assert.Single(history.SubdomainLosses);
        Assert.Equal(2, history.SubdomainLosses[0].Count);
        Assert.Equal(0.1, history.SubdomainLosses[0][0]);
        Assert.Equal(0.2, history.SubdomainLosses[0][1]);
        Assert.Single(history.InterfaceLosses);
        Assert.Equal(0.05, history.InterfaceLosses[0]);
        Assert.Single(history.PhysicsLosses);
        Assert.Equal(0.3, history.PhysicsLosses[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionTrainingHistory_AddEpoch_NullSubdomainLosses_Throws()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);

        Assert.Throws<ArgumentNullException>(() =>
            history.AddEpoch(0.5, null!, 0.05, 0.3));
    }

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionTrainingHistory_AddEpoch_WrongSubdomainCount_Throws()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);
        var wrongLosses = new List<double> { 0.1, 0.2, 0.3 }; // 3 instead of 2

        Assert.Throws<ArgumentException>(() =>
            history.AddEpoch(0.5, wrongLosses, 0.05, 0.3));
    }

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionTrainingHistory_MultipleEpochs()
    {
        var history = new DomainDecompositionTrainingHistory<double>(2);

        history.AddEpoch(1.0, new List<double> { 0.5, 0.5 }, 0.1, 0.4);
        history.AddEpoch(0.5, new List<double> { 0.25, 0.25 }, 0.05, 0.2);

        Assert.Equal(2, history.Losses.Count);
        Assert.Equal(2, history.SubdomainLosses.Count);
        Assert.Equal(2, history.InterfaceLosses.Count);
        Assert.Equal(2, history.PhysicsLosses.Count);
    }

    #endregion

    #region MultiFidelityTrainingHistory

    [Fact(Timeout = 120000)]
    public async Task MultiFidelityTrainingHistory_Construction()
    {
        var history = new MultiFidelityTrainingHistory<double>();

        Assert.Empty(history.Losses);
        Assert.Empty(history.LowFidelityLosses);
        Assert.Empty(history.HighFidelityLosses);
        Assert.Empty(history.CorrelationLosses);
        Assert.Empty(history.PhysicsLosses);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiFidelityTrainingHistory_AddEpoch_TracksAllMetrics()
    {
        var history = new MultiFidelityTrainingHistory<double>();

        history.AddEpoch(1.0, 0.3, 0.2, 0.1, 0.4);

        Assert.Single(history.Losses);
        Assert.Equal(1.0, history.Losses[0]);
        Assert.Single(history.LowFidelityLosses);
        Assert.Equal(0.3, history.LowFidelityLosses[0]);
        Assert.Single(history.HighFidelityLosses);
        Assert.Equal(0.2, history.HighFidelityLosses[0]);
        Assert.Single(history.CorrelationLosses);
        Assert.Equal(0.1, history.CorrelationLosses[0]);
        Assert.Single(history.PhysicsLosses);
        Assert.Equal(0.4, history.PhysicsLosses[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiFidelityTrainingHistory_MultipleEpochs()
    {
        var history = new MultiFidelityTrainingHistory<double>();

        history.AddEpoch(1.0, 0.3, 0.2, 0.1, 0.4);
        history.AddEpoch(0.5, 0.15, 0.1, 0.05, 0.2);
        history.AddEpoch(0.25, 0.08, 0.05, 0.02, 0.1);

        Assert.Equal(3, history.Losses.Count);
        Assert.Equal(3, history.LowFidelityLosses.Count);
        Assert.Equal(3, history.HighFidelityLosses.Count);
        Assert.Equal(3, history.CorrelationLosses.Count);
        Assert.Equal(3, history.PhysicsLosses.Count);

        // Verify losses decrease
        Assert.True(history.Losses[2] < history.Losses[0]);
    }

    #endregion

    #region GpuPINNTrainingOptions

    [Fact(Timeout = 120000)]
    public async Task GpuPINNTrainingOptions_DefaultValues()
    {
        var opts = new GpuPINNTrainingOptions();

        Assert.True(opts.EnableGpu);
        Assert.NotNull(opts.GpuConfig);
        Assert.Equal(1024, opts.BatchSizeGpu);
        Assert.True(opts.ParallelDerivativeComputation);
        Assert.Equal(1000, opts.MinPointsForGpu);
        Assert.True(opts.AsyncTransfers);
        Assert.False(opts.UseMixedPrecision);
        Assert.True(opts.UsePinnedMemory);
        Assert.False(opts.VerboseLogging);
        Assert.Equal(2, opts.NumStreams);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuPINNTrainingOptions_Default_MatchesConstructor()
    {
        var opts = GpuPINNTrainingOptions.Default;

        Assert.True(opts.EnableGpu);
        Assert.Equal(1024, opts.BatchSizeGpu);
        Assert.Equal(2, opts.NumStreams);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuPINNTrainingOptions_HighEnd_LargeBatchSize()
    {
        var opts = GpuPINNTrainingOptions.HighEnd;

        Assert.Equal(4096, opts.BatchSizeGpu);
        Assert.True(opts.ParallelDerivativeComputation);
        Assert.True(opts.AsyncTransfers);
        Assert.True(opts.UseMixedPrecision);
        Assert.True(opts.UsePinnedMemory);
        Assert.Equal(4, opts.NumStreams);
        Assert.Equal(GpuUsageLevel.Aggressive, opts.GpuConfig.UsageLevel);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuPINNTrainingOptions_LowMemory_SmallBatchSize()
    {
        var opts = GpuPINNTrainingOptions.LowMemory;

        Assert.Equal(256, opts.BatchSizeGpu);
        Assert.True(opts.ParallelDerivativeComputation);
        Assert.False(opts.AsyncTransfers);
        Assert.False(opts.UseMixedPrecision);
        Assert.False(opts.UsePinnedMemory);
        Assert.Equal(1, opts.NumStreams);
        Assert.Equal(GpuUsageLevel.Conservative, opts.GpuConfig.UsageLevel);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuPINNTrainingOptions_CpuOnly_DisablesGpu()
    {
        var opts = GpuPINNTrainingOptions.CpuOnly;

        Assert.False(opts.EnableGpu);
        Assert.Equal(GpuDeviceType.CPU, opts.GpuConfig.DeviceType);
    }

    [Fact(Timeout = 120000)]
    public async Task GpuPINNTrainingOptions_CustomValues()
    {
        var opts = new GpuPINNTrainingOptions
        {
            EnableGpu = false,
            BatchSizeGpu = 512,
            ParallelDerivativeComputation = false,
            MinPointsForGpu = 500,
            AsyncTransfers = false,
            UseMixedPrecision = true,
            UsePinnedMemory = false,
            VerboseLogging = true,
            NumStreams = 8
        };

        Assert.False(opts.EnableGpu);
        Assert.Equal(512, opts.BatchSizeGpu);
        Assert.False(opts.ParallelDerivativeComputation);
        Assert.Equal(500, opts.MinPointsForGpu);
        Assert.False(opts.AsyncTransfers);
        Assert.True(opts.UseMixedPrecision);
        Assert.False(opts.UsePinnedMemory);
        Assert.True(opts.VerboseLogging);
        Assert.Equal(8, opts.NumStreams);
    }

    #endregion

    #region Options Classes Construction

    [Fact(Timeout = 120000)]
    public async Task PhysicsInformedNeuralNetworkOptions_Construction()
    {
        var opts = new PhysicsInformedNeuralNetworkOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task DeepOperatorNetworkOptions_Construction()
    {
        var opts = new DeepOperatorNetworkOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task FourierNeuralOperatorOptions_Construction()
    {
        var opts = new FourierNeuralOperatorOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task GraphNeuralOperatorOptions_Construction()
    {
        var opts = new GraphNeuralOperatorOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task DomainDecompositionPINNOptions_Construction()
    {
        var opts = new DomainDecompositionPINNOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiFidelityPINNOptions_Construction()
    {
        var opts = new MultiFidelityPINNOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task DeepRitzMethodOptions_Construction()
    {
        var opts = new DeepRitzMethodOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task VariationalPINNOptions_Construction()
    {
        var opts = new VariationalPINNOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiScalePINNOptions_Construction()
    {
        var opts = new MultiScalePINNOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task HamiltonianNeuralNetworkOptions_Construction()
    {
        var opts = new HamiltonianNeuralNetworkOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task LagrangianNeuralNetworkOptions_Construction()
    {
        var opts = new LagrangianNeuralNetworkOptions();
        Assert.NotNull(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task UniversalDifferentialEquationsOptions_Construction()
    {
        var opts = new UniversalDifferentialEquationsOptions();
        Assert.NotNull(opts);
    }

    #endregion

    #region NavierStokesEquation

    [Fact(Timeout = 120000)]
    public async Task NavierStokesEquation_Construction_DefaultParams()
    {
        var ns = new NavierStokesEquation<double>();

        Assert.Equal("Navier-Stokes (ν=0.01, ρ=1)", ns.Name);
        Assert.Equal(3, ns.InputDimension); // [x, y, t]
        Assert.Equal(3, ns.OutputDimension); // [u, v, p]
    }

    [Fact(Timeout = 120000)]
    public async Task NavierStokesEquation_Construction_CustomParams()
    {
        var ns = new NavierStokesEquation<double>(viscosity: 0.001, density: 1000.0);

        Assert.Contains("0.001", ns.Name);
        Assert.Contains("1000", ns.Name);
        Assert.Equal(3, ns.InputDimension);
        Assert.Equal(3, ns.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task NavierStokesEquation_ComputeResidual_SteadyUniformFlow_ZeroResidual()
    {
        // Uniform flow: u = constant, v = 0, p = constant
        // All derivatives are zero for uniform flow
        var ns = new NavierStokesEquation<double>(viscosity: 0.01, density: 1.0);
        var inputs = new Vector<double>(new double[] { 0.5, 0.5, 0.0 }); // [x, y, t]
        var outputs = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }); // [u, v, p] - uniform velocity in x

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[3, 3], // All zero
            SecondDerivatives = new double[3, 3, 3] // All zero
        };

        double residual = ns.ComputeResidual(inputs, outputs, derivatives);
        Assert.Equal(0.0, residual, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task NavierStokesEquation_ComputeResidual_WrongOutputDim_Throws()
    {
        var ns = new NavierStokesEquation<double>();
        var inputs = new Vector<double>(new double[] { 0.5, 0.5, 0.0 });
        var outputs = new Vector<double>(new double[] { 1.0, 0.0 }); // Wrong - needs 3

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[2, 3],
            SecondDerivatives = new double[2, 3, 3]
        };

        Assert.Throws<ArgumentException>(() =>
            ns.ComputeResidual(inputs, outputs, derivatives));
    }

    [Fact(Timeout = 120000)]
    public async Task NavierStokesEquation_ComputeResidualGradient_ReturnsGradient()
    {
        var ns = new NavierStokesEquation<double>();
        var inputs = new Vector<double>(new double[] { 0.5, 0.5, 0.0 });
        var outputs = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[3, 3],
            SecondDerivatives = new double[3, 3, 3]
        };

        var gradient = ns.ComputeResidualGradient(inputs, outputs, derivatives);

        Assert.NotNull(gradient);
        Assert.Equal(3, gradient.OutputGradients.Length);
    }

    #endregion

    #region BlackScholesEquation

    [Fact(Timeout = 120000)]
    public async Task BlackScholesEquation_Construction_DefaultParams()
    {
        var bs = new BlackScholesEquation<double>();

        Assert.Contains("Black-Scholes", bs.Name);
        Assert.Equal(2, bs.InputDimension); // [S, t]
        Assert.Equal(1, bs.OutputDimension); // [V]
    }

    [Fact(Timeout = 120000)]
    public async Task BlackScholesEquation_Construction_CustomParams()
    {
        var bs = new BlackScholesEquation<double>(volatility: 0.3, riskFreeRate: 0.08);

        Assert.Contains("0.3", bs.Name);
        Assert.Contains("0.08", bs.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task BlackScholesEquation_Construction_ZeroVolatility_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new BlackScholesEquation<double>(volatility: 0.0));
    }

    [Fact(Timeout = 120000)]
    public async Task BlackScholesEquation_ComputeResidual_ConstantOption_KnownResidual()
    {
        // If V = constant and all derivatives are zero,
        // residual = 0 + 0 + 0 - r*V = -r*V
        var r = 0.05;
        var bs = new BlackScholesEquation<double>(volatility: 0.2, riskFreeRate: r);
        var inputs = new Vector<double>(new double[] { 100.0, 0.5 }); // S=100, t=0.5
        var outputs = new Vector<double>(new double[] { 10.0 }); // V=10

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2], // All zero
            SecondDerivatives = new double[1, 2, 2] // All zero
        };

        double residual = bs.ComputeResidual(inputs, outputs, derivatives);

        // Expected: 0 + 0 + 0 - 0.05 * 10 = -0.5
        Assert.Equal(-0.5, residual, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BlackScholesEquation_ComputeResidualGradient_OutputGradient()
    {
        var r = 0.05;
        var bs = new BlackScholesEquation<double>(volatility: 0.2, riskFreeRate: r);
        var inputs = new Vector<double>(new double[] { 100.0, 0.5 });
        var outputs = new Vector<double>(new double[] { 10.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = bs.ComputeResidualGradient(inputs, outputs, derivatives);

        // ∂R/∂V = -r = -0.05
        Assert.Equal(-r, gradient.OutputGradients[0], Tolerance);
        // ∂R/∂(∂V/∂t) = 1
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], Tolerance);
    }

    #endregion

    #region SchrodingerEquation

    [Fact(Timeout = 120000)]
    public async Task SchrodingerEquation_Construction_FreeParticle()
    {
        var se = new SchrodingerEquation<double>();

        Assert.Equal("Schrodinger Equation", se.Name);
        Assert.Equal(2, se.InputDimension); // [x, t]
        Assert.Equal(2, se.OutputDimension); // [psi_r, psi_i]
    }

    [Fact(Timeout = 120000)]
    public async Task SchrodingerEquation_Construction_CustomPotential()
    {
        // Harmonic oscillator V(x) = 0.5 * x^2
        var se = new SchrodingerEquation<double>(x => 0.5 * x * x);

        Assert.Equal("Schrodinger Equation", se.Name);
        Assert.Equal(2, se.InputDimension);
        Assert.Equal(2, se.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task SchrodingerEquation_Construction_NullPotential_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SchrodingerEquation<double>(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task SchrodingerEquation_ComputeResidual_ZeroState_ZeroResidual()
    {
        // Zero wavefunction with zero derivatives should give zero residual
        var se = new SchrodingerEquation<double>();
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 }); // [x, t]
        var outputs = new Vector<double>(new double[] { 0.0, 0.0 }); // [psi_r, psi_i] = 0

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[2, 2], // All zero
            SecondDerivatives = new double[2, 2, 2] // All zero
        };

        double residual = se.ComputeResidual(inputs, outputs, derivatives);
        Assert.Equal(0.0, residual, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task SchrodingerEquation_ComputeResidual_WrongOutputDim_Throws()
    {
        var se = new SchrodingerEquation<double>();
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 1.0 }); // Wrong - needs 2

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        Assert.Throws<ArgumentException>(() =>
            se.ComputeResidual(inputs, outputs, derivatives));
    }

    #endregion

    #region AdvectionDiffusionEquation Extended

    [Fact(Timeout = 120000)]
    public async Task AdvectionDiffusionEquation_1D_DefaultParams()
    {
        var ade = new AdvectionDiffusionEquation<double>();

        Assert.Contains("1D", ade.Name);
        Assert.Equal(2, ade.InputDimension); // [x, t]
        Assert.Equal(1, ade.OutputDimension); // [c]
    }

    [Fact(Timeout = 120000)]
    public async Task AdvectionDiffusionEquation_2D_FourParams()
    {
        var ade = new AdvectionDiffusionEquation<double>(
            diffusionCoeff: 0.1, velocityX: 1.0, velocityY: 0.5, sourceTerm: 0.0);

        Assert.Contains("2D", ade.Name);
        Assert.Equal(3, ade.InputDimension); // [x, y, t]
        Assert.Equal(1, ade.OutputDimension); // [c]
    }

    [Fact(Timeout = 120000)]
    public async Task AdvectionDiffusionEquation_NegativeDiffusion_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new AdvectionDiffusionEquation<double>(diffusionCoeff: -0.1));
    }

    [Fact(Timeout = 120000)]
    public async Task AdvectionDiffusionEquation_1D_UniformConcentration_ZeroResidual()
    {
        // Constant concentration with zero source: all derivatives are zero, residual = 0 - source
        var ade = new AdvectionDiffusionEquation<double>(diffusionCoeff: 0.1, velocityX: 1.0, sourceTerm: 0.0);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 1.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        double residual = ade.ComputeResidual(inputs, outputs, derivatives);
        Assert.Equal(0.0, residual, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task AdvectionDiffusionEquation_ComputeResidualGradient_1D()
    {
        var ade = new AdvectionDiffusionEquation<double>(diffusionCoeff: 0.1, velocityX: 1.0, sourceTerm: 0.0);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 1.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = ade.ComputeResidualGradient(inputs, outputs, derivatives);

        // ∂R/∂(∂c/∂t) = 1
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], Tolerance);
        // ∂R/∂(∂c/∂x) = vx = 1.0
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 0], Tolerance);
        // ∂R/∂(∂²c/∂x²) = -D = -0.1
        Assert.Equal(-0.1, gradient.SecondDerivatives[0, 0, 0], Tolerance);
    }

    #endregion

    #region LinearElasticityEquation

    [Fact(Timeout = 120000)]
    public async Task LinearElasticityEquation_Construction_DefaultParams()
    {
        var le = new LinearElasticityEquation<double>();

        Assert.Contains("Linear Elasticity", le.Name);
        Assert.Equal(2, le.InputDimension); // [x, y]
        Assert.Equal(2, le.OutputDimension); // [u, v]
    }

    [Fact(Timeout = 120000)]
    public async Task LinearElasticityEquation_Construction_CustomParams()
    {
        var le = new LinearElasticityEquation<double>(lambda: 1.0, mu: 0.5);

        Assert.Contains("Linear Elasticity", le.Name);
        Assert.Equal(2, le.InputDimension);
        Assert.Equal(2, le.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task LinearElasticityEquation_ComputeResidual_ZeroDisplacement_ZeroResidual()
    {
        // Zero displacement with zero body forces: all derivatives are zero, residual = 0
        var le = new LinearElasticityEquation<double>(lambda: 1.0, mu: 0.5);
        var inputs = new Vector<double>(new double[] { 0.5, 0.5 });
        var outputs = new Vector<double>(new double[] { 0.0, 0.0 }); // No displacement

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[2, 2],
            SecondDerivatives = new double[2, 2, 2]
        };

        double residual = le.ComputeResidual(inputs, outputs, derivatives);
        Assert.Equal(0.0, residual, Tolerance);
    }

    #endregion

    #region MaxwellEquations

    [Fact(Timeout = 120000)]
    public async Task MaxwellEquations_Construction_DefaultParams()
    {
        var maxwell = new MaxwellEquations<double>(permittivity: 1.0, permeability: 1.0);

        Assert.Contains("Maxwell", maxwell.Name);
        Assert.Equal(3, maxwell.InputDimension); // [x, y, t]
        Assert.Equal(3, maxwell.OutputDimension); // [Ex, Ey, Bz]
    }

    [Fact(Timeout = 120000)]
    public async Task MaxwellEquations_ComputeResidual_ZeroFields_ZeroResidual()
    {
        // Zero fields with zero derivatives: residual = 0
        var maxwell = new MaxwellEquations<double>(permittivity: 1.0, permeability: 1.0);
        var inputs = new Vector<double>(new double[] { 0.5, 0.5, 0.0 });
        var outputs = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }); // No EM fields

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[3, 3],
            SecondDerivatives = new double[3, 3, 3]
        };

        double residual = maxwell.ComputeResidual(inputs, outputs, derivatives);
        Assert.Equal(0.0, residual, Tolerance);
    }

    #endregion

    #region AllenCahnEquation Extended

    [Fact(Timeout = 120000)]
    public async Task AllenCahnEquation_Construction_CustomEpsilon()
    {
        var ac = new AllenCahnEquation<double>(epsilon: 0.1);

        Assert.Equal("Allen-Cahn Equation", ac.Name);
        Assert.Equal(2, ac.InputDimension);
        Assert.Equal(1, ac.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task AllenCahnEquation_Construction_ZeroEpsilon_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new AllenCahnEquation<double>(epsilon: 0.0));
    }

    [Fact(Timeout = 120000)]
    public async Task AllenCahnEquation_ComputeResidualGradient_ReturnsGradient()
    {
        var ac = new AllenCahnEquation<double>(epsilon: 0.01);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 0.5 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = ac.ComputeResidualGradient(inputs, outputs, derivatives);

        Assert.NotNull(gradient);
        Assert.Equal(1, gradient.OutputGradients.Length);
        // ∂R/∂u = 3u^2 - 1 = 3*(0.25) - 1 = -0.25
        Assert.Equal(-0.25, gradient.OutputGradients[0], Tolerance);
    }

    #endregion

    #region BurgersEquation Extended

    [Fact(Timeout = 120000)]
    public async Task BurgersEquation_Construction_ZeroViscosity()
    {
        // Zero viscosity = inviscid Burgers
        var burgers = new BurgersEquation<double>(viscosity: 0.0);

        Assert.Contains("Burgers", burgers.Name);
        Assert.Equal(2, burgers.InputDimension);
        Assert.Equal(1, burgers.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task BurgersEquation_ComputeResidualGradient_ReturnsGradient()
    {
        var burgers = new BurgersEquation<double>(viscosity: 0.01);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 1.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2] { { 0.5, 0.0 } }, // dudx = 0.5, dudt = 0
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = burgers.ComputeResidualGradient(inputs, outputs, derivatives);

        Assert.NotNull(gradient);
        // ∂R/∂u = dudx = 0.5
        Assert.Equal(0.5, gradient.OutputGradients[0], Tolerance);
        // ∂R/∂(∂u/∂x) = u = 1.0
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 0], Tolerance);
        // ∂R/∂(∂u/∂t) = 1
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], Tolerance);
    }

    #endregion

    #region WaveEquation Extended

    [Fact(Timeout = 120000)]
    public async Task WaveEquation_Construction_2D()
    {
        var wave = new WaveEquation<double>(waveSpeed: 2.0, spatialDimension: 2);

        Assert.Contains("2D", wave.Name);
        Assert.Equal(3, wave.InputDimension); // [x, y, t]
        Assert.Equal(1, wave.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task WaveEquation_Construction_3D()
    {
        var wave = new WaveEquation<double>(waveSpeed: 1.0, spatialDimension: 3);

        Assert.Contains("3D", wave.Name);
        Assert.Equal(4, wave.InputDimension); // [x, y, z, t]
        Assert.Equal(1, wave.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task WaveEquation_Construction_InvalidDimension_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new WaveEquation<double>(waveSpeed: 1.0, spatialDimension: 0));

        Assert.Throws<ArgumentException>(() =>
            new WaveEquation<double>(waveSpeed: 1.0, spatialDimension: 4));
    }

    [Fact(Timeout = 120000)]
    public async Task WaveEquation_ComputeResidualGradient_Returns()
    {
        var wave = new WaveEquation<double>(waveSpeed: 2.0);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 1.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = wave.ComputeResidualGradient(inputs, outputs, derivatives);

        Assert.NotNull(gradient);
        // ∂R/∂(∂²u/∂t²) = 1
        Assert.Equal(1.0, gradient.SecondDerivatives[0, 1, 1], Tolerance);
        // ∂R/∂(∂²u/∂x²) = -c² = -4
        Assert.Equal(-4.0, gradient.SecondDerivatives[0, 0, 0], Tolerance);
    }

    #endregion

    #region HeatEquation Extended

    [Fact(Timeout = 120000)]
    public async Task HeatEquation_ComputeResidualGradient_Correctness()
    {
        var heat = new HeatEquation<double>(thermalDiffusivity: 0.5);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 1.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = heat.ComputeResidualGradient(inputs, outputs, derivatives);

        // ∂R/∂(∂u/∂t) = 1
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], Tolerance);
        // ∂R/∂(∂²u/∂x²) = -alpha = -0.5
        Assert.Equal(-0.5, gradient.SecondDerivatives[0, 0, 0], Tolerance);
    }

    #endregion

    #region PoissonEquation Extended

    [Fact(Timeout = 120000)]
    public async Task PoissonEquation_1D()
    {
        var poisson = new PoissonEquation<double>(spatialDimension: 1);

        Assert.Equal(1, poisson.InputDimension);
        Assert.Equal(1, poisson.OutputDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonEquation_3D()
    {
        var poisson = new PoissonEquation<double>(spatialDimension: 3);

        Assert.Equal(3, poisson.InputDimension);
        Assert.Contains("3D", poisson.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonEquation_InvalidDimension_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new PoissonEquation<double>(spatialDimension: 0));

        Assert.Throws<ArgumentException>(() =>
            new PoissonEquation<double>(spatialDimension: 4));
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonEquation_LaplaceCase_NoSource()
    {
        // No source function = Laplace equation
        var laplace = new PoissonEquation<double>(sourceFunction: null, spatialDimension: 2);

        Assert.Contains("Laplace", laplace.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonEquation_ComputeResidualGradient_Returns()
    {
        var poisson = new PoissonEquation<double>(spatialDimension: 2);
        var inputs = new Vector<double>(new double[] { 0.5, 0.5 });
        var outputs = new Vector<double>(new double[] { 1.0 });

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };

        var gradient = poisson.ComputeResidualGradient(inputs, outputs, derivatives);

        Assert.NotNull(gradient);
        // ∂R/∂(∂²u/∂x²) = 1, ∂R/∂(∂²u/∂y²) = 1
        Assert.Equal(1.0, gradient.SecondDerivatives[0, 0, 0], Tolerance);
        Assert.Equal(1.0, gradient.SecondDerivatives[0, 1, 1], Tolerance);
    }

    #endregion

    #region KortewegDeVriesEquation

    [Fact(Timeout = 120000)]
    public async Task KortewegDeVriesEquation_Construction_DefaultParams()
    {
        var kdv = new KortewegDeVriesEquation<double>();

        Assert.Contains("Korteweg-de Vries", kdv.Name);
        Assert.Equal(2, kdv.InputDimension); // [x, t]
        Assert.Equal(1, kdv.OutputDimension); // [u]
    }

    [Fact(Timeout = 120000)]
    public async Task KortewegDeVriesEquation_Construction_CustomParams()
    {
        var kdv = new KortewegDeVriesEquation<double>(alpha: 6.0, beta: 1.0);

        Assert.Contains("Korteweg-de Vries", kdv.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task KortewegDeVriesEquation_ConstantSolution_ZeroResidual()
    {
        var kdv = new KortewegDeVriesEquation<double>(alpha: 6.0, beta: 1.0);
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new double[] { 0.0 }); // Constant = 0

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2], // All zero
            SecondDerivatives = new double[1, 2, 2],
            ThirdDerivatives = new double[1, 2, 2, 2] // All zero
        };

        double residual = kdv.ComputeResidual(inputs, outputs, derivatives);
        Assert.Equal(0.0, residual, Tolerance);
    }

    #endregion

    #region PDEDerivatives Extended

    [Fact(Timeout = 120000)]
    public async Task PDEDerivatives_HigherDerivatives_SetAndGet()
    {
        var derivatives = new PDEDerivatives<double>();
        var higher = new double[1, 2, 2, 2];
        higher[0, 0, 0, 0] = 1.5;
        derivatives.HigherDerivatives = higher;

        Assert.NotNull(derivatives.HigherDerivatives);
        Assert.Equal(1.5, derivatives.HigherDerivatives[0, 0, 0, 0]);
    }

    [Fact(Timeout = 120000)]
    public async Task PDEDerivatives_AllPropertiesNullByDefault()
    {
        var derivatives = new PDEDerivatives<double>();

        Assert.Null(derivatives.FirstDerivatives);
        Assert.Null(derivatives.SecondDerivatives);
        Assert.Null(derivatives.ThirdDerivatives);
        Assert.Null(derivatives.HigherDerivatives);
    }

    #endregion

    #region PDEResidualGradient Extended

    [Fact(Timeout = 120000)]
    public async Task PDEResidualGradient_CorrectDimensions()
    {
        var gradient = new PDEResidualGradient<double>(outputDimension: 3, inputDimension: 4);

        Assert.Equal(3, gradient.OutputGradients.Length);
        Assert.Equal(3, gradient.FirstDerivatives.GetLength(0));
        Assert.Equal(4, gradient.FirstDerivatives.GetLength(1));
        Assert.Equal(3, gradient.SecondDerivatives.GetLength(0));
        Assert.Equal(4, gradient.SecondDerivatives.GetLength(1));
        Assert.Equal(4, gradient.SecondDerivatives.GetLength(2));
        Assert.Equal(3, gradient.ThirdDerivatives.GetLength(0));
        Assert.Equal(4, gradient.ThirdDerivatives.GetLength(1));
        Assert.Equal(4, gradient.ThirdDerivatives.GetLength(2));
        Assert.Equal(4, gradient.ThirdDerivatives.GetLength(3));
    }

    #endregion

    #region PhysicsInformedLoss Extended

    [Fact(Timeout = 120000)]
    public async Task PhysicsInformedLoss_Name()
    {
        var loss = new PhysicsInformedLoss<double>();
        Assert.Equal("Physics-Informed Loss", loss.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task PhysicsInformedLoss_ComputeDerivative_ReturnsCorrectGradient()
    {
        var loss = new PhysicsInformedLoss<double>();
        var predictions = new double[] { 1.0, 2.0, 3.0 };
        var targets = new double[] { 1.5, 1.5, 2.5 };

        var derivative = loss.ComputeDerivative(predictions, targets);

        Assert.Equal(3, derivative.Length);
        // MSE derivative = 2/N * (pred - target)
        // For first element: 2/3 * (1.0 - 1.5) = 2/3 * (-0.5) = -1/3
        Assert.Equal(-1.0 / 3.0, derivative[0], 1e-6);
    }

    #endregion

    #region Cross-Module Integration

    [Fact(Timeout = 120000)]
    public async Task HeatEquation_WithPhysicsInformedLoss_ComputesLoss()
    {
        var heat = new HeatEquation<double>(thermalDiffusivity: 1.0);
        var loss = new PhysicsInformedLoss<double>(pdeSpecification: heat);

        var predictions = new Vector<double>(new double[] { 0.5 });
        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2] { { 0.0, 0.1 } }, // dudx=0, dudt=0.1
            SecondDerivatives = new double[1, 2, 2]
        };
        // d2udx2 = 0, so residual = dudt - alpha * d2udx2 = 0.1 - 0 = 0.1
        // PDE loss = residual^2 = 0.01
        var inputs = new Vector<double>(new double[] { 0.5, 0.1 });

        double totalLoss = loss.ComputePhysicsLoss(predictions, null, derivatives, inputs);
        Assert.Equal(0.01, totalLoss, 1e-6);
    }

    [Fact(Timeout = 120000)]
    public async Task TrainingHistories_InheritFromBase()
    {
        // Both DomainDecomposition and MultiFidelity inherit from TrainingHistory
        var ddHistory = new DomainDecompositionTrainingHistory<double>(2);
        var mfHistory = new MultiFidelityTrainingHistory<double>();

        // Both can use base AddEpoch
        ddHistory.AddEpoch(1.0);
        mfHistory.AddEpoch(1.0);

        Assert.Single(ddHistory.Losses);
        Assert.Single(mfHistory.Losses);
    }

    #endregion
}
