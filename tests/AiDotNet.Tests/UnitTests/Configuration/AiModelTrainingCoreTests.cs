using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training.Memory;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>
/// Unit tests for <see cref="AiModelTrainingCore{T,TInput,TOutput}"/> — the slice-2 extraction
/// of the audit-2026-05 phase-2a <c>AiModelBuilder</c> DI refactor. Tests exercise the component
/// in isolation, without instantiating the rest of <c>AiModelBuilder</c>, to prove the training-
/// core concern can be used by alternative composition roots.
/// </summary>
public class AiModelTrainingCoreTests
{
    [Fact(Timeout = 30000)]
    public async Task InitialState_AllSlotsAreNull()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();

        Assert.Null(core.Model);
        Assert.Null(core.Optimizer);
        Assert.Null(core.Regularization);
        Assert.Null(core.FitnessCalculator);
        Assert.Null(core.FitDetector);
        Assert.Null(core.TrainingPipelineConfiguration);
        Assert.Null(core.CheckpointManager);
        Assert.Null(core.MemoryConfig);
        Assert.Null(core.TrainingMonitor);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureModel_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var model = Mock.Of<IFullModel<double, Matrix<double>, Vector<double>>>();

        core.ConfigureModel(model);

        Assert.Same(model, core.Model);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureOptimizer_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var optimizer = Mock.Of<IOptimizer<double, Matrix<double>, Vector<double>>>();

        core.ConfigureOptimizer(optimizer);

        Assert.Same(optimizer, core.Optimizer);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureRegularization_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var regularization = Mock.Of<IRegularization<double, Matrix<double>, Vector<double>>>();

        core.ConfigureRegularization(regularization);

        Assert.Same(regularization, core.Regularization);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureFitnessCalculator_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var calculator = Mock.Of<IFitnessCalculator<double, Matrix<double>, Vector<double>>>();

        core.ConfigureFitnessCalculator(calculator);

        Assert.Same(calculator, core.FitnessCalculator);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureFitDetector_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var detector = Mock.Of<IFitDetector<double, Matrix<double>, Vector<double>>>();

        core.ConfigureFitDetector(detector);

        Assert.Same(detector, core.FitDetector);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureTrainingPipeline_NullIsValid()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();

        core.ConfigureTrainingPipeline(null);

        Assert.Null(core.TrainingPipelineConfiguration);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureCheckpointManager_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var manager = Mock.Of<ICheckpointManager<double, Matrix<double>, Vector<double>>>();

        core.ConfigureCheckpointManager(manager);

        Assert.Same(manager, core.CheckpointManager);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureMemoryManagement_PassesThrough()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var config = new TrainingMemoryConfig();

        core.ConfigureMemoryManagement(config);

        Assert.Same(config, core.MemoryConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureMemoryManagement_NullIsValid()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();

        core.ConfigureMemoryManagement(null);

        Assert.Null(core.MemoryConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureTrainingMonitor_Stores()
    {
        await Task.Yield();
        var core = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var monitor = Mock.Of<ITrainingMonitor<double>>();

        core.ConfigureTrainingMonitor(monitor);

        Assert.Same(monitor, core.TrainingMonitor);
    }

    [Fact(Timeout = 30000)]
    public async Task Interface_IsImplementedByDefaultComponent()
    {
        await Task.Yield();
        IAiModelTrainingCore<double, Matrix<double>, Vector<double>> component
            = new AiModelTrainingCore<double, Matrix<double>, Vector<double>>();
        var model = Mock.Of<IFullModel<double, Matrix<double>, Vector<double>>>();

        component.ConfigureModel(model);

        Assert.Same(model, component.Model);
    }
}
