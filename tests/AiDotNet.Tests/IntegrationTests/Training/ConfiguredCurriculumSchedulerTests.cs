using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.CurriculumLearning.Schedulers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureCurriculumScheduler reaches the options curriculum learning actually reads.
/// </summary>
/// <remarks>
/// The scheduler was parked in a private field nothing read, while the build passes
/// CurriculumLearningOptions.CustomScheduler to the CurriculumLearner — so a configured scheduler
/// never arrived and the default was used silently.
/// </remarks>
public class ConfiguredCurriculumSchedulerTests
{
    [Fact(Timeout = 60000)]
    public async Task SchedulerConfiguredAfterOptions_ReachesTheOptions()
    {
        var options = new CurriculumLearningOptions<double, Matrix<double>, Vector<double>>();
        var scheduler = new LinearScheduler<double>(totalEpochs: 10);

        new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureCurriculumLearning(options)
            .ConfigureCurriculumScheduler(scheduler);

        Assert.Same(scheduler, options.CustomScheduler);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task SchedulerConfiguredBeforeOptions_StillReachesThem()
    {
        // Order must not matter: a builder is configured fluently and callers should not have to
        // know that one Configure* silently depends on another having run first.
        var options = new CurriculumLearningOptions<double, Matrix<double>, Vector<double>>();
        var scheduler = new LinearScheduler<double>(totalEpochs: 10);

        new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureCurriculumScheduler(scheduler)
            .ConfigureCurriculumLearning(options);

        Assert.Same(scheduler, options.CustomScheduler);
        await Task.CompletedTask;
    }
}
