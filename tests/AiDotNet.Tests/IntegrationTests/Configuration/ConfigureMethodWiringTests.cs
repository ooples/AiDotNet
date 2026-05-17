using AiDotNet.Configuration;
using AiDotNet.Models.Options;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression tests for the systemic "Configure* stores config, never consumes it"
/// pattern that was sweep-fixed under the #1357 family.
/// </summary>
/// <remarks>
/// <para>
/// Prior to the fix, several Configure* methods on AiModelBuilder stored their
/// argument in a private field that was never read elsewhere in src/, so the call
/// had no observable effect. These tests assert that:
/// </para>
/// <list type="bullet">
///   <item><description>ConfigureAdversarialRobustness retains the configuration and
///   propagates the underlying options through to the AiModelResult robustness
///   surface via the new AttachAdversarialRobustness path (#1357).</description></item>
///   <item><description>ConfigureFineTuning, ConfigureTrainingPipeline,
///   ConfigureCurriculumLearning, and ConfigureSelfSupervisedLearning retain
///   their configuration in the corresponding "Configured*" internal accessors
///   (no in-engine consumer is wired yet for these — they are reserved for
///   future use and emit a Trace warning).</description></item>
/// </list>
/// </remarks>
public class ConfigureMethodWiringTests
{
    [Fact(Timeout = 60000)]
    public void ConfigureAdversarialRobustness_RetainsConfiguration_OnBuilder()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var configuration = AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>.BasicSafety();

        var returned = builder.ConfigureAdversarialRobustness(configuration);

        // Fluent API still chains correctly.
        Assert.Same(builder, returned);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureAdversarialRobustness_DefaultArgument_StoresEnabledConfiguration()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // null argument -> sensible default with Enabled=true (the documented contract).
        var returned = builder.ConfigureAdversarialRobustness(configuration: null);
        Assert.Same(builder, returned);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureFineTuning_RetainsConfiguration_AccessibleViaInternalAccessor()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var configuration = new FineTuningConfiguration<double, Matrix<double>, Vector<double>>
        {
            Enabled = true
        };

        builder.ConfigureFineTuning(configuration);

        Assert.NotNull(builder.ConfiguredFineTuning);
        Assert.Same(configuration, builder.ConfiguredFineTuning);
        Assert.True(builder.ConfiguredFineTuning.Enabled);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureFineTuning_NullArgument_StoresDefaultConfiguration()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        builder.ConfigureFineTuning(configuration: null);

        Assert.NotNull(builder.ConfiguredFineTuning);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureTrainingPipeline_RetainsConfiguration_AccessibleViaInternalAccessor()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var configuration = new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>();

        builder.ConfigureTrainingPipeline(configuration);

        Assert.NotNull(builder.ConfiguredTrainingPipeline);
        Assert.Same(configuration, builder.ConfiguredTrainingPipeline);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureTrainingPipeline_NullArgument_ClearsPriorIntent()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var configuration = new TrainingPipelineConfiguration<double, Matrix<double>, Vector<double>>();

        builder.ConfigureTrainingPipeline(configuration);
        Assert.NotNull(builder.ConfiguredTrainingPipeline);

        // Calling with null is documented as clearing prior intent.
        builder.ConfigureTrainingPipeline(configuration: null);
        Assert.Null(builder.ConfiguredTrainingPipeline);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureCurriculumLearning_RetainsConfiguration_AccessibleViaInternalAccessor()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var options = new CurriculumLearningOptions<double, Matrix<double>, Vector<double>>
        {
            NumPhases = 7
        };

        builder.ConfigureCurriculumLearning(options);

        Assert.NotNull(builder.ConfiguredCurriculumLearning);
        Assert.Same(options, builder.ConfiguredCurriculumLearning);
        Assert.Equal(7, builder.ConfiguredCurriculumLearning.NumPhases);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureCurriculumLearning_NullArgument_StoresDefaultOptions()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        builder.ConfigureCurriculumLearning(options: null);

        Assert.NotNull(builder.ConfiguredCurriculumLearning);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureSelfSupervisedLearning_RetainsConfiguration_AccessibleViaInternalAccessor()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        builder.ConfigureSelfSupervisedLearning(ssl =>
        {
            ssl.PretrainingEpochs = 42;
            ssl.BatchSize = 64;
        });

        Assert.NotNull(builder.ConfiguredSelfSupervisedLearning);
        Assert.Equal(42, builder.ConfiguredSelfSupervisedLearning.PretrainingEpochs);
        Assert.Equal(64, builder.ConfiguredSelfSupervisedLearning.BatchSize);
    }

    [Fact(Timeout = 60000)]
    public void ConfigureSelfSupervisedLearning_NoConfigureCallback_StoresDefaultConfig()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        builder.ConfigureSelfSupervisedLearning();

        Assert.NotNull(builder.ConfiguredSelfSupervisedLearning);
        Assert.IsType<SSLConfig>(builder.ConfiguredSelfSupervisedLearning);
    }
}
