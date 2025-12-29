using System;
using AiDotNet.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearnerOptionsBaseIntegrationTests
{
    [Fact]
    public void Defaults_AreValidAndStable()
    {
        var options = new MetaLearnerOptionsBase<double>();

        Assert.True(options.IsValid());
        Assert.Equal(0.01, options.InnerLearningRate, precision: 6);
        Assert.Equal(0.001, options.OuterLearningRate, precision: 6);
        Assert.Equal(5, options.AdaptationSteps);
        Assert.Equal(4, options.MetaBatchSize);
        Assert.Equal(1000, options.NumMetaIterations);
        Assert.False(options.UseFirstOrder);
        Assert.Equal(10.0, options.GradientClipThreshold);
        Assert.Null(options.RandomSeed);
        Assert.Equal(100, options.EvaluationTasks);
        Assert.Equal(100, options.EvaluationFrequency);
        Assert.True(options.EnableCheckpointing);
        Assert.Equal(500, options.CheckpointFrequency);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.5)]
    public void IsValid_InvalidInnerLearningRate_ReturnsFalse(double value)
    {
        var options = new MetaLearnerOptionsBase<double> { InnerLearningRate = value };
        Assert.False(options.IsValid());
    }

    [Fact]
    public void Clone_CopiesValues()
    {
        var options = new MetaLearnerOptionsBase<double>
        {
            InnerLearningRate = 0.02,
            OuterLearningRate = 0.004,
            AdaptationSteps = 3,
            MetaBatchSize = 2,
            NumMetaIterations = 10,
            UseFirstOrder = true,
            GradientClipThreshold = null,
            RandomSeed = 42,
            EvaluationTasks = 12,
            EvaluationFrequency = 3,
            EnableCheckpointing = false,
            CheckpointFrequency = 0
        };

        var cloned = options.Clone() as MetaLearnerOptionsBase<double>;

        Assert.NotNull(cloned);
        Assert.NotSame(options, cloned);
        Assert.Equal(options.InnerLearningRate, cloned!.InnerLearningRate, precision: 6);
        Assert.Equal(options.OuterLearningRate, cloned.OuterLearningRate, precision: 6);
        Assert.Equal(options.AdaptationSteps, cloned.AdaptationSteps);
        Assert.Equal(options.MetaBatchSize, cloned.MetaBatchSize);
        Assert.Equal(options.NumMetaIterations, cloned.NumMetaIterations);
        Assert.Equal(options.UseFirstOrder, cloned.UseFirstOrder);
        Assert.Equal(options.GradientClipThreshold, cloned.GradientClipThreshold);
        Assert.Equal(options.RandomSeed, cloned.RandomSeed);
        Assert.Equal(options.EvaluationTasks, cloned.EvaluationTasks);
        Assert.Equal(options.EvaluationFrequency, cloned.EvaluationFrequency);
        Assert.Equal(options.EnableCheckpointing, cloned.EnableCheckpointing);
        Assert.Equal(options.CheckpointFrequency, cloned.CheckpointFrequency);
    }

    [Fact]
    public void Builder_BuildsConfiguredOptions()
    {
        var options = MetaLearnerOptionsBase<double>.CreateBuilder()
            .WithInnerLearningRate(0.02)
            .WithOuterLearningRate(0.003)
            .WithAdaptationSteps(2)
            .WithMetaBatchSize(1)
            .WithNumMetaIterations(5)
            .WithFirstOrder()
            .WithGradientClipping(null)
            .WithRandomSeed(7)
            .WithEvaluation(8, 2)
            .WithCheckpointing(false, 0)
            .Build();

        Assert.Equal(0.02, options.InnerLearningRate, precision: 6);
        Assert.Equal(0.003, options.OuterLearningRate, precision: 6);
        Assert.Equal(2, options.AdaptationSteps);
        Assert.Equal(1, options.MetaBatchSize);
        Assert.Equal(5, options.NumMetaIterations);
        Assert.True(options.UseFirstOrder);
        Assert.Null(options.GradientClipThreshold);
        Assert.Equal(7, options.RandomSeed);
        Assert.Equal(8, options.EvaluationTasks);
        Assert.Equal(2, options.EvaluationFrequency);
        Assert.False(options.EnableCheckpointing);
        Assert.Equal(0, options.CheckpointFrequency);
    }

    [Fact]
    public void Builder_InvalidConfiguration_Throws()
    {
        var builder = MetaLearnerOptionsBase<double>.CreateBuilder()
            .WithAdaptationSteps(0);

        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }
}
