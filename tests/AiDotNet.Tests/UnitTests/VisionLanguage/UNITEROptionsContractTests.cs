using AiDotNet.VisionLanguage.Foundational;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Guards the official UNITER optimization recipe and its customizable copy surface.
/// </summary>
public sealed class UNITEROptionsContractTests
{
    [Fact]
    public async Task Defaults_MatchOfficialAllDataBaseOptimizationRecipe()
    {
        await Task.Yield();
        var options = new UNITEROptions();

        Assert.Equal(5e-5, options.LearningRate, precision: 12);
        Assert.Equal(0.01, options.WeightDecay, precision: 12);
        Assert.Equal(10_000, options.WarmupSteps);
        Assert.Equal(200_000, options.TotalTrainingSteps);
        Assert.Null(options.WarmupInitialLearningRate);
        Assert.Equal(0.0, options.EndLearningRate, precision: 12);
        Assert.Equal(5.0, options.MaxGradientNorm, precision: 12);
    }

    [Fact]
    public async Task CopyConstructor_PreservesOptimizationCustomization()
    {
        await Task.Yield();
        var source = new UNITEROptions
        {
            LearningRate = 2e-4,
            WeightDecay = 0.02,
            WarmupSteps = 25,
            TotalTrainingSteps = 500,
            WarmupInitialLearningRate = 1e-6,
            EndLearningRate = 2e-7,
            MaxGradientNorm = 3.5
        };

        var copy = new UNITEROptions(source);

        Assert.Equal(source.LearningRate, copy.LearningRate);
        Assert.Equal(source.WeightDecay, copy.WeightDecay);
        Assert.Equal(source.WarmupSteps, copy.WarmupSteps);
        Assert.Equal(source.TotalTrainingSteps, copy.TotalTrainingSteps);
        Assert.Equal(source.WarmupInitialLearningRate, copy.WarmupInitialLearningRate);
        Assert.Equal(source.EndLearningRate, copy.EndLearningRate);
        Assert.Equal(source.MaxGradientNorm, copy.MaxGradientNorm);
    }

    [Fact]
    public async Task Validate_RejectsEveryInvalidOptimizationBoundary()
    {
        await Task.Yield();
        var cases = new (Action<UNITEROptions> Mutate, string ParameterName)[]
        {
            (options => options.LearningRate = 0.0, nameof(UNITEROptions.LearningRate)),
            (options => options.LearningRate = double.NaN, nameof(UNITEROptions.LearningRate)),
            (options => options.LearningRate = double.PositiveInfinity, nameof(UNITEROptions.LearningRate)),
            (options => options.WeightDecay = -0.01, nameof(UNITEROptions.WeightDecay)),
            (options => options.WeightDecay = double.NegativeInfinity, nameof(UNITEROptions.WeightDecay)),
            (options => options.WarmupSteps = -1, nameof(UNITEROptions.WarmupSteps)),
            (options => options.TotalTrainingSteps = 0, nameof(UNITEROptions.TotalTrainingSteps)),
            (options => options.WarmupSteps = options.TotalTrainingSteps + 1, nameof(UNITEROptions.WarmupSteps)),
            (options => options.WarmupInitialLearningRate = double.NaN,
                nameof(UNITEROptions.WarmupInitialLearningRate)),
            (options => options.WarmupInitialLearningRate = double.PositiveInfinity,
                nameof(UNITEROptions.WarmupInitialLearningRate)),
            (options => options.EndLearningRate = -1e-6, nameof(UNITEROptions.EndLearningRate)),
            (options => options.EndLearningRate = double.NegativeInfinity, nameof(UNITEROptions.EndLearningRate)),
            (options => options.MaxGradientNorm = 0.0, nameof(UNITEROptions.MaxGradientNorm)),
            (options => options.MaxGradientNorm = double.PositiveInfinity,
                nameof(UNITEROptions.MaxGradientNorm))
        };

        foreach (var (mutate, parameterName) in cases)
        {
            var options = new UNITEROptions();
            mutate(options);
            var error = Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
            Assert.Equal(parameterName, error.ParamName);
        }
    }
}
