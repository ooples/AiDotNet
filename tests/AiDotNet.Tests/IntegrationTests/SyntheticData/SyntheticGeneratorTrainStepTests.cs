using System.Reflection;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// <c>Train(input, target)</c> on a neural synthetic tabular generator takes one real step of the generator's own
/// objective (the same step its Fit takes per batch) on the target's rows. Before, most generators either threw or ran
/// a supervised step of their generator network against the target, which is not their objective, and three
/// (CTAB-GAN+, REaLTabFormer, TabLLM-Gen) trained nothing at all, in Train or in Fit.
/// </summary>
public class SyntheticGeneratorTrainStepTests
{
    public static IEnumerable<object[]> Generators()
        => typeof(NeuralSyntheticTabularGeneratorBase<>).Assembly.GetTypes()
            .Where(t => t is { IsClass: true, IsAbstract: false, IsGenericTypeDefinition: true }
                && IsGenerator(t)
                && t.GetConstructor(Type.EmptyTypes) is not null)
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .Select(t => new object[] { t.Name.Substring(0, t.Name.IndexOf('`')) });

    private static bool IsGenerator(Type definition)
    {
        for (var t = definition.BaseType; t is not null; t = t.BaseType)
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(NeuralSyntheticTabularGeneratorBase<>)) return true;
        return false;
    }

    /// <summary>
    /// The first Train call prepares the generator from the rows (the layers are rebuilt for the data's width); the
    /// second must move the generator's parameters and leave them finite.
    /// </summary>
    [Theory(Timeout = 600000)]
    [MemberData(nameof(Generators))]
    public async Task Train_TakesARealStepOfTheGeneratorsObjective(string generator)
    {
        await Task.Yield();
        var definition = typeof(NeuralSyntheticTabularGeneratorBase<>).Assembly.GetTypes()
            .Single(t => t.IsGenericTypeDefinition && t.Name == generator + "`1");
        using var model = (NeuralSyntheticTabularGeneratorBase<double>)Activator.CreateInstance(definition.MakeGenericType(typeof(double)))!;

        var rows = new Tensor<double>([32, 3]);
        var rng = new Random(7);
        for (int r = 0; r < 32; r++)
        {
            rows[r, 0] = 50 + 10 * (rng.NextDouble() - 0.5);
            rows[r, 1] = rng.NextDouble();
            rows[r, 2] = -3 + 6 * rng.NextDouble();
        }

        model.Train(rows, rows);
        var before = model.GetParameters().ToArray();
        Assert.NotEmpty(before);

        model.Train(rows, rows);
        var after = model.GetParameters().ToArray();

        Assert.Equal(before.Length, after.Length);
        Assert.All(after, v => Assert.True(double.IsFinite(v), $"{generator}: a parameter is {v} after a Train step."));
        double moved = 0;
        for (int i = 0; i < before.Length; i++) moved = Math.Max(moved, Math.Abs(after[i] - before[i]));
        Assert.True(moved > 0, $"{generator}: Train left every one of its {before.Length} parameters unchanged.");
    }
}
