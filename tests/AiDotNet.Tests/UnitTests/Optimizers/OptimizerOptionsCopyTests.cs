using System.Reflection;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

public class OptimizerOptionsCopyTests
{
    [Fact]
    public void ASGDCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>(), out var unmutated);

        AssertCompleteCopy(source, new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>(source), unmutated);
        Assert.Throws<ArgumentNullException>(() =>
            new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void RAdamCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>(), out var unmutated);

        AssertCompleteCopy(source, new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>(source), unmutated);
        Assert.Throws<ArgumentNullException>(() =>
            new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void RpropCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>(), out var unmutated);

        AssertCompleteCopy(source, new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>(source), unmutated);
        Assert.Throws<ArgumentNullException>(() =>
            new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    /// <summary>
    /// Gives every settable property a value distinguishable from its default, and reports the ones it
    /// could not.
    /// </summary>
    /// <remarks>
    /// The reporting is the point. An earlier version fell back to writing the CURRENT value back for any
    /// type it did not handle, which silently removed those properties from the test: a reference property
    /// defaulting to <c>null</c> — <c>DataSampler</c> and <c>LearningRateScheduler</c> both do — stayed
    /// null on both sides, so <c>Assert.Equal(null, null)</c> passed whether or not the copy constructor
    /// touched it. The test claimed to check every settable property while quietly checking a subset.
    /// </remarks>
    [Fact]
    public void BFGSCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(), out var unmutated);

        AssertCompleteCopy(source, new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(source), unmutated);
        Assert.Throws<ArgumentNullException>(() =>
            new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void LBFGSCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(), out var unmutated);

        AssertCompleteCopy(source, new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(source), unmutated);
        Assert.Throws<ArgumentNullException>(() =>
            new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void TrustRegionCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>(), out var unmutated);

        AssertCompleteCopy(source, new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>(source), unmutated);
        Assert.Throws<ArgumentNullException>(() =>
            new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    private static TOptions Populate<TOptions>(TOptions options, out List<string> unmutated)
    {
        var skipped = new List<string>();
        foreach (var property in WritableProperties(typeof(TOptions)))
        {
            object? current = property.GetValue(options);
            object? value = property.PropertyType switch
            {
                // Derived from the current value rather than a literal, so a property whose default
                // happens to equal the literal cannot silently drop out of the sweep. Rprop's
                // InitialLearningRate defaults to exactly the 0.25 an earlier version used.
                Type type when type == typeof(bool) => !(bool)current!,
                Type type when type == typeof(bool?) => !(current as bool? ?? false),
                Type type when type == typeof(int) => (int)current! + 17,
                Type type when type == typeof(int?) => (current as int? ?? 0) + 19,
                Type type when type == typeof(long) => (long)current! + 23L,
                Type type when type == typeof(double) => (double)current! + 1.5,
                Type type when type == typeof(double?) => (current as double? ?? 0.0) + 1.75,
                Type type when type == typeof(float) => (float)current! + 1.25f,
                Type type when type == typeof(string) => (current as string ?? "") + "copy-test-sentinel",
                // FirstOrDefault, not First: a single-member enum (or one whose only member already equals
                // the current value) would otherwise throw out of the helper and read as a crash rather
                // than as this property being uncoverable.
                Type type when type.IsEnum => Enum.GetValues(type).Cast<object>()
                    .FirstOrDefault(candidate => !Equals(candidate, current)),
                _ => null,
            };

            if (value is null || Equals(value, current))
            {
                // A reference property that defaults to a FRESH instance is still covered: source and a
                // default-constructed copy hold different instances, so the reference-equality assertion
                // fails if the copy constructor skipped it. Only a property that is null on both sides
                // proves nothing, and those are the real holes.
                if (current is null)
                {
                    skipped.Add($"{property.Name} ({property.PropertyType.Name})");
                }
                continue;
            }

            property.SetValue(options, value);
        }

        unmutated = skipped;
        return options;
    }

    /// <summary>
    /// Properties that are null by default and whose types this helper cannot instantiate, so a copy of
    /// them cannot be distinguished from an omission.
    /// </summary>
    /// <remarks>
    /// Kept explicit so a NEW uncoverable property fails the test rather than joining a silent hole. Both
    /// entries are interfaces with no in-repo test double reachable from here; every other collaborator
    /// defaults to a fresh instance and is therefore covered by reference identity.
    /// </remarks>
    private static readonly HashSet<string> KnownUncoverableProperties = new HashSet<string>
    {
        "LearningRateScheduler (ILearningRateScheduler)",
        "DataSampler (IDataSampler)",
    };

    private static void AssertCompleteCopy<TOptions>(TOptions source, TOptions copy, List<string> unmutated)
    {
        Assert.NotSame(source, copy);
        foreach (var property in WritableProperties(typeof(TOptions)))
        {
            Assert.True(Equals(property.GetValue(source), property.GetValue(copy)),
                $"{typeof(TOptions).Name}.{property.Name} was not copied: source has " +
                $"{property.GetValue(source) ?? "null"}, copy has {property.GetValue(copy) ?? "null"}.");
        }

        var unexpected = unmutated.Where(name => !KnownUncoverableProperties.Contains(name)).ToList();
        Assert.True(unexpected.Count == 0,
            $"{typeof(TOptions).Name} has settable properties this test could not distinguish from their " +
            $"defaults, so the copy is unverified for them: {string.Join(", ", unexpected)}. Add a case to " +
            "Populate for the type, or add the property to KnownUncoverableProperties with a reason.");
    }

    private static IEnumerable<PropertyInfo> WritableProperties(Type type)
        => type.GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.CanRead && property.CanWrite && property.GetIndexParameters().Length == 0);
}
