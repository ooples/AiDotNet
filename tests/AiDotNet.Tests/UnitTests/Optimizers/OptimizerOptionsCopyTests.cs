using System.Reflection;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

public class OptimizerOptionsCopyTests
{
    [Fact]
    public void ASGDCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>());

        AssertCompleteCopy(source, new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>(source));
        Assert.Throws<ArgumentNullException>(() =>
            new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void RAdamCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>());

        AssertCompleteCopy(source, new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>(source));
        Assert.Throws<ArgumentNullException>(() =>
            new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    [Fact]
    public void RpropCopyConstructor_CopiesEverySettablePropertyAndRejectsNull()
    {
        var source = Populate(new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>());

        AssertCompleteCopy(source, new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>(source));
        Assert.Throws<ArgumentNullException>(() =>
            new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>(null!));
    }

    private static TOptions Populate<TOptions>(TOptions options)
    {
        foreach (var property in WritableProperties(typeof(TOptions)))
        {
            object? current = property.GetValue(options);
            object? value = property.PropertyType switch
            {
                Type type when type == typeof(bool) => !(bool)current!,
                Type type when type == typeof(int) => 17,
                Type type when type == typeof(int?) => 19,
                Type type when type == typeof(double) => 0.25,
                Type type when type.IsEnum => Enum.GetValues(type).Cast<object>()
                    .First(candidate => !Equals(candidate, current)),
                _ => current,
            };

            property.SetValue(options, value);
        }

        return options;
    }

    private static void AssertCompleteCopy<TOptions>(TOptions source, TOptions copy)
    {
        Assert.NotSame(source, copy);
        foreach (var property in WritableProperties(typeof(TOptions)))
        {
            Assert.Equal(property.GetValue(source), property.GetValue(copy));
        }
    }

    private static IEnumerable<PropertyInfo> WritableProperties(Type type)
        => type.GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.CanRead && property.CanWrite && property.GetIndexParameters().Length == 0);
}
