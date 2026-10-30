using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using AiDotNet.Attributes;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Checks that an options class's parameterless constructor ships every value its
/// <see cref="PaperDefaultAttribute"/> declarations say the paper states.
/// </summary>
/// <remarks>
/// The scaffold generator emits one test per options class that declares any paper default, calling
/// <see cref="AssertConstructorDefaultsMatchPaper"/> with the closed options type. All mismatches are
/// collected and reported together, each with its declared source, so one run lists every drifted
/// default instead of stopping at the first.
/// </remarks>
public static class PaperDefaultsHarness
{
    /// <summary>
    /// Constructs <paramref name="optionsType"/> with its parameterless constructor and asserts every
    /// <see cref="PaperDefaultAttribute"/> on it (including inherited ones) holds.
    /// </summary>
    /// <param name="optionsType">The closed options type to construct.</param>
    public static void AssertConstructorDefaultsMatchPaper(Type optionsType)
    {
        if (optionsType is null) throw new ArgumentNullException(nameof(optionsType));
        object instance = Activator.CreateInstance(optionsType)
            ?? throw new InvalidOperationException($"{optionsType.Name} could not be constructed.");

        var mismatches = new List<string>();
        int declared = 0;
        foreach (var property in optionsType.GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            var declaration = property.GetCustomAttribute<PaperDefaultAttribute>(inherit: true);
            if (declaration is null) continue;
            declared++;
            object? actual = property.GetValue(instance);
            if (!Matches(declaration.Value, actual))
            {
                mismatches.Add($"{property.Name}: paper value {Describe(declaration.Value)} ({declaration.Source}), "
                    + $"constructor default {Describe(actual)}");
            }
        }

        Assert.True(declared > 0, $"{optionsType.Name} declares no [PaperDefault]; this test should not have been generated.");
        Assert.True(mismatches.Count == 0,
            $"{optionsType.Name} does not ship the values its paper states:\n  " + string.Join("\n  ", mismatches));
    }

    /// <summary>Whether a property value equals a declared paper value.</summary>
    /// <remarks>
    /// Numbers compare after conversion to <see cref="double"/>, so a generic <c>T</c> property closed
    /// over double or float matches a declared <c>0.99</c>; a float property is compared at float
    /// precision, because the declared double literal is rounded on assignment. Lists compare element
    /// by element.
    /// </remarks>
    internal static bool Matches(object? declared, object? actual)
    {
        if (declared is null || actual is null) return declared is null && actual is null;
        if (declared is not string && declared is IEnumerable expectedItems)
        {
            if (actual is not IEnumerable actualItems) return false;
            var e = expectedItems.Cast<object?>().ToList();
            var a = actualItems.Cast<object?>().ToList();
            return e.Count == a.Count && e.Zip(a, Matches).All(match => match);
        }

        if (IsNumeric(declared) && IsNumeric(actual))
        {
            double expected = Convert.ToDouble(declared, CultureInfo.InvariantCulture);
            double value = Convert.ToDouble(actual, CultureInfo.InvariantCulture);
            if (actual is float) return (float)expected == (float)value;
            return expected == value;
        }

        if (actual.GetType().IsEnum && !declared.GetType().IsEnum)
            return Convert.ToInt64(actual, CultureInfo.InvariantCulture) == Convert.ToInt64(declared, CultureInfo.InvariantCulture);

        return declared.Equals(actual);
    }

    private static bool IsNumeric(object value)
        => value is byte or sbyte or short or ushort or int or uint or long or ulong or float or double or decimal;

    private static string Describe(object? value)
    {
        if (value is null) return "null";
        if (value is string text) return "\"" + text + "\"";
        if (value is IEnumerable sequence)
            return "[" + string.Join(", ", sequence.Cast<object?>().Select(Describe)) + "]";
        return Convert.ToString(value, CultureInfo.InvariantCulture) ?? value.GetType().Name;
    }
}
