using System.Linq;
using Microsoft.CodeAnalysis;

namespace AiDotNet.Generators;

/// <summary>
/// Whether a hand-written <c>*PaperTests</c> class can actually run a test, which is what clears ADNGEN003.
/// </summary>
/// <remarks>
/// Its own file so AiDotNetTests can compile it in and check it against real symbols, as it does for the other
/// generator helpers it links.
/// </remarks>
internal static class PaperTestDetection
{
    /// <summary>
    /// True when <paramref name="type"/> is public, can be run by xUnit (a concrete class, or a static class, whose
    /// test methods are static), and declares at least one test method xUnit will execute: a fact, or a theory with a
    /// data source, whose attribute carries no constant <c>Skip</c> reason. An abstract class, an always-skipped test
    /// and a theory with no data all prove nothing about the paper.
    /// </summary>
    public static bool HasRunnableTest(INamedTypeSymbol type)
    {
        // A static class is abstract and sealed to Roslyn, and xUnit runs its static test methods.
        if ((type.IsAbstract && !type.IsStatic) || type.DeclaredAccessibility != Accessibility.Public) return false;

        foreach (var method in type.GetMembers().OfType<IMethodSymbol>())
        {
            var attributes = method.GetAttributes();
            foreach (var attribute in attributes)
            {
                string? name = attribute.AttributeClass?.Name;
                bool fact = name is "FactAttribute" or "SkippableFactAttribute";
                bool theory = name is "TheoryAttribute" or "SkippableTheoryAttribute";
                if (!fact && !theory) continue;
                bool staticallySkipped = attribute.NamedArguments.Any(argument =>
                    argument.Key == "Skip" && argument.Value.Value is string reason && reason.Length > 0);
                if (staticallySkipped) continue;
                // xUnit does not run a theory without data (xUnit1003).
                if (theory && !attributes.Any(IsDataSource)) continue;
                return true;
            }
        }

        return false;
    }

    // InlineData, MemberData, ClassData and any custom source: every xUnit data attribute derives from DataAttribute.
    private static bool IsDataSource(AttributeData attribute)
    {
        for (var type = attribute.AttributeClass; type is not null; type = type.BaseType)
        {
            if (type.Name == "DataAttribute") return true;
        }

        return false;
    }
}
