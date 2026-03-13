using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that cross-references model classes against test classes
/// to identify untested models and generate a coverage report.
/// </summary>
/// <remarks>
/// <para>
/// Discovers all concrete IFullModel implementations decorated with [ModelDomain] and checks
/// for matching test classes. Emits a static <c>TestCoverage</c> class with coverage statistics
/// and compile-time warnings for untested models.
/// </para>
/// </remarks>
[Generator]
public class TestScaffoldGenerator : IIncrementalGenerator
{
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";
    private const string ModelDomainAttr = "AiDotNet.Attributes.ModelDomainAttribute";

    private static readonly DiagnosticDescriptor UntestedModel = new(
        id: "AIDN040",
        title: "Model has no test coverage",
        messageFormat: "Model '{0}' has no corresponding test class (expected: '{0}Tests' or similar)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Every model class should have at least basic test coverage.");

    private static readonly DiagnosticDescriptor CoverageSummary = new(
        id: "AIDN041",
        title: "Model test coverage summary",
        messageFormat: "{0} of {1} annotated models have test coverage ({2:F1}%)",
        category: "AiDotNet.TestCoverage",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Collect model classes
        var modelClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsModelCandidate(node),
            transform: static (ctx, _) => GetModelClassOrNull(ctx))
            .Where(static s => s is not null);

        // Collect test classes (classes containing [Fact] or [Theory])
        var testClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsTestCandidate(node),
            transform: static (ctx, _) => GetTestClassName(ctx))
            .Where(static s => s is not null);

        var combined = modelClasses.Collect()
            .Combine(testClasses.Collect())
            .Combine(context.CompilationProvider);

        context.RegisterSourceOutput(combined, static (spc, source) =>
        {
            var ((models, tests), compilation) = source;
            Execute(spc, models, tests, compilation);
        });
    }

    private static bool IsModelCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;
        if (cds.BaseList is null || cds.BaseList.Types.Count == 0)
            return false;
        foreach (var modifier in cds.Modifiers)
        {
            if (modifier.Text == "abstract")
                return false;
        }
        return true;
    }

    private static bool IsTestCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;

        // Check if class name ends with "Tests" or "Test"
        if (cds.Identifier.Text.EndsWith("Tests", System.StringComparison.Ordinal) ||
            cds.Identifier.Text.EndsWith("Test", System.StringComparison.Ordinal))
        {
            return true;
        }

        // Check if any method has [Fact] or [Theory] attribute
        foreach (var member in cds.Members)
        {
            if (member is MethodDeclarationSyntax method)
            {
                foreach (var attrList in method.AttributeLists)
                {
                    foreach (var attr in attrList.Attributes)
                    {
                        var name = attr.Name.ToString();
                        if (name == "Fact" || name == "Theory" ||
                            name == "Xunit.Fact" || name == "Xunit.Theory")
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    private static INamedTypeSymbol? GetModelClassOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(IFullModelName, System.StringComparison.Ordinal))
            {
                return symbol;
            }
        }
        return null;
    }

    private static string? GetTestClassName(GeneratorSyntaxContext ctx)
    {
        if (ctx.Node is not ClassDeclarationSyntax cds)
            return null;
        return cds.Identifier.Text;
    }

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> models,
        ImmutableArray<string?> testClassNames,
        Compilation compilation)
    {
        var domainAttrSymbol = compilation.GetTypeByMetadataName(ModelDomainAttr);

        // Build test class name set for fast lookup
        var testNames = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
        foreach (var name in testClassNames)
        {
            if (name is not null)
                testNames.Add(name);
        }

        var testedModels = new List<ModelTestInfo>();
        var untestedModels = new List<ModelTestInfo>();
        var seen = new HashSet<string>();

        foreach (var modelClass in models)
        {
            if (modelClass is null)
                continue;

            var fullName = modelClass.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName))
                continue;

            // Only include models with [ModelDomain] attribute
            bool hasModelDomain = false;
            var domains = new List<int>();
            if (domainAttrSymbol is not null)
            {
                foreach (var attr in modelClass.GetAttributes())
                {
                    if (attr.AttributeClass is not null &&
                        SymbolEqualityComparer.Default.Equals(attr.AttributeClass, domainAttrSymbol))
                    {
                        hasModelDomain = true;
                        if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is int d)
                            domains.Add(d);
                    }
                }
            }

            if (!hasModelDomain)
                continue;

            var className = modelClass.Name;
            var info = new ModelTestInfo
            {
                ClassName = className,
                FullyQualifiedName = fullName,
                TypeParameterCount = modelClass.TypeParameters.Length,
                Domains = domains,
                Location = modelClass.Locations.Length > 0 ? modelClass.Locations[0] : null
            };

            // Check for matching test class
            bool hasCoverage = HasTestCoverage(className, testNames);
            info.HasTests = hasCoverage;

            if (hasCoverage)
            {
                testedModels.Add(info);
            }
            else
            {
                untestedModels.Add(info);
            }
        }

        testedModels.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));
        untestedModels.Sort((a, b) => string.Compare(a.ClassName, b.ClassName, System.StringComparison.Ordinal));

        // Emit AIDN040 diagnostic for each untested model
        foreach (var model in untestedModels)
        {
            if (model.Location is not null)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    UntestedModel,
                    model.Location,
                    model.ClassName));
            }
        }

        // Emit AIDN041 summary diagnostic
        var totalCount = testedModels.Count + untestedModels.Count;
        if (totalCount > 0)
        {
            var coveragePct = testedModels.Count * 100.0 / totalCount;
            context.ReportDiagnostic(Diagnostic.Create(
                CoverageSummary,
                Location.None,
                testedModels.Count,
                totalCount,
                coveragePct));
        }

        EmitTestCoverageClass(context, testedModels, untestedModels);
    }

    private static bool HasTestCoverage(string modelClassName, HashSet<string> testNames)
    {
        // Strip generic arity suffix
        var baseName = modelClassName;
        var backtick = baseName.IndexOf('`');
        if (backtick >= 0)
            baseName = baseName.Substring(0, backtick);

        // Check common test naming conventions
        if (testNames.Contains(baseName + "Tests")) return true;
        if (testNames.Contains(baseName + "Test")) return true;
        if (testNames.Contains(baseName + "_Tests")) return true;
        if (testNames.Contains(baseName + "IntegrationTests")) return true;
        if (testNames.Contains(baseName + "UnitTests")) return true;

        // Check if any test class name contains the model name (for broad matches)
        foreach (var testName in testNames)
        {
            if (testName.IndexOf(baseName, System.StringComparison.OrdinalIgnoreCase) >= 0)
                return true;
        }

        return false;
    }

    private static void EmitTestCoverageClass(
        SourceProductionContext context,
        List<ModelTestInfo> testedModels,
        List<ModelTestInfo> untestedModels)
    {
        var totalCount = testedModels.Count + untestedModels.Count;
        var coveragePercent = totalCount > 0
            ? (testedModels.Count * 100.0 / totalCount)
            : 0.0;

        // Domain name map for grouping
        var domainNames = new Dictionary<int, string>
        {
            {0, "General"}, {1, "Vision"}, {2, "Language"}, {3, "Audio"},
            {4, "Video"}, {5, "Multimodal"}, {6, "Healthcare"}, {7, "Finance"},
            {8, "Science"}, {9, "Robotics"}, {10, "GraphAnalysis"}, {11, "ThreeD"},
            {12, "Tabular"}, {13, "TimeSeries"}, {14, "Generative"},
            {15, "ReinforcementLearning"}, {16, "Causal"}, {17, "MachineLearning"}
        };

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Collections.Generic;");
        sb.AppendLine("using System.Linq;");
        sb.AppendLine("using AiDotNet.Enums;");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Generated;");
        sb.AppendLine();

        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Auto-generated test coverage report for annotated model classes.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("internal static class TestCoverage");
        sb.AppendLine("{");

        // Constants
        sb.AppendLine($"    /// <summary>Total annotated models tracked.</summary>");
        sb.AppendLine($"    public const int TotalModels = {totalCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Models with test coverage.</summary>");
        sb.AppendLine($"    public const int TestedCount = {testedModels.Count};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Models without test coverage.</summary>");
        sb.AppendLine($"    public const int UntestedCount = {untestedModels.Count};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Coverage percentage.</summary>");
        sb.AppendLine($"    public const double CoveragePercent = {coveragePercent:F1};");
        sb.AppendLine();

        // TestedModels array
        sb.AppendLine("    /// <summary>Models that have corresponding test classes.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> TestedModels { get; } = new Type[]");
        sb.AppendLine("    {");
        foreach (var model in testedModels)
        {
            sb.AppendLine($"        {BuildTypeOfExpression(model)},");
        }
        sb.AppendLine("    };");
        sb.AppendLine();

        // UntestedModels array
        sb.AppendLine("    /// <summary>Models that do NOT have corresponding test classes.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> UntestedModels { get; } = new Type[]");
        sb.AppendLine("    {");
        foreach (var model in untestedModels)
        {
            sb.AppendLine($"        {BuildTypeOfExpression(model)},");
        }
        sb.AppendLine("    };");
        sb.AppendLine();

        // GetUntestedByDomain
        sb.AppendLine("    /// <summary>Gets untested models for a specific domain.</summary>");
        sb.AppendLine("    public static IReadOnlyList<Type> GetUntestedByDomain(ModelDomain domain)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (_untestedByDomain.TryGetValue(domain, out var types))");
        sb.AppendLine("            return types;");
        sb.AppendLine("        return Array.Empty<Type>();");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Build untested by domain dictionary
        var untestedByDomain = new Dictionary<int, List<ModelTestInfo>>();
        foreach (var model in untestedModels)
        {
            foreach (var d in model.Domains)
            {
                if (!untestedByDomain.TryGetValue(d, out var list))
                {
                    list = new List<ModelTestInfo>();
                    untestedByDomain[d] = list;
                }
                list.Add(model);
            }
        }

        sb.AppendLine("    private static readonly Dictionary<ModelDomain, Type[]> _untestedByDomain = new Dictionary<ModelDomain, Type[]>");
        sb.AppendLine("    {");
        foreach (var kvp in untestedByDomain.OrderBy(k => k.Key))
        {
            if (!domainNames.TryGetValue(kvp.Key, out var domainName))
                continue;

            sb.Append($"        {{ ModelDomain.{domainName}, new Type[] {{ ");
            var sorted = kvp.Value.OrderBy(m => m.ClassName).ToList();
            for (int i = 0; i < sorted.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(BuildTypeOfExpression(sorted[i]));
            }
            sb.AppendLine(" } },");
        }
        sb.AppendLine("    };");

        sb.AppendLine("}");

        context.AddSource("TestCoverage.g.cs", sb.ToString());
    }

    private static string BuildTypeOfExpression(ModelTestInfo entry)
    {
        var typeName = StripGenericSuffix(entry.FullyQualifiedName);

        if (entry.TypeParameterCount > 0)
        {
            var commas = new string(',', entry.TypeParameterCount - 1);
            return $"typeof({typeName}<{commas}>)";
        }
        return $"typeof({typeName})";
    }

    private static string StripGenericSuffix(string fullyQualifiedName)
    {
        var name = fullyQualifiedName;
        if (name.StartsWith("global::", System.StringComparison.Ordinal))
            name = name.Substring("global::".Length);
        var angleBracketIdx = name.IndexOf('<');
        if (angleBracketIdx >= 0)
            name = name.Substring(0, angleBracketIdx);
        return name;
    }

    private class ModelTestInfo
    {
        public string ClassName { get; set; } = string.Empty;
        public string FullyQualifiedName { get; set; } = string.Empty;
        public int TypeParameterCount { get; set; }
        public List<int> Domains { get; set; } = new List<int>();
        public bool HasTests { get; set; }
        public Location? Location { get; set; }
    }
}
