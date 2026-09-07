using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Semantic contract tests for paper-recipe declaration and wiring diagnostics.</summary>
public sealed class PaperOptimizerAnalyzerTests
{
    private const string Infrastructure = @"
using System;
namespace AiDotNet.Enums
{
    public enum OptimizerKind { Unspecified, Adam, AdamW }
}
namespace AiDotNet.Attributes
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    public sealed class PaperOptimizerAttribute : Attribute
    {
        public PaperOptimizerAttribute(AiDotNet.Enums.OptimizerKind optimizer) { Optimizer = optimizer; }
        public AiDotNet.Enums.OptimizerKind Optimizer { get; }
        public string Source { get; set; } = string.Empty;
        public string Variant { get; set; } = string.Empty;
        public double Momentum { get; set; } = double.NaN;
    }
    [AttributeUsage(AttributeTargets.Class, Inherited = false)]
    public sealed class ResearchPaperAttribute : Attribute { }
}
namespace AiDotNet.Interfaces
{
    public interface IGradientBasedOptimizer<T, TInput, TOutput> { }
}
namespace AiDotNet.Optimizers
{
    public sealed class RealOptimizer : AiDotNet.Interfaces.IGradientBasedOptimizer<object, object, object>
    {
        public RealOptimizer(object model) { }
        public RealOptimizer(object model, object options) { }
    }
    public sealed class CacheOptimizer
    {
        public CacheOptimizer(object owner) { }
    }
    internal static class PaperOptimizerFactory
    {
        internal static AiDotNet.Interfaces.IGradientBasedOptimizer<T, TInput, TOutput> CreateFor<T, TInput, TOutput>(object model)
            => throw new NotImplementedException();
    }
}";

    private static ImmutableArray<MetadataReference> BaseReferences()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location))
                continue;
            references.Add(MetadataReference.CreateFromFile(assembly.Location));
        }

        return references.ToImmutableArray();
    }

    private static async Task<ImmutableArray<Diagnostic>> RunAsync(params string[] sources)
    {
        var trees = new List<SyntaxTree> { CSharpSyntaxTree.ParseText(Infrastructure) };
        trees.AddRange(sources.Select(source => CSharpSyntaxTree.ParseText(source)));
        var compilation = CSharpCompilation.Create(
            "AiDotNet",
            trees,
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        return await compilation.WithAnalyzers(
                ImmutableArray.Create<DiagnosticAnalyzer>(new AiDotNet.Generators.PaperOptimizerAnalyzer()))
            .GetAnalyzerDiagnosticsAsync();
    }

    [Fact]
    public async Task OptimizerOnlyDeclaration_StillRequiresSource()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
[PaperOptimizer(OptimizerKind.Adam)]
public sealed class Model { }";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN102"));
    }

    [Fact]
    public async Task AbstractOwner_IsDiagnosedOnce_NotOncePerDerivedType()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
[PaperOptimizer(OptimizerKind.Adam)]
public abstract class FamilyBase { }
public sealed class Small : FamilyBase { }
public sealed class Large : FamilyBase { }";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN102"));
    }

    [Fact]
    public async Task InheritedRecipe_RequiresConcreteSelectionToUseFactory()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Optimizers;
[PaperOptimizer(OptimizerKind.Adam, Source = ""fixture"")]
public abstract class FamilyBase { }
public sealed class Concrete : FamilyBase
{
    private readonly object optimizer;
    public Concrete() { optimizer = new RealOptimizer(this); }
}";

        Diagnostic diagnostic = Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN104"));
        Assert.Contains("Concrete", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task FullyQualifiedFactoryCall_InSameSelectionExpression_IsAccepted()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Optimizers;
[PaperOptimizer(OptimizerKind.Adam, Source = ""fixture"")]
public sealed class Model
{
    private readonly object optimizer;
    public Model()
    {
        optimizer = AiDotNet.Optimizers.PaperOptimizerFactory.CreateFor<object, object, object>(this)
            ?? new RealOptimizer(this);
    }
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "AIDN104"));
    }

    [Fact]
    public async Task UnrelatedClassWhoseNameEndsInOptimizer_IsNotTrainingWiring()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Optimizers;
[PaperOptimizer(OptimizerKind.Adam, Source = ""fixture"")]
public sealed class Model
{
    private readonly object cache = new CacheOptimizer(new object());
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "AIDN104"));
    }

    [Fact]
    public async Task FactoryCallElsewhere_DoesNotPretendToControlHardcodedOptimizer()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Optimizers;
[PaperOptimizer(OptimizerKind.Adam, Source = ""fixture"")]
public sealed class Model
{
    private readonly object optimizer = new RealOptimizer(new object());
    public object Unused() => PaperOptimizerFactory.CreateFor<object, object, object>(this);
}";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN104"));
    }

    [Fact]
    public async Task VariantKey_IsUniqueAcrossOptimizerKinds()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
[PaperOptimizer(OptimizerKind.Adam, Variant = ""Tiny"", Source = ""fixture"")]
[PaperOptimizer(OptimizerKind.AdamW, Variant = ""Tiny"", Source = ""fixture"")]
public sealed class Model { }";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN103"));
    }

    [Fact]
    public async Task DerivedDeclaration_CannotShadowInheritedVariant()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
[PaperOptimizer(OptimizerKind.Adam, Variant = ""Tiny"", Source = ""base fixture"")]
public abstract class FamilyBase { }
[PaperOptimizer(OptimizerKind.AdamW, Variant = ""Tiny"", Source = ""derived fixture"")]
public sealed class Model : FamilyBase { }";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN103"));
    }

    [Fact]
    public async Task PartialType_IsAnalyzedAsOneCompleteType()
    {
        const string first = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
[PaperOptimizer(OptimizerKind.Adam)]
public sealed partial class Model { }";
        const string second = @"
using AiDotNet.Optimizers;
public sealed partial class Model
{
    private readonly object optimizer = new RealOptimizer(new object());
}";

        ImmutableArray<Diagnostic> diagnostics = await RunAsync(first, second);
        Assert.Single(diagnostics.Where(item => item.Id == "AIDN102"));
        Assert.Single(diagnostics.Where(item => item.Id == "AIDN104"));
    }

    [Fact]
    public async Task MissingRecipeUsesSemanticOptimizerIdentity()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Optimizers;
[ResearchPaper]
public sealed class RealModel
{
    private readonly object optimizer = new RealOptimizer(new object());
}
[ResearchPaper]
public sealed class CacheOwner
{
    private readonly object cache = new CacheOptimizer(new object());
}";

        Diagnostic diagnostic = Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN101"));
        Assert.Contains("RealModel", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task DeliberateOptionsConstruction_IsNotReportedAsGenericDefault()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Optimizers;
[ResearchPaper]
public sealed class Model
{
    private readonly object optimizer = new RealOptimizer(new object(), new object());
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id is "AIDN101" or "AIDN104"));
    }
}
