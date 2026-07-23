using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Regression tests for <see cref="AiDotNet.Generators.ModelMetadataValidationGenerator"/> proving
/// the AIDN001 governance is SCOPED to AiDotNet's own shipped library.
/// </summary>
/// <remarks>
/// The generator's AIDN001 diagnostic is a compile Error that fires on any non-abstract
/// <c>IFullModel</c> implementer missing the required metadata attributes. Before the scoping fix it
/// fired against the test project's own IFullModel doubles (and would fire against any downstream
/// consumer's models), producing ~732 spurious errors that blocked the test suite. The fix no-ops
/// the generator unless the compilation being analyzed IS the assembly that DEFINES the metadata
/// attributes (the AiDotNet core library). These tests drive the generator over two synthetic
/// compilations to lock that behavior in:
///   * a library-like compilation that DEFINES the attributes and a model missing one → AIDN001 fires;
///   * a consumer compilation that only REFERENCES the attribute-defining assembly → AIDN001 does NOT fire.
/// </remarks>
public class ModelMetadataValidationGeneratorTests
{
    // Minimal stand-ins for the real AiDotNet metadata attributes and IFullModel interface. The
    // generator matches purely by fully-qualified metadata name, so these reproduce the exact
    // namespaces/names it looks for. A model class is left MISSING the [ModelDomain] attribute so a
    // correctly-scoped run reports AIDN001 for it.
    private const string AttributesAndInterfaceSource = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Class)] public sealed class ModelDomainAttribute : Attribute { public ModelDomainAttribute(int d) { } }
    [AttributeUsage(AttributeTargets.Class)] public sealed class ModelCategoryAttribute : Attribute { public ModelCategoryAttribute(int c) { } }
    [AttributeUsage(AttributeTargets.Class)] public sealed class ModelTaskAttribute : Attribute { public ModelTaskAttribute(int t) { } }
    [AttributeUsage(AttributeTargets.Class)] public sealed class ModelComplexityAttribute : Attribute { public ModelComplexityAttribute(int c) { } }
    [AttributeUsage(AttributeTargets.Class)] public sealed class ModelInputAttribute : Attribute { public ModelInputAttribute(int i) { } }
    [AttributeUsage(AttributeTargets.Class)] public sealed class ResearchPaperAttribute : Attribute { public ResearchPaperAttribute(string title, string url) { } }
    [AttributeUsage(AttributeTargets.Class)] public sealed class ModelMetadataExemptAttribute : Attribute { }
}
namespace AiDotNet.Interfaces
{
    public interface IFullModel<T, TInput, TOutput> { }
}";

    // A concrete IFullModel implementer with NO metadata attributes. Placed in the library
    // compilation it must trigger AIDN001; placed in a consumer compilation it must not.
    private const string ModelSource = @"
namespace SampleModels
{
    using AiDotNet.Interfaces;
    public class UnannotatedModel : IFullModel<double, double, double> { }
}";

    private static ImmutableArray<MetadataReference> BaseReferences()
    {
        // Reference the same core BCL assemblies this test process is running against so the
        // synthetic compilations resolve System.Object / System.Attribute, etc.
        var refs = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (asm.IsDynamic)
                continue;
            var loc = asm.Location;
            if (string.IsNullOrEmpty(loc) || !seen.Add(loc))
                continue;
            try
            {
                refs.Add(MetadataReference.CreateFromFile(loc));
            }
            catch
            {
                // Skip assemblies that can't be turned into a metadata reference.
            }
        }
        return refs.ToImmutableArray();
    }

    private static ImmutableArray<Diagnostic> RunGenerator(CSharpCompilation compilation)
    {
        var generator = new AiDotNet.Generators.ModelMetadataValidationGenerator();
        GeneratorDriver driver = CSharpGeneratorDriver.Create(generator);
        driver = driver.RunGeneratorsAndUpdateCompilation(
            compilation, out _, out var diagnostics);
        return diagnostics;
    }

    private static bool HasAidn001(ImmutableArray<Diagnostic> diagnostics)
        => diagnostics.Any(d => d.Id == "AIDN001");

    /// <summary>
    /// When the compilation DEFINES the metadata attributes (i.e. it IS the AiDotNet core library),
    /// a concrete model missing a required attribute must raise AIDN001.
    /// </summary>
    [Fact]
    public void Fires_AIDN001_For_Library_Model_Missing_Attribute()
    {
        var baseRefs = BaseReferences();
        // Attributes + interface + the unannotated model all live in ONE assembly named "AiDotNet",
        // reproducing the core library: the attribute types are defined in this compilation itself.
        var library = CSharpCompilation.Create(
            assemblyName: "AiDotNet",
            syntaxTrees: new[]
            {
                CSharpSyntaxTree.ParseText(AttributesAndInterfaceSource),
                CSharpSyntaxTree.ParseText(ModelSource),
            },
            references: baseRefs,
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        var diagnostics = RunGenerator(library);

        Assert.True(
            HasAidn001(diagnostics),
            "Expected AIDN001 to fire for a library model missing [ModelDomain], but it did not. " +
            "Diagnostics: " + string.Join(", ", diagnostics.Select(d => d.Id)));
    }

    /// <summary>
    /// When the compilation only REFERENCES the attribute-defining assembly (a downstream consumer,
    /// or the test project with its IFullModel doubles), the generator must no-op — AIDN001 must NOT
    /// fire even though the consumer's model lacks the metadata attributes.
    /// </summary>
    [Fact]
    public void Does_Not_Fire_AIDN001_For_Consumer_Assembly_Model()
    {
        var baseRefs = BaseReferences();
        // The attribute/interface definitions live in a SEPARATE referenced library.
        var library = CSharpCompilation.Create(
            assemblyName: "AiDotNet",
            syntaxTrees: new[] { CSharpSyntaxTree.ParseText(AttributesAndInterfaceSource) },
            references: baseRefs,
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        // The consumer defines its own IFullModel model but only REFERENCES the attribute assembly.
        var consumer = CSharpCompilation.Create(
            assemblyName: "ConsumerApp",
            syntaxTrees: new[] { CSharpSyntaxTree.ParseText(ModelSource) },
            references: baseRefs.Add(library.ToMetadataReference()),
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        // Sanity: the consumer model genuinely implements IFullModel and lacks the attributes, so
        // an UNSCOPED generator WOULD have flagged it. (The library compilation itself, run above,
        // proves the flagging path is live.)
        var diagnostics = RunGenerator(consumer);

        Assert.False(
            HasAidn001(diagnostics),
            "AIDN001 must NOT fire for a consumer/test assembly that only references AiDotNet, " +
            "but it did. Diagnostics: " + string.Join(", ", diagnostics.Select(d => d.Id)));
    }
}
