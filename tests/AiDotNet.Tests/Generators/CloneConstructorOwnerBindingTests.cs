using System.Reflection;
using AiDotNet.Generators;
using AiDotNet.Models;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class CloneConstructorOwnerBindingTests
{
    public CloneConstructorOwnerBindingTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, false, true)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, false)]
    [InlineData(true, false, true)]
    public void GeneratedOwnerIdentity_PreservesShadowedGenericConfiguration(bool baseProperty, bool derivedProperty, bool metadataBase)
    {
        var rootType = BuildFixture(baseProperty, derivedProperty, metadataBase);
        var create = rootType.GetMethod("Create") ?? throw new InvalidOperationException("Missing fixture factory.");
        var original = create.Invoke(null, null) ?? throw new InvalidOperationException("Factory returned null.");
        Assert.True(CloneRegistry.IsVerified(rootType));

        var clone = CloneEngine.CopyConfiguration(original);
        Assert.NotSame(original, clone);
        Assert.Equal(23, ReadInt(clone, "OwnWidth"));
        Assert.Equal(47, ReadInt(clone, "BaseWidth"));
        Assert.Equal(23, ReadInt(original, "OwnWidth"));
        Assert.Equal(47, ReadInt(original, "BaseWidth"));
    }

    private static int ReadInt(object value, string property)
        => Assert.IsType<int>(value.GetType().GetProperty(property)?.GetValue(value));

    private static Type BuildFixture(bool baseProperty, bool derivedProperty, bool metadataBase)
    {
        string fixtureNamespace = "CloneOwner_" + Guid.NewGuid().ToString("N");
        string baseMember = baseProperty
            ? "protected BaseSettings<T> _options { get; }"
            : "protected readonly BaseSettings<T> _options;";
        string derivedMember = derivedProperty
            ? "private new OwnSettings _options { get; }"
            : "private new readonly OwnSettings _options;";
        string baseSource = $$"""
            #nullable enable
            namespace {{fixtureNamespace}};
            public sealed class OwnSettings : AiDotNet.Models.Options.ModelOptions
            {
                public int Width { get; set; } = 1;
            }
            public sealed class BaseSettings<T> : AiDotNet.Models.Options.ModelOptions
            {
                public int Width { get; set; } = 2;
            }
            public abstract class GenericBase<T> : AiDotNet.Models.Options.ModelOptions
            {
                {{baseMember}}
                public int BaseWidth => _options.Width;
                protected GenericBase(BaseSettings<T>? options) => _options = options ?? new BaseSettings<T>();
            }
            """;
        string derivedSource = $$"""
            #nullable enable
            namespace {{fixtureNamespace}};
            public sealed class Concrete<T> : GenericBase<T>
            {
                {{derivedMember}}
                public int OwnWidth => _options.Width;
                public Concrete(OwnSettings? options = null, BaseSettings<T>? baseOptions = null) : base(baseOptions)
                    => _options = options ?? new OwnSettings();
                public static Concrete<T> Create() => new Concrete<T>(
                    new OwnSettings { Width = 23 }, new BaseSettings<T> { Width = 47 });
            }
            """;
        var references = RuntimeReferences();
        var trees = new List<SyntaxTree> { CSharpSyntaxTree.ParseText(derivedSource) };
        if (metadataBase)
        {
            var baseCompilation = CSharpCompilation.Create(fixtureNamespace + "_Base",
                new[] { CSharpSyntaxTree.ParseText(baseSource) }, references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
            using var baseBytes = new MemoryStream();
            var baseEmit = baseCompilation.Emit(baseBytes);
            Assert.True(baseEmit.Success, string.Join(Environment.NewLine, baseEmit.Diagnostics));
            byte[] image = baseBytes.ToArray();
            _ = LoadFixtureAssembly(image);
            references.Add(MetadataReference.CreateFromImage(image));
        }
        else
        {
            trees.Add(CSharpSyntaxTree.ParseText(baseSource));
        }

        var compilation = CSharpCompilation.Create(fixtureNamespace + "_Model", trees, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
                .WithMetadataImportOptions(MetadataImportOptions.All));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new ClonePlanGenerator().AsSourceGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out var generated, out var diagnostics);
        Assert.Empty(diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        using var bytes = new MemoryStream();
        var emit = generated.Emit(bytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics));
        var assembly = LoadFixtureAssembly(bytes.ToArray());
        var register = assembly.GetType("AiDotNet.Generated.CloneRegistrations")?.GetMethod("RegisterAll",
            BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            ?? throw new InvalidOperationException("Missing actual generated registration.");
        register.Invoke(null, null);
        return (assembly.GetType(fixtureNamespace + ".Concrete`1")
            ?? throw new InvalidOperationException("Missing fixture type.")).MakeGenericType(typeof(int));
    }

    private static List<MetadataReference> RuntimeReferences()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location)) continue;
            references.Add(MetadataReference.CreateFromFile(assembly.Location));
        }
        return references;
    }

    private static Assembly LoadFixtureAssembly(byte[] image)
    {
#if NETFRAMEWORK
        return Assembly.Load(image);
#else
        using var bytes = new MemoryStream(image, writable: false);
        return System.Runtime.Loader.AssemblyLoadContext.Default.LoadFromStream(bytes);
#endif
    }
}
