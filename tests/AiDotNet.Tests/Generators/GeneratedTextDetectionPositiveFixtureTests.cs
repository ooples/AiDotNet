using System.Reflection;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Generators;
using AiDotNet.Models;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Uses the real scaffold generator and actual detector metadata, not hand-written leaf fixtures.</summary>
public sealed class GeneratedTextDetectionPositiveFixtureTests
{
    public enum TextDetectorKind { Craft, DifferentiableBinarization, East }
    public enum InvalidPositiveResult { Empty, EmptyPolygon, DegeneratePolygon, InvertedBox, NonFinitePolygon, NonFiniteBox, InvalidConfidence, BoxDoesNotBoundPolygon }

    private static readonly Lazy<IReadOnlyDictionary<TextDetectorKind, SyntaxTree>> GeneratedFixtures = new(GenerateFixtures);
    private static readonly Lazy<Assembly> RuntimeFixtures = new(CompileFixtures);

    public GeneratedTextDetectionPositiveFixtureTests() => TestModuleInitializer.EnsureInitialized();

    [Theory(Timeout = 120000)]
    [InlineData(TextDetectorKind.Craft)]
    [InlineData(TextDetectorKind.DifferentiableBinarization)]
    [InlineData(TextDetectorKind.East)]
    public async Task GeneratedPositiveFactory_UsesTheTypedOptionsConstructor(TextDetectorKind kind)
    {
        await Task.Yield();
        var declaration = Assert.Single(GeneratedFixtures.Value[kind].GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>());
        var factory = Assert.Single(declaration.Members.OfType<MethodDeclarationSyntax>(),
            method => method.Identifier.ValueText == "CreatePositiveTextDetector");
        Assert.Equal("TextDetectionOptions<double>", Assert.Single(factory.ParameterList.Parameters).Type?.ToString().Split('.').Last());
        var construction = Assert.Single(factory.DescendantNodes().OfType<ObjectCreationExpressionSyntax>());
        Assert.Equal(ModelType(kind).Name.Split('`')[0] + "<double>", construction.Type.ToString().Split('.').Last());
        Assert.Equal("options", Assert.Single(construction.ArgumentList?.Arguments ?? default).ToString());
    }

    [Theory(Timeout = 120000)]
    [InlineData(TextDetectorKind.Craft)]
    [InlineData(TextDetectorKind.DifferentiableBinarization)]
    [InlineData(TextDetectorKind.East)]
    public async Task GeneratedPositiveInvariant_RunsTheActualDetector(TextDetectorKind kind)
    {
        await Task.Yield();
        var fixture = CreateFixture(kind);
        await fixture.Detect_ControlledPositiveHead_ShouldProduceScorableRegions();
    }

    [Theory(Timeout = 120000)]
    [InlineData(TextDetectorKind.Craft)]
    [InlineData(TextDetectorKind.DifferentiableBinarization)]
    [InlineData(TextDetectorKind.East)]
    public async Task GeneratedDefaultFixture_PreservesTheExistingGeometryInvariants(TextDetectorKind kind)
    {
        await Task.Yield();
        var fixture = CreateFixture(kind);
        await fixture.Detect_PolygonsShouldEncloseArea();
        await fixture.Detect_BoxesShouldBeGeometricallyValid();
        await fixture.Detect_BoxShouldBoundItsPolygon();
    }

    [Fact]
    public void PositiveOracle_AcceptsTheKnownNonemptyGeometry()
        => TextDetectionTestBase<double>.AssertPositiveTextResult(ValidResult(), 1, (0, 0, 63, 63));

    [Theory]
    [InlineData(InvalidPositiveResult.Empty)]
    [InlineData(InvalidPositiveResult.EmptyPolygon)]
    [InlineData(InvalidPositiveResult.DegeneratePolygon)]
    [InlineData(InvalidPositiveResult.InvertedBox)]
    [InlineData(InvalidPositiveResult.NonFinitePolygon)]
    [InlineData(InvalidPositiveResult.NonFiniteBox)]
    [InlineData(InvalidPositiveResult.InvalidConfidence)]
    [InlineData(InvalidPositiveResult.BoxDoesNotBoundPolygon)]
    public void PositiveOracle_RejectsEmptyOrMalformedResults(InvalidPositiveResult corruption)
    {
        var result = ValidResult();
        var region = Assert.Single(result.TextRegions);
        var expectedFirstBox = (Left: 0.0, Top: 0.0, Right: 63.0, Bottom: 63.0);
        switch (corruption)
        {
            case InvalidPositiveResult.Empty:
                result.TextRegions.Clear();
                break;
            case InvalidPositiveResult.EmptyPolygon:
                region.Polygon = new List<(double X, double Y)>();
                break;
            case InvalidPositiveResult.DegeneratePolygon:
                region.Polygon = new List<(double X, double Y)> { (0, 0), (1, 1), (2, 2), (3, 3) };
                break;
            case InvalidPositiveResult.InvertedBox:
                region.Box = new BoundingBox<double>(10, 0, 0, 63);
                break;
            case InvalidPositiveResult.NonFinitePolygon:
                var vertices = region.Polygon ?? throw new InvalidOperationException("The valid fixture has no polygon.");
                vertices[0] = (double.NaN, 0);
                break;
            case InvalidPositiveResult.NonFiniteBox:
                region.Box = new BoundingBox<double>(0, 0, double.PositiveInfinity, 63);
                expectedFirstBox = (0, 0, double.PositiveInfinity, 63);
                break;
            case InvalidPositiveResult.InvalidConfidence:
                region.Confidence = double.NaN;
                break;
            case InvalidPositiveResult.BoxDoesNotBoundPolygon:
                region.Box = new BoundingBox<double>(1, 1, 62, 62);
                expectedFirstBox = (1, 1, 62, 62);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(corruption));
        }
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() =>
            TextDetectionTestBase<double>.AssertPositiveTextResult(result, 1, expectedFirstBox));
    }

    private static TextDetectionResult<double> ValidResult() => new()
    {
        ImageWidth = 64,
        ImageHeight = 64,
        TextRegions = new List<TextRegion<double>>
        {
            TextRegion<double>.FromPolygon(new List<(double X, double Y)> { (0, 0), (63, 0), (63, 63), (0, 63) }, 0.5)
        }
    };

    private static TextDetectionTestBase CreateFixture(TextDetectorKind kind)
    {
        string typeName = "AiDotNet.Tests.ModelFamilyTests.Generated." + ModelType(kind).Name.Split('`')[0] + "Tests";
        var fixtureType = RuntimeFixtures.Value.GetType(typeName)
            ?? throw new InvalidOperationException("The exact generated text fixture was not compiled.");
        return Assert.IsAssignableFrom<TextDetectionTestBase>(Activator.CreateInstance(fixtureType));
    }

    private static Assembly CompileFixtures()
    {
        var trees = GeneratedFixtures.Value.Values.Concat(new[]
        {
            CSharpSyntaxTree.ParseText("global using System; global using System.Linq; global using System.Collections.Generic; global using AiDotNet.Tensors.LinearAlgebra;")
        });
        var compilation = CSharpCompilation.Create("GeneratedTextDetectionReview_" + Guid.NewGuid().ToString("N"),
            trees, References(includeTestAssembly: true), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var stream = new MemoryStream();
        var emit = compilation.Emit(stream);
        Assert.True(emit.Success, string.Join(Environment.NewLine,
            emit.Diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        return Assembly.Load(stream.ToArray());
    }

    private static IReadOnlyDictionary<TextDetectorKind, SyntaxTree> GenerateFixtures()
    {
        // Discover the actual production models from metadata. Run the generator's normal test
        // entry point, then retain exactly this bounded cohort for subsequent runtime compilation.
        var compilation = CSharpCompilation.Create("AiDotNetTests",
            new[] { CSharpSyntaxTree.ParseText("namespace AiDotNet.Tests { internal sealed class PositiveFixtureMarker {} }") },
            References(includeTestAssembly: false), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new TestScaffoldGenerator().AsSourceGenerator());
        var run = driver.RunGenerators(compilation).GetRunResult();
        var generatorResult = Assert.Single(run.Results);
        Assert.Null(generatorResult.Exception);
        // This focused discovery compilation omits the rest of the repository's manual test
        // census. Its global coverage/name-collision diagnostics are not whole-project proof.
        // Require exact cohort identities here; compile every retained source against real bases.
        Console.WriteLine("Metadata-census diagnostics (selected-fixture compilation is checked separately): "
            + string.Join(", ", run.Diagnostics.GroupBy(diagnostic => (diagnostic.Severity, diagnostic.Id))
                .Select(group => $"{group.Key.Severity}/{group.Key.Id}={group.Count()}")));
        var generated = generatorResult.GeneratedSources;
        var fixtures = new Dictionary<TextDetectorKind, SyntaxTree>();
        foreach (TextDetectorKind kind in Enum.GetValues(typeof(TextDetectorKind)))
        {
            var type = ModelType(kind).GetGenericTypeDefinition();
            string metadataName = type.FullName ?? throw new InvalidOperationException("The detector has no metadata name.");
            string hintName = metadataName.Split('`')[0].Replace('.', '_') + "Tests.g.cs";
            var fixture = Assert.Single(generated, item => item.HintName == hintName);
            fixtures.Add(kind, fixture.SyntaxTree);
        }
        Assert.Equal(3, fixtures.Count);
        return fixtures;
    }

    private static IEnumerable<MetadataReference> References(bool includeTestAssembly)
    {
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string trusted)
            paths.UnionWith(trusted.Split(Path.PathSeparator));
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location)) paths.Add(assembly.Location);
        paths.Add(typeof(ModelBase<,,>).Assembly.Location);
        paths.Add(typeof(Tensor<>).Assembly.Location);
        if (!includeTestAssembly) paths.Remove(typeof(GeneratedTextDetectionPositiveFixtureTests).Assembly.Location);
        return paths.Select(path => MetadataReference.CreateFromFile(path));
    }

    private static Type ModelType(TextDetectorKind kind) => kind switch
    {
        TextDetectorKind.Craft => typeof(CRAFT<double>),
        TextDetectorKind.DifferentiableBinarization => typeof(DBNet<double>),
        TextDetectorKind.East => typeof(EAST<double>),
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };
}
