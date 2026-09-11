using System.Reflection;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.Generators;
using AiDotNet.Models;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;
using ExpectedDetection = AiDotNet.Tests.ModelFamilyTests.Base.ObjectDetectionPositiveFixture<double>.ExpectedDetection;

namespace AiDotNet.Tests.Generators;

/// <summary>Compiles the real generator's nine object fixtures against actual detector implementations.</summary>
public sealed class GeneratedObjectDetectionPositiveFixtureTests
{
    public enum DetectorKind { Yolo8, Yolo9, Yolo10, Yolo11, Detr, RtDetr, Dino, FasterRcnn, CascadeRcnn }
    public enum ResultCorruption { Empty, MissingCandidate, ReverseOrder, EqualScores, WrongClass, InvertedBox, NonFiniteBox, NonFiniteScore, WrongScore, WrongGeometry, WrongImageSize }
    public enum SuppressionCorruption { Ignored, LowerScoreWinner, OverSuppressed }
    public enum MissingPrecondition { Empty, SingleCandidate, EqualScores, DisjointBoxes }
    public enum DetectorMutation { EmptyResult, ReverseOrder, IgnoreSuppression }

    private static readonly Lazy<IReadOnlyDictionary<DetectorKind, SyntaxTree>> GeneratedFixtures = new(GenerateFixtures);
    private static readonly Lazy<Assembly> RuntimeFixtures = new(CompileFixtures);

    public GeneratedObjectDetectionPositiveFixtureTests() => TestModuleInitializer.EnsureInitialized();
    public static IEnumerable<object[]> Models => Enum.GetValues(typeof(DetectorKind)).Cast<DetectorKind>().Select(kind => new object[] { kind });

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Models))]
    public async Task GeneratedPositiveFactory_UsesTheTypedOptionsConstructor(DetectorKind kind)
    {
        await Task.Yield();
        var declaration = Assert.Single(GeneratedFixtures.Value[kind].GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>());
        var factory = Assert.Single(declaration.Members.OfType<MethodDeclarationSyntax>(),
            method => method.Identifier.ValueText == "CreatePositiveObjectDetector");
        Assert.Equal("ObjectDetectionOptions<double>", Assert.Single(factory.ParameterList.Parameters).Type?.ToString().Split('.').Last());
        var construction = Assert.Single(factory.DescendantNodes().OfType<ObjectCreationExpressionSyntax>());
        Assert.Equal(ModelType(kind).Name.Split('`')[0] + "<double>", construction.Type.ToString().Split('.').Last());
        Assert.Equal("options", Assert.Single(construction.ArgumentList?.Arguments ?? default).ToString());
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Models))]
    public async Task GeneratedPositiveInvariant_RunsActualForwardDecodeOrderingAndNms(DetectorKind kind)
    {
        await Task.Yield();
        await CreateFixture(kind).Detect_ControlledPositiveHead_ShouldDecodeRankAndSuppressKnownCandidates();
    }

    [Theory(Timeout = 180000)]
    [MemberData(nameof(Models))]
    public async Task GeneratedDefaultFixture_PreservesExistingGeometryOrderingAndNmsInvariants(DetectorKind kind)
    {
        await Task.Yield();
        var fixture = CreateFixture(kind);
        await fixture.Detect_ShouldProduceGeometricallyValidBoxes();
        await fixture.Detect_ScoresShouldBeDescending();
        await fixture.Detect_SurvivorsShouldNotOverlapAboveTheNmsThreshold();
    }

    [Fact]
    public void PositiveOracle_AcceptsKnownCandidatesAndIndependentSuppression()
    {
        var expected = KnownCandidates();
        ObjectDetectionPositiveFixture<double>.AssertCandidatePreconditions(expected);
        ObjectDetectionPositiveFixture<double>.AssertMatches(Result(expected), expected);
        var suppressed = ObjectDetectionPositiveFixture<double>.SuppressIndependently(expected, 0.45);
        Assert.Equal(new[] { 0.9, 0.4 }, suppressed.Select(candidate => candidate.Score));
        ObjectDetectionPositiveFixture<double>.AssertMatches(Result(suppressed), suppressed);
    }

    [Theory(Timeout = 120000)]
    [InlineData(DetectorMutation.EmptyResult)]
    [InlineData(DetectorMutation.ReverseOrder)]
    [InlineData(DetectorMutation.IgnoreSuppression)]
    public async Task ActualPositiveInvariant_RejectsBrokenDetectionImplementations(DetectorMutation mutation)
    {
        await Task.Yield();
        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
        using var detector = new MutatedYoloDetection(mutation);
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => ObjectDetectionPositiveFixture<double>.Verify(detector));
        // A failure in construction, live state, or the independent raw oracle is not evidence
        // against this mutant. Prove the actual Detect path reached the requested corruption.
        Assert.True(detector.MutationApplied);
    }

    [Theory]
    [InlineData(ResultCorruption.Empty)]
    [InlineData(ResultCorruption.MissingCandidate)]
    [InlineData(ResultCorruption.ReverseOrder)]
    [InlineData(ResultCorruption.EqualScores)]
    [InlineData(ResultCorruption.WrongClass)]
    [InlineData(ResultCorruption.InvertedBox)]
    [InlineData(ResultCorruption.NonFiniteBox)]
    [InlineData(ResultCorruption.NonFiniteScore)]
    [InlineData(ResultCorruption.WrongScore)]
    [InlineData(ResultCorruption.WrongGeometry)]
    [InlineData(ResultCorruption.WrongImageSize)]
    public void PositiveOracle_RejectsEmptyMalformedAndMisorderedResults(ResultCorruption corruption)
    {
        var expected = KnownCandidates();
        var result = Result(expected);
        var first = result.Detections[0];
        switch (corruption)
        {
            case ResultCorruption.Empty: result.Detections.Clear(); break;
            case ResultCorruption.MissingCandidate: result.Detections.RemoveAt(1); break;
            case ResultCorruption.ReverseOrder: result.Detections.Reverse(); break;
            case ResultCorruption.EqualScores:
                foreach (var detection in result.Detections) detection.Confidence = 0.9;
                break;
            case ResultCorruption.WrongClass: first.ClassId = 1; break;
            case ResultCorruption.InvertedBox: first.Box = new BoundingBox<double>(32, 0, 0, 32); break;
            case ResultCorruption.NonFiniteBox: first.Box = new BoundingBox<double>(0, 0, double.NaN, 32); break;
            case ResultCorruption.NonFiniteScore: first.Confidence = double.NaN; break;
            case ResultCorruption.WrongScore: first.Confidence = 0.8; break;
            case ResultCorruption.WrongGeometry: first.Box = new BoundingBox<double>(0, 0, 31, 32); break;
            case ResultCorruption.WrongImageSize: result.ImageWidth = 63; break;
            default: throw new ArgumentOutOfRangeException(nameof(corruption));
        }
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => ObjectDetectionPositiveFixture<double>.AssertMatches(result, expected));
    }

    [Theory]
    [InlineData(SuppressionCorruption.Ignored)]
    [InlineData(SuppressionCorruption.LowerScoreWinner)]
    [InlineData(SuppressionCorruption.OverSuppressed)]
    public void PositiveOracle_RejectsWrongSuppressionEvenWhenResultsRemainNonempty(SuppressionCorruption corruption)
    {
        var candidates = KnownCandidates();
        var expected = ObjectDetectionPositiveFixture<double>.SuppressIndependently(candidates, 0.45);
        var actual = corruption switch
        {
            SuppressionCorruption.Ignored => candidates,
            SuppressionCorruption.LowerScoreWinner => new[] { candidates[1], candidates[2] },
            SuppressionCorruption.OverSuppressed => new[] { candidates[0] },
            _ => throw new ArgumentOutOfRangeException(nameof(corruption))
        };
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => ObjectDetectionPositiveFixture<double>.AssertMatches(Result(actual), expected));
    }

    [Theory]
    [InlineData(MissingPrecondition.Empty)]
    [InlineData(MissingPrecondition.SingleCandidate)]
    [InlineData(MissingPrecondition.EqualScores)]
    [InlineData(MissingPrecondition.DisjointBoxes)]
    public void PositiveFixture_RejectsVacuousOrderingOrSuppressionPreconditions(MissingPrecondition missing)
    {
        var candidates = missing switch
        {
            MissingPrecondition.Empty => Array.Empty<ExpectedDetection>(),
            MissingPrecondition.SingleCandidate => new[] { new ExpectedDetection(0.9, 0, 0, 32, 32) },
            MissingPrecondition.EqualScores => new[] { new ExpectedDetection(0.9, 0, 0, 32, 32), new ExpectedDetection(0.9, 0, 0, 32, 32) },
            MissingPrecondition.DisjointBoxes => new[] { new ExpectedDetection(0.9, 0, 0, 10, 10), new ExpectedDetection(0.7, 20, 20, 30, 30) },
            _ => throw new ArgumentOutOfRangeException(nameof(missing))
        };
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => ObjectDetectionPositiveFixture<double>.AssertCandidatePreconditions(candidates));
    }

    private static ExpectedDetection[] KnownCandidates() => new[]
    {
        new ExpectedDetection(0.9, 0, 0, 32, 32),
        new ExpectedDetection(0.7, 0, 0, 32, 32),
        new ExpectedDetection(0.4, 40, 40, 60, 60)
    };

    private static DetectionResult<double> Result(IEnumerable<ExpectedDetection> expected) => new()
    {
        ImageWidth = 64, ImageHeight = 64,
        Detections = expected.Select(candidate => new Detection<double>(new BoundingBox<double>(candidate.Left,
            candidate.Top, candidate.Right, candidate.Bottom), 0, candidate.Score)).ToList()
    };

    private static ObjectDetectionTestBase CreateFixture(DetectorKind kind)
    {
        string typeName = "AiDotNet.Tests.ModelFamilyTests.Generated." + ModelType(kind).Name.Split('`')[0] + "Tests";
        var fixtureType = RuntimeFixtures.Value.GetType(typeName)
            ?? throw new InvalidOperationException("The exact generated object fixture was not compiled.");
        return Assert.IsAssignableFrom<ObjectDetectionTestBase>(Activator.CreateInstance(fixtureType));
    }

    private static Assembly CompileFixtures()
    {
        var trees = GeneratedFixtures.Value.Values.Concat(new[]
        {
            CSharpSyntaxTree.ParseText("global using System; global using System.Linq; global using System.Collections.Generic; global using AiDotNet.Tensors.LinearAlgebra;")
        });
        var compilation = CSharpCompilation.Create("GeneratedObjectDetectionReview_" + Guid.NewGuid().ToString("N"),
            trees, References(includeTestAssembly: true), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var stream = new MemoryStream();
        var emit = compilation.Emit(stream);
        Assert.True(emit.Success, string.Join(Environment.NewLine,
            emit.Diagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)));
        return Assembly.Load(stream.ToArray());
    }

    private static IReadOnlyDictionary<DetectorKind, SyntaxTree> GenerateFixtures()
    {
        var compilation = CSharpCompilation.Create("AiDotNetTests",
            new[] { CSharpSyntaxTree.ParseText("namespace AiDotNet.Tests { internal sealed class PositiveFixtureMarker {} }") },
            References(includeTestAssembly: false), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new TestScaffoldGenerator().AsSourceGenerator());
        var run = driver.RunGenerators(compilation).GetRunResult();
        var generatorResult = Assert.Single(run.Results);
        Assert.Null(generatorResult.Exception);
        // Discovery omits the repository-wide manual test census. Global diagnostics are
        // reported, not presented as whole-project proof. Every selected real fixture must compile.
        Console.WriteLine("Metadata-census diagnostics (selected-fixture compilation is checked separately): "
            + string.Join(", ", run.Diagnostics.GroupBy(diagnostic => (diagnostic.Severity, diagnostic.Id))
                .Select(group => $"{group.Key.Severity}/{group.Key.Id}={group.Count()}")));
        var fixtures = new Dictionary<DetectorKind, SyntaxTree>();
        foreach (DetectorKind kind in Enum.GetValues(typeof(DetectorKind)))
        {
            string metadataName = ModelType(kind).GetGenericTypeDefinition().FullName
                ?? throw new InvalidOperationException("The detector has no metadata name.");
            string hintName = metadataName.Split('`')[0].Replace('.', '_') + "Tests.g.cs";
            var fixture = Assert.Single(generatorResult.GeneratedSources, item => item.HintName == hintName);
            fixtures.Add(kind, fixture.SyntaxTree);
        }
        Assert.Equal(9, fixtures.Count);
        return fixtures;
    }

    private static IEnumerable<MetadataReference> References(bool includeTestAssembly)
    {
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string trusted) paths.UnionWith(trusted.Split(Path.PathSeparator));
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location)) paths.Add(assembly.Location);
        paths.Add(typeof(ModelBase<,,>).Assembly.Location);
        paths.Add(typeof(Tensor<>).Assembly.Location);
        if (!includeTestAssembly) paths.Remove(typeof(GeneratedObjectDetectionPositiveFixtureTests).Assembly.Location);
        return paths.Select(path => MetadataReference.CreateFromFile(path));
    }

    private static Type ModelType(DetectorKind kind) => kind switch
    {
        DetectorKind.Yolo8 => typeof(YOLOv8<double>), DetectorKind.Yolo9 => typeof(YOLOv9<double>),
        DetectorKind.Yolo10 => typeof(YOLOv10<double>), DetectorKind.Yolo11 => typeof(YOLOv11<double>),
        DetectorKind.Detr => typeof(DETR<double>), DetectorKind.RtDetr => typeof(RTDETR<double>),
        DetectorKind.Dino => typeof(DINO<double>), DetectorKind.FasterRcnn => typeof(FasterRCNN<double>),
        DetectorKind.CascadeRcnn => typeof(CascadeRCNN<double>),
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };

    /// <summary>Negative control only; the real inherited numerical forward is never overridden.</summary>
    private sealed class MutatedYoloDetection : YOLOv8<double>
    {
        private readonly DetectorMutation _mutation;
        public bool MutationApplied { get; private set; }
        public MutatedYoloDetection(DetectorMutation mutation) : base(ObjectDetectionPositiveFixture<double>.CreateOptions())
            => _mutation = mutation;

        public override DetectionResult<double> Detect(Tensor<double> image, double confidenceThreshold = 0.25, double nmsThreshold = 0.45)
        {
            var result = base.Detect(image, confidenceThreshold,
                _mutation == DetectorMutation.IgnoreSuppression ? 1.0 : nmsThreshold);
            switch (_mutation)
            {
                case DetectorMutation.EmptyResult: result.Detections.Clear(); MutationApplied = true; break;
                case DetectorMutation.ReverseOrder: result.Detections.Reverse(); MutationApplied = true; break;
                case DetectorMutation.IgnoreSuppression: MutationApplied |= nmsThreshold < 1.0; break;
                default: throw new ArgumentOutOfRangeException(nameof(_mutation));
            }
            return result;
        }
    }
}
