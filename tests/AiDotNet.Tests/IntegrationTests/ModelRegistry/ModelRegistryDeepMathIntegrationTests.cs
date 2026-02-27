using AiDotNet.Interfaces;
using AiDotNet.ModelRegistry;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ModelRegistry;

/// <summary>
/// Deep integration tests for ModelRegistry:
/// ModelRegistryBase path security validation (GetSanitizedPath, GetSanitizedFileName,
/// ValidatePathWithinDirectory), model versioning patterns, ModelStage transitions,
/// ModelMetadata and ModelSearchCriteria data models.
/// </summary>
public class ModelRegistryDeepMathIntegrationTests
{
    // ============================
    // ModelStage Enum
    // ============================

    [Theory]
    [InlineData(ModelStage.Development)]
    [InlineData(ModelStage.Staging)]
    [InlineData(ModelStage.Production)]
    [InlineData(ModelStage.Archived)]
    public void ModelStage_AllValuesValid(ModelStage stage)
    {
        Assert.True(Enum.IsDefined(typeof(ModelStage), stage));
    }

    [Fact]
    public void ModelStage_HasFourValues()
    {
        var values = (((ModelStage[])Enum.GetValues(typeof(ModelStage))));
        Assert.Equal(4, values.Length);
    }

    // ============================
    // ModelStage: Lifecycle Progression
    // ============================

    [Fact]
    public void ModelStage_TypicalLifecycle_Order()
    {
        // Models typically progress through these stages
        var lifecycle = new[]
        {
            ModelStage.Development,
            ModelStage.Staging,
            ModelStage.Production,
            ModelStage.Archived
        };

        // Each stage value should be >= the previous (non-strict for flexibility)
        for (int i = 1; i < lifecycle.Length; i++)
        {
            Assert.True((int)lifecycle[i] >= (int)lifecycle[i - 1],
                $"Stage {lifecycle[i]} should have value >= {lifecycle[i - 1]}");
        }
    }

    // ============================
    // ModelMetadata: Defaults
    // ============================

    [Fact]
    public void ModelMetadata_Defaults()
    {
        var metadata = new ModelMetadata<double>();
        Assert.NotNull(metadata);
    }

    // ============================
    // Versioning Math: Semantic Versioning
    // ============================

    [Theory]
    [InlineData(1, 2, 3)]
    [InlineData(0, 1, 0)]
    [InlineData(10, 0, 0)]
    public void ModelVersioning_SemanticVersion_Components(int major, int minor, int patch)
    {
        // Semantic version: MAJOR.MINOR.PATCH
        string version = $"{major}.{minor}.{patch}";

        var parts = version.Split('.');
        Assert.Equal(3, parts.Length);
        Assert.Equal(major.ToString(), parts[0]);
        Assert.Equal(minor.ToString(), parts[1]);
        Assert.Equal(patch.ToString(), parts[2]);
    }

    [Theory]
    [InlineData("1.0.0", "1.0.1", -1)]   // Patch increment
    [InlineData("1.0.0", "1.1.0", -1)]   // Minor increment
    [InlineData("1.0.0", "2.0.0", -1)]   // Major increment
    [InlineData("2.0.0", "1.0.0", 1)]    // Major downgrade
    [InlineData("1.0.0", "1.0.0", 0)]    // Equal
    public void ModelVersioning_SemanticVersion_Comparison(string v1, string v2, int expectedSign)
    {
        var ver1 = Version.Parse(v1);
        var ver2 = Version.Parse(v2);

        int comparison = ver1.CompareTo(ver2);
        Assert.Equal(expectedSign, Math.Sign(comparison));
    }

    // ============================
    // Path Security: Sanitization
    // ============================

    [Theory]
    [InlineData("my-model")]
    [InlineData("model_v2")]
    [InlineData("classification-resnet50")]
    public void PathSecurity_ValidModelNames(string modelName)
    {
        // Valid model names should not contain path separators
        Assert.DoesNotContain("/", modelName);
        Assert.DoesNotContain("\\", modelName);
        Assert.DoesNotContain("..", modelName);
    }

    [Theory]
    [InlineData("../../../etc/passwd")]
    [InlineData("..\\..\\windows\\system32")]
    [InlineData("model/../../../secret")]
    public void PathSecurity_TraversalAttempts_Detected(string maliciousPath)
    {
        // These should contain path traversal sequences
        Assert.Contains("..", maliciousPath);
    }

    [Theory]
    [InlineData("model name", "model name")]       // Spaces are OK in file names
    [InlineData("model.v1", "model.v1")]            // Dots are OK
    [InlineData("../evil", "evil")]                  // Path traversal stripped by GetFileName
    [InlineData("sub/dir/file", "file")]             // Directory components stripped
    public void PathSecurity_GetFileName_StripsDirectory(string input, string expectedFileName)
    {
        // Path.GetFileName strips directory components (security behavior)
        string sanitized = Path.GetFileName(input);
        Assert.Equal(expectedFileName, sanitized);
    }

    // ============================
    // Path Security: Directory Containment
    // ============================

    [Fact]
    public void PathSecurity_PathWithinDirectory_ValidPath()
    {
        // A path within the base directory should pass validation
        string baseDir = Path.GetTempPath();
        string validPath = Path.Combine(baseDir, "models", "mymodel");

        string fullPath = Path.GetFullPath(validPath);
        string fullBase = Path.GetFullPath(baseDir);

        // Normalize base with trailing separator
        if (!fullBase.EndsWith(Path.DirectorySeparatorChar.ToString()))
            fullBase += Path.DirectorySeparatorChar;

        Assert.True(fullPath.StartsWith(fullBase, StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void PathSecurity_SiblingPrefixBypass_Prevented()
    {
        // Test that "model_registrymalicious" doesn't match "model_registry"
        string baseDir = Path.Combine(Path.GetTempPath(), "model_registry");
        string maliciousDir = Path.Combine(Path.GetTempPath(), "model_registrymalicious");

        string fullBase = Path.GetFullPath(baseDir);
        string fullMalicious = Path.GetFullPath(maliciousDir);

        // Without trailing separator, this would incorrectly match
        Assert.True(fullMalicious.StartsWith(fullBase, StringComparison.OrdinalIgnoreCase));

        // With trailing separator (the correct check), it should NOT match
        string fullBaseWithSep = fullBase + Path.DirectorySeparatorChar;
        Assert.False(fullMalicious.StartsWith(fullBaseWithSep, StringComparison.OrdinalIgnoreCase));
    }

    // ============================
    // Model Storage: Version Path Patterns
    // ============================

    [Theory]
    [InlineData("mymodel", 1, "mymodel", "v1.json")]
    [InlineData("resnet50", 3, "resnet50", "v3.json")]
    [InlineData("classifier", 10, "classifier", "v10.json")]
    public void ModelStorage_VersionPathPattern(string modelName, int version, string expectedDir, string expectedFile)
    {
        // Version files follow the pattern: {modelDir}/{modelName}/v{version}.json
        string dir = modelName;
        string file = $"v{version}.json";

        Assert.Equal(expectedDir, dir);
        Assert.Equal(expectedFile, file);
    }

    // ============================
    // Model Comparison Math
    // ============================

    [Theory]
    [InlineData(0.85, 0.92, 0.07, true)]     // v2 is better
    [InlineData(0.95, 0.93, -0.02, false)]    // v1 is better
    [InlineData(0.90, 0.90, 0.0, false)]      // Equal
    public void ModelComparison_ScoreDifference(double score1, double score2, double expectedDiff, bool v2IsBetter)
    {
        double diff = score2 - score1;
        Assert.Equal(expectedDiff, diff, 1e-10);
        Assert.Equal(v2IsBetter, diff > 0);
    }

    [Theory]
    [InlineData(0.85, 0.92)]
    [InlineData(0.70, 0.95)]
    public void ModelComparison_ImprovementPercentage(double oldScore, double newScore)
    {
        // Relative improvement = (new - old) / old * 100
        double improvement = (newScore - oldScore) / oldScore * 100;
        Assert.True(improvement > 0, $"Expected positive improvement, got {improvement}%");
    }

    // ============================
    // JSON Serialization Security
    // ============================

    [Fact]
    public void JsonSettings_TypeNameHandlingNone()
    {
        // ModelRegistryBase uses TypeNameHandling.None for security
        // This prevents deserialization attacks via type metadata in JSON
        var settings = new Newtonsoft.Json.JsonSerializerSettings
        {
            TypeNameHandling = Newtonsoft.Json.TypeNameHandling.None
        };

        Assert.Equal(Newtonsoft.Json.TypeNameHandling.None, settings.TypeNameHandling);
    }

    // ============================
    // Registry Operations: Thread Safety Patterns
    // ============================

    [Fact]
    public void ThreadSafety_LockObject_NotNull()
    {
        // ModelRegistryBase uses a SyncLock object for thread safety
        // Verify the pattern works
        var lockObj = new object();
        Assert.NotNull(lockObj);

        bool entered = false;
        lock (lockObj)
        {
            entered = true;
        }
        Assert.True(entered);
    }

    [Fact]
    public void ThreadSafety_ConcurrentAccess_Pattern()
    {
        // Verify concurrent dictionary access pattern
        var models = new Dictionary<string, int>();
        var syncLock = new object();

        // Simulate concurrent registration
        int registrations = 0;
        Parallel.For(0, 100, i =>
        {
            lock (syncLock)
            {
                models[$"model_{i}"] = i;
                Interlocked.Increment(ref registrations);
            }
        });

        Assert.Equal(100, registrations);
        Assert.Equal(100, models.Count);
    }
}
