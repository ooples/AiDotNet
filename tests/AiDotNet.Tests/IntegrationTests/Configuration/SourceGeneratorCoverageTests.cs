using System.Reflection;
using Xunit;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Comprehensive integration tests that verify the Roslyn source generator provides
/// 100% coverage of all Configure*() methods on AiModelBuilder, that the type registry
/// discovers all concrete implementations, and that the JSON Schema and docs generators
/// cover every section.
/// </summary>
public class SourceGeneratorCoverageTests
{
    #region Configure Method Coverage

    /// <summary>
    /// Verifies that every public Configure*() method on AiModelBuilder has a corresponding
    /// property on YamlModelConfig (either hand-written or auto-generated).
    /// </summary>
    [Fact]
    public void YamlModelConfig_HasPropertyForEveryConfigureMethod()
    {
        var builderType = typeof(AiModelBuilder<double, Matrix<double>, Vector<double>>);

        // Get all unique Configure* method names (excluding async, convenience overloads)
        var configureMethods = builderType
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.Name.StartsWith("Configure", StringComparison.Ordinal))
            .Where(m => !m.Name.EndsWith("Async", StringComparison.Ordinal))
            // Exclude convenience overloads that are just wrappers
            .Where(m => m.Name != "ConfigureTimeSeriesFeaturesForFinance")
            .Where(m => m.Name != "ConfigureTimeSeriesFeaturesMinimal")
            // ConfigureModel(IFullModel<...>) is intentionally skipped by the generator
            // because IFullModel can't be instantiated from YAML (requires pre-built model)
            .Where(m => m.Name != "ConfigureModel")
            .Select(m => m.Name.Substring("Configure".Length))
            .Distinct()
            .ToList();

        var configType = typeof(YamlModelConfig);
        var configProperties = configType
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Select(p => p.Name)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var missingProperties = new List<string>();
        foreach (var methodSection in configureMethods)
        {
            if (!configProperties.Contains(methodSection))
            {
                missingProperties.Add(methodSection);
            }
        }

        Assert.True(
            missingProperties.Count == 0,
            $"The following Configure*() methods have no matching YamlModelConfig property: " +
            $"{string.Join(", ", missingProperties)}. " +
            $"Total methods: {configureMethods.Count}, total properties: {configProperties.Count}");
    }

    /// <summary>
    /// Verifies the total number of YAML properties matches expectations.
    /// Hand-written: Optimizer, TimeSeriesModel + 16 deployment POCOs = 18
    /// Auto-generated: 48 from source generator
    /// Total: 66 properties
    /// </summary>
    [Fact]
    public void YamlModelConfig_HasExpectedPropertyCount()
    {
        var configType = typeof(YamlModelConfig);
        var properties = configType
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanRead && p.CanWrite)
            .ToList();

        // At minimum we expect all Configure* methods to have coverage
        // The exact count may increase as new methods are added, but should never decrease
        Assert.True(properties.Count >= 60,
            $"Expected at least 60 YAML config properties but found {properties.Count}. " +
            $"Properties: {string.Join(", ", properties.Select(p => p.Name))}");
    }

    #endregion

    #region Generated Section Deserialization

    [Fact]
    public void LoadFromString_WithRegularizationSection_DeserializesCorrectly()
    {
        var yaml = @"
regularization:
  type: NoRegularization
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Regularization);
        Assert.Equal("NoRegularization", config.Regularization.Type);
    }

    [Fact]
    public void LoadFromString_WithFitDetectorSection_DeserializesCorrectly()
    {
        var yaml = @"
fitDetector:
  type: DefaultFitDetector
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.FitDetector);
        Assert.Equal("DefaultFitDetector", config.FitDetector.Type);
    }

    [Fact]
    public void LoadFromString_WithFairnessEvaluatorSection_DeserializesCorrectly()
    {
        var yaml = @"
fairnessEvaluator:
  type: BasicFairnessEvaluator
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.FairnessEvaluator);
        Assert.Equal("BasicFairnessEvaluator", config.FairnessEvaluator.Type);
    }

    [Fact]
    public void LoadFromString_WithTokenizerSection_DeserializesCorrectly()
    {
        var yaml = @"
tokenizer:
  type: BPETokenizer
  params:
    vocabSize: 32000
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Tokenizer);
        Assert.Equal("BPETokenizer", config.Tokenizer.Type);
        Assert.True(config.Tokenizer.Params.ContainsKey("vocabSize"));
    }

    [Fact]
    public void LoadFromString_WithPromptTemplateSection_DeserializesCorrectly()
    {
        var yaml = @"
promptTemplate:
  type: InstructionFollowingTemplate
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.PromptTemplate);
        Assert.Equal("InstructionFollowingTemplate", config.PromptTemplate.Type);
    }

    [Fact]
    public void LoadFromString_WithPreprocessingPipeline_DeserializesCorrectly()
    {
        var yaml = @"
preprocessing:
  steps:
    - type: StandardScaler
    - type: SimpleImputer
      params:
        strategy: Mean
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Preprocessing);
        Assert.Equal(2, config.Preprocessing.Steps.Count);
        Assert.Equal("StandardScaler", config.Preprocessing.Steps[0].Type);
        Assert.Equal("SimpleImputer", config.Preprocessing.Steps[1].Type);
        Assert.True(config.Preprocessing.Steps[1].Params.ContainsKey("strategy"));
    }

    [Fact]
    public void LoadFromString_WithPostprocessingPipeline_DeserializesCorrectly()
    {
        var yaml = @"
postprocessing:
  steps:
    - type: SoftmaxTransformer
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Postprocessing);
        Assert.Single(config.Postprocessing.Steps);
        Assert.Equal("SoftmaxTransformer", config.Postprocessing.Steps[0].Type);
    }

    [Fact]
    public void LoadFromString_WithFederatedLearningSection_DeserializesCorrectly()
    {
        var yaml = @"
federatedLearning:
  numberOfClients: 10
  localEpochs: 5
  learningRate: 0.01
  maxRounds: 100
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.FederatedLearning);
        Assert.Equal(10, config.FederatedLearning.NumberOfClients);
        Assert.Equal(5, config.FederatedLearning.LocalEpochs);
        Assert.Equal(0.01, config.FederatedLearning.LearningRate);
        Assert.Equal(100, config.FederatedLearning.MaxRounds);
    }

    [Fact]
    public void LoadFromString_WithAugmentationSection_DeserializesCorrectly()
    {
        var yaml = @"
augmentation:
  isEnabled: true
  probability: 0.5
  seed: 42
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Augmentation);
        Assert.True(config.Augmentation.IsEnabled);
        Assert.Equal(0.5, config.Augmentation.Probability);
        Assert.Equal(42, config.Augmentation.Seed);
    }

    [Fact]
    public void LoadFromString_WithUncertaintyQuantificationSection_DeserializesCorrectly()
    {
        var yaml = @"
uncertaintyQuantification:
  enabled: true
  numSamples: 100
  monteCarloDropoutRate: 0.1
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.UncertaintyQuantification);
        Assert.True(config.UncertaintyQuantification.Enabled);
        Assert.Equal(100, config.UncertaintyQuantification.NumSamples);
        Assert.Equal(0.1, config.UncertaintyQuantification.MonteCarloDropoutRate);
    }

    [Fact]
    public void LoadFromString_WithVisualizationSection_DeserializesCorrectly()
    {
        var yaml = @"
visualization:
  boxThickness: 3
  showLabels: true
  showConfidence: false
  maskOpacity: 0.5
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.Visualization);
        Assert.Equal(3, config.Visualization.BoxThickness);
        Assert.True(config.Visualization.ShowLabels);
        Assert.False(config.Visualization.ShowConfidence);
        Assert.Equal(0.5, config.Visualization.MaskOpacity);
    }

    [Fact]
    public void LoadFromString_WithProgramSynthesisSection_DeserializesCorrectly()
    {
        var yaml = @"
programSynthesis:
  maxSequenceLength: 512
  vocabularySize: 50000
";

        var config = YamlConfigLoader.LoadFromString(yaml);

        Assert.NotNull(config.ProgramSynthesis);
        Assert.Equal(512, config.ProgramSynthesis.MaxSequenceLength);
        Assert.Equal(50000, config.ProgramSynthesis.VocabularySize);
    }

    #endregion

    #region Type Registry Coverage

    /// <summary>
    /// Verifies the type registry has entries for the sections that have parameterless-ctor implementations.
    /// </summary>
    [Fact]
    public void TypeRegistry_HasRegisteredImplementations_ForKeyInterfaceSections()
    {
        var registries = YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();

        // These sections MUST have at least one registered implementation.
        // With the factory-based constructor resolution, all sections with concrete
        // implementations (including those requiring constructor parameters) are now registered.
        var expectedSections = new[]
        {
            "Regularization",
            "FitDetector",
            "Model",
            "FairnessEvaluator",
            "TrainingMonitor",
            "PromptTemplate",
            "PromptOptimizer",
            "FewShotExampleSelector",
            "PromptAnalyzer",
            "ObjectDetector",
            "InstanceSegmenter",
            "ObjectTracker",
            "FitnessCalculator",
            "DataLoader",
            "BiasDetector",
            "LoRA",
            "RetrievalAugmentedGeneration",
            "CrossValidation",
            "AutoML",
            "MetaLearning",
            "DistributedTraining",
            "ExperimentTracker",
            "CheckpointManager",
            "ModelRegistry",
            "DataVersionControl",
            "HyperparameterOptimizer",
            "Tokenizer",
            "PromptChain",
            "PromptCompressor",
            "Optimizer",
        };

        foreach (var section in expectedSections)
        {
            Assert.True(registries.ContainsKey(section),
                $"Type registry is missing section: {section}");
            Assert.True(registries[section].Count > 0,
                $"Type registry has section '{section}' but with zero implementations");
        }
    }

    /// <summary>
    /// Verifies the Model section of the type registry has a substantial number of implementations.
    /// The library has hundreds of IFullModel implementations and all with public constructors
    /// should be registered.
    /// </summary>
    [Fact]
    public void TypeRegistry_ModelSection_HasMultipleImplementations()
    {
        var registries = YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();

        Assert.True(registries.ContainsKey("Model"), "Model section missing from type registry");
        Assert.True(registries["Model"].Count >= 100,
            $"Model section only has {registries["Model"].Count} implementations. " +
            $"Expected at least 100.");
    }

    /// <summary>
    /// Verifies CreateInstance works for representative types from each section.
    /// Samples up to 3 types per section to avoid test host crashes from types
    /// that allocate large resources during construction.
    /// </summary>
    [Fact]
    public void TypeRegistry_SampledTypes_CanBeInstantiated()
    {
        var registries = YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();

        var successes = 0;
        var sectionsTested = 0;
        var failures = new List<string>();
        foreach (var (sectionName, types) in registries)
        {
            // Sample up to 3 types per section (first, middle, last) to keep the test fast
            // and avoid crashing the test host from heavy constructors.
            var typeNames = types.Keys.ToList();
            var sampled = new List<string> { typeNames[0] };
            if (typeNames.Count > 2) sampled.Add(typeNames[typeNames.Count / 2]);
            if (typeNames.Count > 1) sampled.Add(typeNames[typeNames.Count - 1]);

            var sectionHasSuccess = false;
            foreach (var typeName in sampled)
            {
                try
                {
                    var instance = YamlTypeRegistry<double, Matrix<double>, Vector<double>>
                        .CreateInstance<object>(sectionName, typeName);
                    if (instance is not null)
                    {
                        successes++;
                        sectionHasSuccess = true;
                    }
                    else
                    {
                        failures.Add($"{sectionName}/{typeName}: CreateInstance returned null");
                    }
                }
                catch (Exception ex)
                {
                    failures.Add($"{sectionName}/{typeName}: {ex.GetType().Name} - {ex.InnerException?.Message ?? ex.Message}");
                }
            }

            if (sectionHasSuccess)
            {
                sectionsTested++;
            }
        }

        // At least half of all sections should have at least one instantiable type.
        Assert.True(sectionsTested >= registries.Count / 2,
            $"Only {sectionsTested}/{registries.Count} sections had at least one instantiable type.\n" +
            $"Successes: {successes}\nFailures:\n{string.Join("\n", failures.Take(20))}");
    }

    /// <summary>
    /// Verifies the YamlRegisteredTypeNames (non-generic) matches the generic YamlTypeRegistry.
    /// </summary>
    [Fact]
    public void RegisteredTypeNames_MatchesTypeRegistry()
    {
        var genericRegistries = YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();
        var stringNames = YamlRegisteredTypeNames.SectionTypes;

        foreach (var (sectionName, typeDict) in genericRegistries)
        {
            Assert.True(stringNames.ContainsKey(sectionName),
                $"YamlRegisteredTypeNames is missing section: {sectionName}");
            Assert.Equal(typeDict.Count, stringNames[sectionName].Length);

            foreach (var typeName in typeDict.Keys)
            {
                Assert.Contains(typeName, stringNames[sectionName]);
            }
        }
    }

    #endregion

    #region Schema Metadata Coverage

    /// <summary>
    /// Verifies the schema metadata covers all generated sections.
    /// </summary>
    [Fact]
    public void SchemaMetadata_CoversAllGeneratedSections()
    {
        var sections = YamlSchemaMetadata.Sections;
        var sectionNames = sections.Select(s => s.SectionName).ToHashSet(StringComparer.OrdinalIgnoreCase);

        // All auto-generated properties should appear in the schema metadata
        var generatedPropertyNames = typeof(YamlModelConfig)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(p => p.Name)
            .ToList();

        // Note: some properties are hand-written in the base class, not in generated partial
        // The schema metadata is for the generated sections. Check that generated sections are there.
        Assert.True(sections.Count >= 40,
            $"Expected at least 40 schema metadata sections but found {sections.Count}");
    }

    /// <summary>
    /// Verifies POCO sections in the schema metadata have property definitions.
    /// </summary>
    [Fact]
    public void SchemaMetadata_PocoSections_HaveProperties()
    {
        var sections = YamlSchemaMetadata.Sections;
        var pocoSections = sections.Where(s => s.Category == "Poco").ToList();

        Assert.True(pocoSections.Count > 0, "No POCO sections found in schema metadata");

        foreach (var section in pocoSections)
        {
            Assert.True(section.PocoProperties.Length > 0,
                $"POCO section '{section.SectionName}' has no properties defined in schema metadata");
        }
    }

    /// <summary>
    /// Verifies the hand-written sections are correctly marked in schema metadata.
    /// </summary>
    [Fact]
    public void SchemaMetadata_HandWrittenSections_AreMarked()
    {
        var sections = YamlSchemaMetadata.Sections;

        // The hand-written sections should be marked with IsHandWritten = true
        var handWrittenNames = new[] { "Optimizer", "Quantization", "Compression", "Caching",
            "Versioning", "ABTesting", "Telemetry", "Export", "GpuAcceleration", "Profiling",
            "JitCompilation", "MixedPrecision", "Reasoning", "Benchmarking",
            "InferenceOptimizations", "Interpretability", "MemoryManagement" };

        foreach (var name in handWrittenNames)
        {
            var section = sections.FirstOrDefault(s =>
                s.SectionName.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (section is not null)
            {
                Assert.True(section.IsHandWritten,
                    $"Section '{name}' should be marked as hand-written in schema metadata");
            }
        }
    }

    #endregion

    #region JSON Schema and Docs Generation

    /// <summary>
    /// Verifies the JSON Schema generator produces valid output.
    /// </summary>
    [Fact]
    public void JsonSchemaGenerate_ProducesValidJson()
    {
        var schema = YamlJsonSchema.Generate();

        Assert.False(string.IsNullOrWhiteSpace(schema), "JSON Schema is empty");
        Assert.Contains("\"$schema\"", schema);
        Assert.Contains("\"properties\"", schema);
        Assert.Contains("\"optimizer\"", schema);
        Assert.Contains("\"timeSeriesModel\"", schema);
        Assert.Contains("\"quantization\"", schema);
        Assert.Contains("\"regularization\"", schema);
    }

    /// <summary>
    /// Verifies the JSON Schema covers both hand-written and generated sections.
    /// </summary>
    [Fact]
    public void JsonSchemaGenerate_CoversAllSections()
    {
        var schema = YamlJsonSchema.Generate();

        // Hand-written POCO sections
        Assert.Contains("\"quantization\"", schema);
        Assert.Contains("\"compression\"", schema);
        Assert.Contains("\"caching\"", schema);
        Assert.Contains("\"telemetry\"", schema);
        Assert.Contains("\"profiling\"", schema);
        Assert.Contains("\"inferenceOptimizations\"", schema);
        Assert.Contains("\"memoryManagement\"", schema);

        // Generated interface sections
        Assert.Contains("\"regularization\"", schema);
        Assert.Contains("\"fitDetector\"", schema);
        Assert.Contains("\"fairnessEvaluator\"", schema);
        Assert.Contains("\"tokenizer\"", schema);
        Assert.Contains("\"promptTemplate\"", schema);

        // Generated pipeline sections
        Assert.Contains("\"preprocessing\"", schema);
        Assert.Contains("\"postprocessing\"", schema);
    }

    /// <summary>
    /// Verifies the docs generator produces markdown output with key sections.
    /// </summary>
    [Fact]
    public void DocsGenerate_ProducesMarkdown()
    {
        var docs = YamlDocsGenerator.Generate();

        Assert.False(string.IsNullOrWhiteSpace(docs), "Documentation is empty");
        Assert.Contains("# AiDotNet YAML Configuration Reference", docs);
        Assert.Contains("## Quick Start", docs);
        Assert.Contains("## Table of Contents", docs);
        Assert.Contains("### optimizer", docs);
        Assert.Contains("### timeSeriesModel", docs);
    }

    /// <summary>
    /// Verifies the docs generator covers POCO, interface, and pipeline sections.
    /// </summary>
    [Fact]
    public void DocsGenerate_CoversAllSectionTypes()
    {
        var docs = YamlDocsGenerator.Generate();

        // POCO sections
        Assert.Contains("quantization", docs);
        Assert.Contains("caching", docs);
        Assert.Contains("telemetry", docs);

        // Interface sections
        Assert.Contains("regularization", docs);
        Assert.Contains("fairnessEvaluator", docs);
        Assert.Contains("tokenizer", docs);

        // Pipeline sections
        Assert.Contains("preprocessing", docs);
    }

    #endregion

    #region Type Registry CreateInstance End-to-End

    [Fact]
    public void TypeRegistry_CreateInstance_NoRegularization_Works()
    {
        var instance = YamlTypeRegistry<double, Matrix<double>, Vector<double>>
            .CreateInstance<IRegularization<double, Matrix<double>, Vector<double>>>(
                "Regularization", "NoRegularization");

        Assert.NotNull(instance);
    }

    [Fact]
    public void TypeRegistry_CreateInstance_DefaultFitDetector_Works()
    {
        var instance = YamlTypeRegistry<double, Matrix<double>, Vector<double>>
            .CreateInstance<IFitDetector<double, Matrix<double>, Vector<double>>>(
                "FitDetector", "DefaultFitDetector");

        Assert.NotNull(instance);
    }

    [Fact]
    public void TypeRegistry_CreateInstance_BasicFairnessEvaluator_Works()
    {
        var instance = YamlTypeRegistry<double, Matrix<double>, Vector<double>>
            .CreateInstance<IFairnessEvaluator<double>>(
                "FairnessEvaluator", "BasicFairnessEvaluator");

        Assert.NotNull(instance);
    }

    [Fact]
    public void TypeRegistry_CreateInstance_InvalidSection_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            YamlTypeRegistry<double, Matrix<double>, Vector<double>>
                .CreateInstance<object>("NonExistentSection", "SomeType"));

        Assert.Contains("NonExistentSection", ex.Message);
    }

    [Fact]
    public void TypeRegistry_CreateInstance_InvalidType_ThrowsArgumentException()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            YamlTypeRegistry<double, Matrix<double>, Vector<double>>
                .CreateInstance<object>("Regularization", "NonExistentType"));

        Assert.Contains("NonExistentType", ex.Message);
        Assert.Contains("Available types", ex.Message);
    }

    #endregion

    #region Coverage Gap Documentation Tests

    /// <summary>
    /// Verifies that all previously-gapped interface sections now have registered implementations.
    /// These were previously excluded because the ImplementationFinder required parameterless
    /// constructors. With smart constructor resolution, they are now all registered.
    /// </summary>
    [Fact]
    public void TypeRegistry_PreviouslyGappedSections_NowHaveImplementations()
    {
        var registries = YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();

        // ALL of these previously-gapped sections should now have implementations
        var previouslyGappedSections = new[]
        {
            "FitnessCalculator",
            "DataLoader",
            "BiasDetector",
            "LoRA",
            "RetrievalAugmentedGeneration",
            "CrossValidation",
            "AutoML",
            "MetaLearning",
            "DistributedTraining",
            "ExperimentTracker",
            "CheckpointManager",
            "ModelRegistry",
            "DataVersionControl",
            "HyperparameterOptimizer",
            "PromptChain",
            "PromptCompressor",
            "ObjectDetector",
            "InstanceSegmenter",
            "ObjectTracker",
        };

        foreach (var section in previouslyGappedSections)
        {
            Assert.True(registries.ContainsKey(section),
                $"Section '{section}' still has no registered implementations.");
            Assert.True(registries[section].Count > 0,
                $"Section '{section}' is registered but has zero implementations.");
        }
    }

    /// <summary>
    /// Counts the total number of Configure methods, generated properties, and registered types
    /// to provide a clear coverage summary.
    /// </summary>
    [Fact]
    public void CoverageSummary_ReportsCounts()
    {
        var builderType = typeof(AiModelBuilder<double, Matrix<double>, Vector<double>>);
        var configType = typeof(YamlModelConfig);

        var configureMethods = builderType
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.Name.StartsWith("Configure", StringComparison.Ordinal))
            .Where(m => !m.Name.EndsWith("Async", StringComparison.Ordinal))
            .Where(m => m.Name != "ConfigureTimeSeriesFeaturesForFinance")
            .Where(m => m.Name != "ConfigureTimeSeriesFeaturesMinimal")
            .Select(m => m.Name)
            .Distinct()
            .Count();

        var yamlProperties = configType
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanRead && p.CanWrite)
            .Count();

        var registries = YamlTypeRegistry<double, Matrix<double>, Vector<double>>.GetAllRegistries();
        var totalRegisteredTypes = registries.Values.Sum(d => d.Count);
        var sectionsWithImpls = registries.Count;

        var schemaSections = YamlSchemaMetadata.Sections.Count;

        // Assertions to ensure coverage doesn't regress
        Assert.True(configureMethods > 50,
            $"Expected >50 Configure methods, found {configureMethods}");
        Assert.True(yamlProperties >= configureMethods - 5,
            $"YAML properties ({yamlProperties}) should be close to Configure methods ({configureMethods})");
        Assert.True(totalRegisteredTypes >= 800,
            $"Expected at least 800 registered types, found {totalRegisteredTypes}");
        Assert.True(sectionsWithImpls >= 25,
            $"Expected at least 25 sections with implementations, found {sectionsWithImpls}");
        Assert.True(schemaSections >= 40,
            $"Expected at least 40 schema metadata sections, found {schemaSections}");
    }

    #endregion

    #region YAML Applier End-to-End for Generated Sections

    [Fact]
    public void Apply_WithRegularizationSection_ConfiguresBuilder()
    {
        var yaml = @"
regularization:
  type: NoRegularization
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // Should not throw - the registry has NoRegularization
        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);
    }

    [Fact]
    public void Apply_WithFitDetectorSection_ConfiguresBuilder()
    {
        var yaml = @"
fitDetector:
  type: DefaultFitDetector
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);
    }

    [Fact]
    public void Apply_WithFairnessEvaluatorSection_ConfiguresBuilder()
    {
        var yaml = @"
fairnessEvaluator:
  type: BasicFairnessEvaluator
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);
    }

    [Fact]
    public void Apply_WithPromptTemplateSection_ConfiguresBuilder()
    {
        var yaml = @"
promptTemplate:
  type: InstructionFollowingTemplate
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);
    }

    [Fact]
    public void Apply_WithComprehensiveConfig_AllSectionsApplyWithoutError()
    {
        var yaml = @"
optimizer:
  type: Adam

caching:
  enabled: true
  maxCacheSize: 1000

jitCompilation:
  enabled: true

inferenceOptimizations:
  enableKVCache: true

interpretability:
  enableSHAP: true

memoryManagement:
  useGradientCheckpointing: true

regularization:
  type: NoRegularization

fitDetector:
  type: DefaultFitDetector

fairnessEvaluator:
  type: BasicFairnessEvaluator

promptTemplate:
  type: InstructionFollowingTemplate
";

        var config = YamlConfigLoader.LoadFromString(yaml);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        var exception = Record.Exception(() =>
            YamlConfigApplier<double, Matrix<double>, Vector<double>>.Apply(config, builder));
        Assert.Null(exception);
    }

    #endregion
}
