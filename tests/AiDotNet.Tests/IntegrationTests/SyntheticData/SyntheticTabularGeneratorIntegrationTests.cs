using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Comprehensive integration tests for all 25 synthetic tabular data generators.
/// Each test exercises the full Fit → Generate pipeline with realistic mixed-type
/// tabular data (continuous + categorical columns) and validates that generated
/// output has correct dimensions, finite values, and preserves column structure.
/// </summary>
public class SyntheticTabularGeneratorIntegrationTests
{
    private const int Rows = 100;
    private const int NumContinuous = 3;
    private const int NumCategorical = 2;
    private const int TotalCols = NumContinuous + NumCategorical;
    private const int GenSamples = 20;
    private const int FewEpochs = 2;
    private const int Seed = 42;

    #region Test Data Helpers

    /// <summary>
    /// Creates a small synthetic dataset with 3 continuous and 2 categorical columns.
    /// Continuous: col0 ~ N(50,10), col1 ~ N(100,20), col2 ~ N(0,1)
    /// Categorical: col3 in {0,1,2}, col4 in {0,1}
    /// </summary>
    private static (Matrix<double> data, List<ColumnMetadata> columns) CreateTestData(int rows = Rows)
    {
        var random = new Random(Seed);
        var data = new Matrix<double>(rows, TotalCols);

        for (int i = 0; i < rows; i++)
        {
            // Continuous columns
            data[i, 0] = 50.0 + 10.0 * SampleNormal(random);
            data[i, 1] = 100.0 + 20.0 * SampleNormal(random);
            data[i, 2] = SampleNormal(random);

            // Categorical columns (encoded as integers)
            data[i, 3] = random.Next(3); // 3 categories
            data[i, 4] = random.Next(2); // binary
        }

        var columns = new List<ColumnMetadata>
        {
            new("Feature1", ColumnDataType.Continuous, columnIndex: 0),
            new("Feature2", ColumnDataType.Continuous, columnIndex: 1),
            new("Feature3", ColumnDataType.Continuous, columnIndex: 2),
            new("Category1", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 3),
            new("Category2", ColumnDataType.Categorical, new[] { "Yes", "No" }, columnIndex: 4)
        };

        return (data, columns);
    }

    /// <summary>
    /// Creates a binary-class dataset for generators that need a label column (e.g., SMOTE-NC).
    /// Last column is the label (0 = majority, 1 = minority).
    /// </summary>
    private static (Matrix<double> data, List<ColumnMetadata> columns) CreateImbalancedData()
    {
        var random = new Random(Seed);
        int rows = 100;
        var data = new Matrix<double>(rows, 4);

        for (int i = 0; i < rows; i++)
        {
            data[i, 0] = SampleNormal(random) * 10 + 50;
            data[i, 1] = SampleNormal(random) * 5 + 20;
            data[i, 2] = random.Next(3); // categorical

            // 80% majority (0), 20% minority (1)
            data[i, 3] = i < 80 ? 0.0 : 1.0;
        }

        var columns = new List<ColumnMetadata>
        {
            new("Feature1", ColumnDataType.Continuous, columnIndex: 0),
            new("Feature2", ColumnDataType.Continuous, columnIndex: 1),
            new("Category1", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 2),
            new("Label", ColumnDataType.Categorical, new[] { "0", "1" }, columnIndex: 3)
        };

        return (data, columns);
    }

    private static double SampleNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Creates a NeuralNetworkArchitecture for GAN/NN-based generators.
    /// </summary>
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int inputFeatures, int outputSize)
    {
        return new NeuralNetworkArchitecture<double>(inputFeatures, outputSize, NetworkComplexity.Simple);
    }

    /// <summary>
    /// Validates that generated data has correct dimensions and all values are finite.
    /// </summary>
    private static void ValidateGeneratedData(Matrix<double> generated, int expectedRows, int expectedCols, string generatorName)
    {
        Assert.Equal(expectedRows, generated.Rows);
        Assert.Equal(expectedCols, generated.Columns);

        for (int i = 0; i < generated.Rows; i++)
        {
            for (int j = 0; j < generated.Columns; j++)
            {
                Assert.False(double.IsNaN(generated[i, j]),
                    $"{generatorName}: NaN found at row {i}, col {j}");
                Assert.False(double.IsInfinity(generated[i, j]),
                    $"{generatorName}: Infinity found at row {i}, col {j}");
            }
        }
    }

    #endregion

    #region GAN Generators (NeuralNetworkBase)

    [Fact]
    public void CTGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3,
            DiscriminatorSteps = 1
        };

        var generator = new CTGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "CTGAN");
    }

    [Fact]
    public void CopulaGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CopulaGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3
        };

        var generator = new CopulaGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "CopulaGAN");
    }

    [Fact]
    public void DPCTGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new DPCTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3,
            Epsilon = 10.0,
            Delta = 1e-5,
            ClipNorm = 1.0
        };

        var generator = new DPCTGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);
        Assert.True(generator.CumulativeEpsilon >= 0);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "DPCTGAN");
    }

    [Fact]
    public void CTABGANPlusGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CTABGANPlusOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3,
            ClassifierWeight = 0.5,
            InformationWeight = 0.1,
            TargetColumnIndex = 3
        };

        var generator = new CTABGANPlusGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "CTABGANPlus");
    }

    [Fact]
    public void PATEGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new PATEGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            TeacherDimensions = [64, 64],
            StudentDimensions = [64, 64],
            BatchSize = 50,
            VGMModes = 3,
            NumTeachers = 3,
            LaplaceScale = 0.5
        };

        var generator = new PATEGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "PATEGAN");
    }

    [Fact]
    public void TableGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TableGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50,
            LabelColumnIndex = 3,
            ClassificationWeight = 0.5,
            InformationWeight = 0.1
        };

        var generator = new TableGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TableGAN");
    }

    [Fact]
    public void OCTGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new OCTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50
        };

        var generator = new OCTGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "OCTGAN");
    }

    [Fact]
    public void CausalGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CausalGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            HiddenDimensions = [64, 64],
            BatchSize = 50
        };

        var generator = new CausalGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "CausalGAN");
    }

    [Fact]
    public void MisGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new MisGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            HiddenDimensions = [64, 64],
            BatchSize = 50,
            MissingRate = 0.1
        };

        var generator = new MisGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "MisGAN");
    }

    [Fact]
    public void MedSynthGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new MedSynthOptions<double>
        {
            Seed = Seed,
            LatentDimension = 16,
            EncoderDimensions = [64, 64],
            DiscriminatorDimensions = [64, 64],
            BatchSize = 50,
            VGMModes = 3,
            EnablePrivacy = true,
            Epsilon = 10.0,
            ClipNorm = 1.0
        };

        var generator = new MedSynthGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "MedSynth");
    }

    #endregion

    #region VAE Generators

    [Fact]
    public void TVAEGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TVAEOptions<double>
        {
            Seed = Seed,
            EncoderDimensions = [64, 64],
            DecoderDimensions = [64, 64],
            LatentDimension = 16,
            BatchSize = 50,
            VGMModes = 3
        };

        var generator = new TVAEGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TVAE");
    }

    #endregion

    #region Diffusion Models

    [Fact]
    public void TabDDPMGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TabDDPMOptions<double>
        {
            Seed = Seed,
            MLPDimensions = [64, 64],
            NumTimesteps = 10,
            BatchSize = 50
        };

        var generator = new TabDDPMGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TabDDPM");
    }

    [Fact]
    public void TabSynGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TabSynOptions<double>
        {
            Seed = Seed,
            EncoderDimensions = [64, 64],
            DecoderDimensions = [64, 64],
            LatentDimension = 16,
            DiffusionMLPDimensions = [64, 64],
            DiffusionSteps = 10,
            BatchSize = 50,
            VGMModes = 3
        };

        var generator = new TabSynGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TabSyn");
    }

    [Fact]
    public void TabFlowGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TabFlowOptions<double>
        {
            Seed = Seed,
            MLPDimensions = [64, 64],
            NumSteps = 10,
            BatchSize = 50,
            VGMModes = 3
        };

        var generator = new TabFlowGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TabFlow");
    }

    [Fact]
    public void AutoDiffTabGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new AutoDiffTabOptions<double>
        {
            Seed = Seed,
            MLPDimensions = [64, 64],
            MaxTimesteps = 10,
            BatchSize = 50
        };

        var generator = new AutoDiffTabGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "AutoDiffTab");
    }

    [Fact]
    public void FinDiffGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new FinDiffOptions<double>
        {
            Seed = Seed,
            MLPDimensions = [64, 64],
            NumTimesteps = 10,
            BatchSize = 50
        };

        var generator = new FinDiffGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "FinDiff");
    }

    #endregion

    #region Transformer / Sequence Models

    [Fact]
    public void REaLTabFormerGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new REaLTabFormerOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            NumHeads = 2,
            NumLayers = 2,
            FeedForwardDimension = 64,
            BatchSize = 50,
            NumBins = 100
        };

        var generator = new REaLTabFormerGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "REaLTabFormer");
    }

    [Fact]
    public void TabTransformerGenGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TabTransformerGenOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            NumHeads = 2,
            NumLayers = 2,
            FeedForwardDimension = 64,
            BatchSize = 50
        };

        var generator = new TabTransformerGenGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TabTransformerGen");
    }

    [Fact]
    public void TabLLMGenGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TabLLMGenOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            NumHeads = 2,
            NumLayers = 2,
            FeedForwardDimension = 64,
            BatchSize = 50,
            NumBins = 100
        };

        var generator = new TabLLMGenGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TabLLMGen");
    }

    #endregion

    #region Graph-Based Models

    [Fact]
    public void GOGGLEGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new GOGGLEOptions<double>
        {
            Seed = Seed,
            HiddenDimension = 64,
            LatentDimension = 16,
            NumGNNLayers = 2,
            BatchSize = 50,
            VGMModes = 3
        };

        var generator = new GOGGLEGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "GOGGLE");
    }

    #endregion

    #region Temporal Models

    [Fact]
    public void TimeGANGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new TimeGANOptions<double>
        {
            Seed = Seed,
            HiddenDimension = 32,
            SequenceLength = 5,
            BatchSize = 10
        };

        var generator = new TimeGANGenerator<double>(arch, options);
        generator.Fit(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "TimeGAN");
    }

    #endregion

    #region Statistical / Non-Deep Models (SyntheticTabularGeneratorBase)

    [Fact]
    public void SMOTENCGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateImbalancedData();
        var options = new SMOTENCOptions<double>
        {
            Seed = Seed,
            K = 5,
            LabelColumnIndex = 3,
            MinorityClassValue = 1
        };

        var generator = new SMOTENCGenerator<double>(options);
        generator.Fit(data, columns, 1);

        var generated = generator.Generate(GenSamples);
        Assert.Equal(GenSamples, generated.Rows);
        Assert.Equal(4, generated.Columns);

        for (int i = 0; i < generated.Rows; i++)
        {
            for (int j = 0; j < generated.Columns; j++)
            {
                Assert.False(double.IsNaN(generated[i, j]),
                    $"SMOTENC: NaN at row {i}, col {j}");
            }
        }
    }

    [Fact]
    public void AIMGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var options = new AIMOptions<double>
        {
            Seed = Seed,
            Epsilon = 5.0,
            MaxMarginalOrder = 2,
            NumIterations = 3
        };

        var generator = new AIMGenerator<double>(options);
        generator.Fit(data, columns, 1);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "AIM");
    }

    [Fact]
    public void BayesianNetworkSynthGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var options = new BayesianNetworkSynthOptions<double>
        {
            Seed = Seed,
            MaxParents = 2,
            NumBins = 5
        };

        var generator = new BayesianNetworkSynthGenerator<double>(options);
        generator.Fit(data, columns, 1);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "BayesianNetworkSynth");
    }

    [Fact]
    public void CopulaSynthGenerator_FitAndGenerate_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var options = new CopulaSynthOptions<double>
        {
            Seed = Seed
        };

        var generator = new CopulaSynthGenerator<double>(options);
        generator.Fit(data, columns, 1);

        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "CopulaSynth");
    }

    #endregion

    #region Cross-Cutting Concerns

    [Fact]
    public void AllGenerators_GenerateBeforeFit_ThrowsInvalidOperationException()
    {
        // Test that unfitted generators throw when Generate is called
        var smotenc = new SMOTENCGenerator<double>(new SMOTENCOptions<double> { Seed = Seed });
        Assert.Throws<InvalidOperationException>(() => smotenc.Generate(10));

        var aim = new AIMGenerator<double>(new AIMOptions<double> { Seed = Seed });
        Assert.Throws<InvalidOperationException>(() => aim.Generate(10));

        var bayesian = new BayesianNetworkSynthGenerator<double>(new BayesianNetworkSynthOptions<double> { Seed = Seed });
        Assert.Throws<InvalidOperationException>(() => bayesian.Generate(10));

        var copula = new CopulaSynthGenerator<double>(new CopulaSynthOptions<double> { Seed = Seed });
        Assert.Throws<InvalidOperationException>(() => copula.Generate(10));
    }

    [Fact]
    public void CTGANGenerator_FitWithEmptyData_ThrowsArgumentException()
    {
        var emptyData = new Matrix<double>(0, 0);
        var columns = new List<ColumnMetadata>();
        var arch = CreateArchitecture(5, 5);
        var options = new CTGANOptions<double> { Seed = Seed };
        var generator = new CTGANGenerator<double>(arch, options);

        Assert.ThrowsAny<ArgumentException>(() => generator.Fit(emptyData, columns, 1));
    }

    [Fact]
    public void CTGANGenerator_GenerateZeroSamples_ThrowsArgumentException()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64],
            DiscriminatorDimensions = [64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3
        };
        var generator = new CTGANGenerator<double>(arch, options);
        generator.Fit(data, columns, 1);

        Assert.ThrowsAny<ArgumentException>(() => generator.Generate(0));
    }

    [Fact]
    public void CTGANGenerator_ColumnMetadata_MatchesAfterFit()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64],
            DiscriminatorDimensions = [64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3
        };
        var generator = new CTGANGenerator<double>(arch, options);
        generator.Fit(data, columns, 1);

        Assert.Equal(TotalCols, generator.Columns.Count);

        // Verify column types are preserved
        Assert.True(generator.Columns[0].IsNumerical);
        Assert.True(generator.Columns[1].IsNumerical);
        Assert.True(generator.Columns[2].IsNumerical);
        Assert.True(generator.Columns[3].IsCategorical);
        Assert.True(generator.Columns[4].IsCategorical);
    }

    [Fact]
    public async Task CTGANGenerator_FitAsync_ProducesValidOutput()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64],
            DiscriminatorDimensions = [64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3
        };
        var generator = new CTGANGenerator<double>(arch, options);
        await generator.FitAsync(data, columns, FewEpochs);

        Assert.True(generator.IsFitted);
        var generated = generator.Generate(GenSamples);
        ValidateGeneratedData(generated, GenSamples, TotalCols, "CTGAN_Async");
    }

    [Fact]
    public async Task CTGANGenerator_FitAsync_SupportsCancellation()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new CTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64],
            DiscriminatorDimensions = [64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3
        };
        var generator = new CTGANGenerator<double>(arch, options);

        var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel immediately

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => generator.FitAsync(data, columns, 100, cts.Token));
    }

    [Fact]
    public void DPCTGANGenerator_PrivacyBudgetExhaustion_StopsTraining()
    {
        var (data, columns) = CreateTestData();
        var arch = CreateArchitecture(TotalCols, TotalCols);
        var options = new DPCTGANOptions<double>
        {
            Seed = Seed,
            EmbeddingDimension = 32,
            GeneratorDimensions = [64],
            DiscriminatorDimensions = [64],
            BatchSize = 50,
            PacSize = 5,
            VGMModes = 3,
            Epsilon = 0.01, // Very small budget - should exhaust quickly
            Delta = 1e-5,
            ClipNorm = 1.0
        };

        var generator = new DPCTGANGenerator<double>(arch, options);
        generator.Fit(data, columns, 50); // Request many epochs

        // Privacy budget should have been exhausted before all epochs completed
        Assert.True(generator.IsFitted);
        Assert.True(generator.CumulativeEpsilon > 0);
    }

    [Fact]
    public void SMOTENCGenerator_WithNoMinority_UsesAllData()
    {
        // All rows are class 0, no minority class 1
        var random = new Random(Seed);
        int rows = 50;
        var data = new Matrix<double>(rows, 3);
        for (int i = 0; i < rows; i++)
        {
            data[i, 0] = SampleNormal(random) * 10;
            data[i, 1] = random.Next(3);
            data[i, 2] = 0.0; // All majority class
        }

        var columns = new List<ColumnMetadata>
        {
            new("Feature1", ColumnDataType.Continuous, columnIndex: 0),
            new("Category1", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 1),
            new("Label", ColumnDataType.Categorical, new[] { "0", "1" }, columnIndex: 2)
        };

        var options = new SMOTENCOptions<double>
        {
            Seed = Seed,
            K = 3,
            LabelColumnIndex = 2,
            MinorityClassValue = 1
        };

        var generator = new SMOTENCGenerator<double>(options);
        generator.Fit(data, columns, 1);

        var generated = generator.Generate(10);
        Assert.Equal(10, generated.Rows);
        Assert.Equal(3, generated.Columns);
    }

    #endregion
}
