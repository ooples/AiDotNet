using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for <see cref="TableGANGenerator{T}"/> (Park et al. 2018,
/// "Data Synthesis based on Generative Adversarial Networks"). Like
/// <see cref="DocumentReaderTests"/>, the model's overridden
/// <see cref="TableGANGenerator{T}.Train"/> is intentionally a no-op because
/// real training runs through <see cref="TableGANGenerator{T}.Fit"/> against
/// a tabular matrix plus column metadata — the standard
/// <c>NeuralNetworkModelTestBase</c> invariants exercise <c>Train(Tensor, Tensor)</c>
/// and therefore can never observe parameter updates here. This standalone
/// scaffold covers the model's real surface (construction, Predict, the
/// no-op Train contract, and Fit + Generate) and, by virtue of being named
/// <c>TableGANGeneratorTests</c>, suppresses the auto-generated stub from
/// <c>TestScaffoldGenerator</c> that would otherwise emit failing
/// invariant tests under the <c>Generated.</c> namespace.
/// </summary>
public class TableGANGeneratorTests
{
    private static TableGANGenerator<double> CreateGenerator(int? seed = 42)
    {
        // Defaults match TableGANOptions: EmbeddingDimension=100,
        // GeneratorDimensions=[256, 256], DiscriminatorDimensions=[256, 256].
        // Seed is set for reproducibility of the Fit-driven training step;
        // the rest matches the paper's hyperparameters exactly.
        var options = new TableGANOptions<double>
        {
            Seed = seed,
        };
        return new TableGANGenerator<double>(options.ToArchitecture(), options);
    }

    /// <summary>
    /// Paper-faithful 4-column synthetic dataset:
    /// 3 continuous numeric columns + 1 categorical label column.
    /// Mirrors the dataset shape the model's classification head + information
    /// loss were designed for (see TableGANOptions.LabelColumnIndex).
    /// </summary>
    private static (Matrix<double> data, ColumnMetadata[] columns) BuildToyDataset(int rows = 32)
    {
        var data = new Matrix<double>(rows, 4);
        var rand = new Random(123);
        for (int r = 0; r < rows; r++)
        {
            // Three correlated continuous features
            double x = rand.NextDouble();
            data[r, 0] = x;
            data[r, 1] = x * 0.7 + rand.NextDouble() * 0.3;
            data[r, 2] = x * x;
            // Binary label derived from the first feature
            data[r, 3] = x > 0.5 ? 1.0 : 0.0;
        }

        var columns = new[]
        {
            new ColumnMetadata("f0", ColumnDataType.Continuous,  columnIndex: 0),
            new ColumnMetadata("f1", ColumnDataType.Continuous,  columnIndex: 1),
            new ColumnMetadata("f2", ColumnDataType.Continuous,  columnIndex: 2),
            new ColumnMetadata("label", ColumnDataType.Categorical, new[] { "0", "1" }, columnIndex: 3),
        };
        return (data, columns);
    }

    /// <summary>
    /// Construction smoke: the GAN must initialize without throwing under
    /// paper-default options. Covers the layered generator init in
    /// <see cref="TableGANGenerator{T}.InitializeLayers"/> and the deferred
    /// classifier-head wiring.
    /// </summary>
    [Fact]
    public void Constructor_DefaultOptions_DoesNotThrow()
    {
        using var gen = CreateGenerator();
        Assert.NotNull(gen);
        Assert.False(gen.IsFitted);
    }

    /// <summary>
    /// Predict contract: the generator's forward pass takes a noise tensor
    /// shaped <c>[batch, EmbeddingDimension]</c> and emits a synthetic
    /// row tensor through the configured generator layers. Output must be
    /// finite and have the configured output dimensionality.
    /// </summary>
    [Fact]
    public void Predict_NoiseInput_ReturnsFiniteRow()
    {
        using var gen = CreateGenerator();
        var options = new TableGANOptions<double>();
        var noise = new Tensor<double>([1, options.EmbeddingDimension]);
        for (int i = 0; i < noise.Length; i++)
            noise[i] = 0.0;

        var output = gen.Predict(noise);

        Assert.NotNull(output);
        // Strict size assertion — without it, the finite-value loop below is
        // skipped on an empty output and the test silently passes for an
        // entirely-missing prediction.
        Assert.Equal(options.EmbeddingDimension, output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            double v = output[i];
            Assert.False(double.IsNaN(v) || double.IsInfinity(v),
                $"Predict output[{i}] is not finite: {v}");
        }
    }

    /// <summary>
    /// Train contract: <see cref="TableGANGenerator{T}.Train(Tensor{double}, Tensor{double})"/>
    /// is intentionally a no-op (TableGAN trains via <c>Fit</c>). Verify it
    /// doesn't throw on arbitrary tensor input — important because the
    /// composite is plumbed through the standard <c>IFullModel</c> surface
    /// where <c>PredictionModelBuilder</c> code paths can call Train.
    /// </summary>
    [Fact]
    public void Train_NoOp_DoesNotThrow()
    {
        using var gen = CreateGenerator();
        var input = new Tensor<double>([1, 100]);
        var target = new Tensor<double>([1, 4]);

        var ex = Record.Exception(() => gen.Train(input, target));

        Assert.Null(ex);
    }

    /// <summary>
    /// End-to-end Fit smoke: paper-faithful training path. Fit must
    /// transform the input matrix, build the discriminator + classifier,
    /// run the configured number of epochs, and flag the model as fitted.
    /// This is the real training entry point — the auto-generated
    /// <c>Training_ShouldChangeParameters</c> invariant only observed the
    /// no-op <c>Train</c> override and therefore reported a false failure.
    /// <para>
    /// Asserts an actual training-effect signal (output on a fixed-noise
    /// probe changes between the pre-Fit and post-Fit generator layers)
    /// in addition to the metadata flags, so a no-op Fit implementation
    /// or a Fit that silently skips the generator update would fail this
    /// test rather than passing on bookkeeping alone.
    /// </para>
    /// </summary>
    [Fact]
    public void Fit_TinyDataset_MarksGeneratorAsFitted()
    {
        var (data, columns) = BuildToyDataset(rows: 32);
        using var gen = CreateGenerator();

        // Fixed-noise probe drawn from the same RNG that the generator
        // builder uses, so identical seed → identical baseline across runs.
        // EmbeddingDimension here matches CreateGenerator()'s default
        // TableGANOptions so the noise tensor shape lines up with the
        // generator's input layer.
        var probeOptions = new TableGANOptions<double>();
        var probe = new Tensor<double>([1, probeOptions.EmbeddingDimension]);
        var probeRng = new System.Random(7);  // positional — Random's seed param name differs across TFMs
        for (int i = 0; i < probe.Length; i++)
            probe[i] = probeRng.NextDouble();

        // First Fit absorbs the generator-rebuild-to-transformed-width step
        // (the no-trainable-weight-update edge case CodeRabbit flagged) so
        // the snapshot we take afterward reflects the stable post-rebuild
        // weights — any further change must come from actual gradient steps.
        gen.Fit(data, columns, epochs: 1);

        Assert.True(gen.IsFitted);
        Assert.Equal(columns.Length, gen.Columns.Count);

        var stableWidthOutput = gen.Predict(probe);
        var stableSnapshot = new double[stableWidthOutput.Length];
        for (int i = 0; i < stableWidthOutput.Length; i++)
            stableSnapshot[i] = stableWidthOutput[i];

        // Second Fit runs purely against the stable-width architecture, so a
        // shape-change-only "training" pass cannot satisfy the assertion below.
        gen.Fit(data, columns, epochs: 1);

        var postSecondFit = gen.Predict(probe);
        Assert.Equal(stableSnapshot.Length, postSecondFit.Length);

        double l2 = 0.0;
        for (int i = 0; i < postSecondFit.Length; i++)
        {
            double d = postSecondFit[i] - stableSnapshot[i];
            l2 += d * d;
        }
        Assert.True(l2 > 0.0,
            $"Fit produced no observable training effect across two epochs of "
            + $"stable-width training: outputs on a fixed noise probe are "
            + $"identical (L2={l2}). Either Fit silently skipped the "
            + $"generator update or the no-op Train override slipped into "
            + $"the Fit path.");
    }

    /// <summary>
    /// Generate contract: after Fit, <see cref="TableGANGenerator{T}.Generate"/>
    /// produces a matrix of synthetic rows with the same column count as
    /// the input. Values must be finite — NaN / Inf in any cell indicates
    /// either an exploding generator or a broken inverse-transform step.
    /// </summary>
    [Fact]
    public void Generate_AfterFit_ProducesFiniteSyntheticRows()
    {
        var (data, columns) = BuildToyDataset(rows: 32);
        using var gen = CreateGenerator();
        gen.Fit(data, columns, epochs: 1);

        var synthetic = gen.Generate(numSamples: 8);

        Assert.NotNull(synthetic);
        Assert.Equal(8, synthetic.Rows);
        Assert.Equal(columns.Length, synthetic.Columns);
        for (int r = 0; r < synthetic.Rows; r++)
        for (int c = 0; c < synthetic.Columns; c++)
        {
            double v = synthetic[r, c];
            Assert.False(double.IsNaN(v) || double.IsInfinity(v),
                $"Synthetic[{r},{c}] not finite: {v}");
        }
    }
}

/// <summary>
/// Small <see cref="TableGANOptions{T}"/> extension to derive the paper's
/// noise→row architecture from option values without forcing every test
/// site to repeat the same <see cref="NeuralNetworkArchitecture{T}"/>
/// boilerplate.
/// </summary>
internal static class TableGANOptionsArchitectureExtensions
{
    public static NeuralNetworkArchitecture<double> ToArchitecture(this TableGANOptions<double> opts)
        => new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: opts.EmbeddingDimension,
            outputSize: opts.EmbeddingDimension);
}
