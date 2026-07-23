using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression tests for <c>ConfigureAdversarialDefense</c>, which was a total no-op:
/// <c>AiModelBuilder.Coverage.cs:410</c> assigned <c>_configuredAdversarialDefense</c> and
/// nothing in the repository ever read it, so the caller's defense was stored and dropped.
///
/// <para>Two independent gates had to fall for a defense to actually run:</para>
/// <list type="number">
///   <item><c>AttachAdversarialRobustness</c> only ever reached
///         <c>SetAdversarialDefense</c> via <c>configuration.CustomDefense</c>; a standalone
///         <c>ConfigureAdversarialDefense</c> left <c>_adversarialRobustnessConfiguration</c>
///         null and returned early.</item>
///   <item><c>AiModelResult.HasAdversarialRobustness</c> was driven purely by
///         <c>EnableAdversarialTraining</c>/<c>UseInputPreprocessing</c> — both default false —
///         so <c>PredictWithDefense</c> silently fell through to <c>Predict</c> even when a
///         defense had been supplied.</item>
/// </list>
///
/// <para>These tests assert OBSERVABLE BEHAVIOUR — that the input genuinely routes through the
/// defense — rather than that a builder call constructed without throwing.</para>
/// </summary>
public class ConfigureAdversarialDefenseTests
{
    /// <summary>
    /// A defense the library does NOT ship, proving the parameter accepts any caller
    /// implementation. <see cref="PreprocessInput"/> records that it ran and returns a
    /// recognisably transformed input (all zeros) so that a prediction made through the
    /// defense is distinguishable from one made without it — the assertion cannot pass by
    /// accident.
    /// </summary>
    private sealed class SentinelDefense : IAdversarialDefense<double, Matrix<double>, Vector<double>>
    {
        public int PreprocessCallCount { get; private set; }

        public bool PreprocessCalled => PreprocessCallCount > 0;

        public Matrix<double> PreprocessInput(Matrix<double> input)
        {
            PreprocessCallCount++;
            // Zero every element: a transformation that is trivially detectable downstream.
            return new Matrix<double>(new double[input.Rows, input.Columns]);
        }

        public IFullModel<double, Matrix<double>, Vector<double>> ApplyDefense(
            Matrix<double>[] trainingData,
            Vector<double>[] labels,
            IFullModel<double, Matrix<double>, Vector<double>> model) => model;

        public RobustnessMetrics<double> EvaluateRobustness(
            IFullModel<double, Matrix<double>, Vector<double>> model,
            Matrix<double>[] testData,
            Vector<double>[] labels,
            IAdversarialAttack<double, Matrix<double>, Vector<double>> attack) => new();

        public AdversarialDefenseOptions<double> GetOptions() => new();

        public void Reset() => PreprocessCallCount = 0;

        public byte[] Serialize() => Array.Empty<byte>();

        public void Deserialize(byte[] data) { }

        public void SaveModel(string filePath) { }

        public void LoadModel(string filePath) { }
    }

    private static (Matrix<double> x, Vector<double> y) BuildDataset(int rows = 20, int features = 3)
    {
        var rng = new System.Random(123);
        var xData = new double[rows, features];
        var yData = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < features; c++)
            {
                xData[r, c] = rng.NextDouble() * 2 - 1;
                sum += xData[r, c];
            }
            yData[r] = sum;
        }
        return (new Matrix<double>(xData), new Vector<double>(yData));
    }

    private static Matrix<double> Zeroed(Matrix<double> m) => new(new double[m.Rows, m.Columns]);

    /// <summary>
    /// THE TARGET'S CORE CLAIM: ConfigureAdversarialDefense alone — no
    /// ConfigureAdversarialRobustness, no option flags — is sufficient for the defense to run.
    /// Fails on unmodified HEAD: _configuredAdversarialDefense is read nowhere.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task Standalone_ConfigureAdversarialDefense_IsUsedByPredictWithDefense()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var sentinel = new SentinelDefense();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureAdversarialDefense(sentinel)
            .BuildAsync();

        Assert.NotNull(result);

        var defended = result.PredictWithDefense(x);

        Assert.True(sentinel.PreprocessCalled,
            "The configured defense's PreprocessInput was never invoked — ConfigureAdversarialDefense is still a no-op.");

        // The defended prediction must be the prediction of the DEFENSE-TRANSFORMED input...
        var expected = result.Predict(Zeroed(x));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], defended[i], precision: 10);
        }

        // ...and must differ from the undefended prediction, or the assertion above would be
        // satisfied by a defense that had silently been skipped.
        var undefended = result.Predict(x);
        bool differs = false;
        for (int i = 0; i < undefended.Length; i++)
        {
            if (Math.Abs(undefended[i] - defended[i]) > 1e-8)
            {
                differs = true;
                break;
            }
        }
        Assert.True(differs, "Defended and undefended predictions are identical; the test cannot distinguish a wired defense from a skipped one.");
    }

    /// <summary>
    /// Negative control: with no defense configured, PredictWithDefense falls through to Predict.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task WithoutDefense_PredictWithDefense_EqualsPredict()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Null(result.AdversarialRobustnessOptions);

        var plain = result.Predict(x);
        var defended = result.PredictWithDefense(x);
        for (int i = 0; i < plain.Length; i++)
        {
            Assert.Equal(plain[i], defended[i], precision: 10);
        }
    }

    /// <summary>
    /// Pre-existing-bug regression guard (gate 2): a defense supplied via the SUPPORTED path —
    /// AdversarialRobustnessConfiguration.CustomDefense — with DEFAULT options (both
    /// EnableAdversarialTraining and UseInputPreprocessing false) was stored on the result and
    /// then silently ignored by PredictWithDefense. Fails on unmodified HEAD.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ConfigurationCustomDefense_StillHonored()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var sentinel = new SentinelDefense();

        var configuration = new AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>
        {
            CustomDefense = sentinel
            // Options left at default: EnableAdversarialTraining = false, UseInputPreprocessing = false.
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureAdversarialRobustness(configuration)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Same(configuration.Options, result.AdversarialRobustnessOptions);

        result.PredictWithDefense(x);

        Assert.True(sentinel.PreprocessCalled,
            "A defense supplied via AdversarialRobustnessConfiguration.CustomDefense must run without the caller also having to flip an unrelated option flag.");
    }

    /// <summary>
    /// An explicit Enabled = false opt-out suppresses even a standalone ConfigureAdversarialDefense.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task DisabledConfiguration_SuppressesDefense()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var sentinel = new SentinelDefense();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureAdversarialRobustness(
                AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>.Disabled())
            .ConfigureAdversarialDefense(sentinel)
            .BuildAsync();

        Assert.NotNull(result);

        var plain = result.Predict(x);
        var defended = result.PredictWithDefense(x);

        Assert.False(sentinel.PreprocessCalled,
            "An explicit Enabled = false opt-out must suppress the configured defense.");
        for (int i = 0; i < plain.Length; i++)
        {
            Assert.Equal(plain[i], defended[i], precision: 10);
        }
    }

    /// <summary>
    /// A SHIPPED defense is accepted through the same standalone call — the parameter is not
    /// limited to test doubles.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task Standalone_ShippedDefense_IsAttachedToResult()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var shipped = new AdversarialTraining<double, Matrix<double>, Vector<double>>(
            new AdversarialDefenseOptions<double>());

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureAdversarialDefense(shipped)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Same(shipped, result.AdversarialDefense);
        Assert.True(result.HasAdversarialRobustness,
            "A configured defense is adversarial robustness; the inference gate must be open.");
    }
}
