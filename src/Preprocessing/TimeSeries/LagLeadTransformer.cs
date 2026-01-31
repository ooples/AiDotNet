using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Creates lagged and leading features from time series data.
/// </summary>
/// <remarks>
/// <para>
/// This transformer shifts data backward (lag) or forward (lead) in time to create
/// features representing past or future values at each time step.
/// </para>
/// <para><b>For Beginners:</b> Lagged features capture historical information:
///
/// For example, if you want to predict today's stock price:
/// - Lag 1: Yesterday's price
/// - Lag 2: Price from 2 days ago
/// - Lag 7: Price from a week ago
///
/// Lead features capture future values (useful for targets, not features):
/// - Lead 1: Tomorrow's price (what you might want to predict)
///
/// Why lag features matter:
/// - Many time series have autocorrelation (past values predict future)
/// - They help models learn temporal patterns
/// - They're the simplest form of "memory" for a model
///
/// Example:
/// Original: [100, 101, 102, 103, 104]
/// Lag-1:    [NaN, 100, 101, 102, 103]
/// Lag-2:    [NaN, NaN, 100, 101, 102]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LagLeadTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The lag steps to create.
    /// </summary>
    private readonly int[] _lagSteps;

    /// <summary>
    /// The lead steps to create.
    /// </summary>
    private readonly int[] _leadSteps;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new lag/lead transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public LagLeadTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _lagSteps = Options.LagSteps;
        _leadSteps = Options.LeadSteps;
    }

    #endregion

    #region Properties

    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    #endregion

    #region Core Implementation

    /// <inheritdoc />
    protected override void FitCore(Tensor<T> data)
    {
        // Lag/Lead don't need to learn parameters
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        for (int t = 0; t < timeSteps; t++)
        {
            int outputIdx = 0;

            // Create lag features
            foreach (int lag in _lagSteps)
            {
                for (int f = 0; f < inputFeatures; f++)
                {
                    int sourceTime = t - lag;
                    T value = sourceTime >= 0
                        ? GetValue(data, sourceTime, f)
                        : GetNaN();
                    output[t, outputIdx++] = value;
                }
            }

            // Create lead features
            foreach (int lead in _leadSteps)
            {
                for (int f = 0; f < inputFeatures; f++)
                {
                    int sourceTime = t + lead;
                    T value = sourceTime < timeSteps
                        ? GetValue(data, sourceTime, f)
                        : GetNaN();
                    output[t, outputIdx++] = value;
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        Parallel.For(0, timeSteps, t =>
        {
            int outputIdx = 0;

            foreach (int lag in _lagSteps)
            {
                for (int f = 0; f < inputFeatures; f++)
                {
                    int sourceTime = t - lag;
                    T value = sourceTime >= 0
                        ? GetValue(data, sourceTime, f)
                        : GetNaN();
                    output[t, outputIdx++] = value;
                }
            }

            foreach (int lead in _leadSteps)
            {
                for (int f = 0; f < inputFeatures; f++)
                {
                    int sourceTime = t + lead;
                    T value = sourceTime < timeSteps
                        ? GetValue(data, sourceTime, f)
                        : GetNaN();
                    output[t, outputIdx++] = value;
                }
            }
        });

        return output;
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var inputNames = GetInputFeatureNames();
        var sep = GetSeparator();

        // Lag feature names
        foreach (int lag in _lagSteps)
        {
            foreach (string inputName in inputNames)
            {
                names.Add($"{inputName}{sep}lag{sep}{lag}");
            }
        }

        // Lead feature names
        foreach (int lead in _leadSteps)
        {
            foreach (string inputName in inputNames)
            {
                names.Add($"{inputName}{sep}lead{sep}{lead}");
            }
        }

        return [.. names];
    }

    /// <inheritdoc />
    protected override string[] GetOperationNames()
    {
        var ops = new List<string>();

        foreach (int lag in _lagSteps)
        {
            ops.Add($"lag_{lag}");
        }

        foreach (int lead in _leadSteps)
        {
            ops.Add($"lead_{lead}");
        }

        return [.. ops];
    }

    #endregion
}
