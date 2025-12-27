using System.IO;
using System.Net.Http;
using AiDotNet.Data.Loaders;
using AiDotNet.Helpers;
using FilePolyfill = System.IO.FilePolyfill;

namespace AiDotNet.Data.TimeSeries;

/// <summary>
/// Loads time series datasets from the M4 Competition for benchmarking forecasting models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The M4 Competition (Makridakis Competitions) is a highly influential forecasting competition
/// that provides 100,000 time series across multiple frequencies for benchmarking forecasting methods.
/// </para>
/// <para><b>For Beginners:</b> The M4 Competition is the gold standard for evaluating time series forecasting models.
///
/// **What is M4?**
/// - A collection of 100,000 real-world time series
/// - Multiple frequencies: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
/// - Standardized train/test splits for fair comparison
/// - Established benchmark metrics (SMAPE, MASE, OWA)
///
/// **Why M4 matters:**
/// - **Industry standard**: Used by researchers and practitioners worldwide
/// - **Diverse data**: Business, economic, demographic, and financial series
/// - **Published baselines**: Compare your model against known benchmarks
/// - **Academic recognition**: Results published in major forecasting journals
///
/// **M4 Dataset Statistics:**
///
/// | Frequency | Series Count | Forecast Horizon | Typical History |
/// |-----------|-------------|------------------|-----------------|
/// | Yearly    | 23,000      | 6                | 13-835 years    |
/// | Quarterly | 24,000      | 8                | 16-866 quarters |
/// | Monthly   | 48,000      | 18               | 42-2794 months  |
/// | Weekly    | 359         | 13               | 80-2597 weeks   |
/// | Daily     | 4,227       | 14               | 93-9919 days    |
/// | Hourly    | 414         | 48               | 700-960 hours   |
/// </para>
/// </remarks>
public class M4DatasetLoader<T> : DataLoaderBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Shared HttpClient instance to avoid socket exhaustion.
    /// </summary>
    private static readonly HttpClient SharedHttpClient = new() { Timeout = TimeSpan.FromMinutes(30) };

    private readonly M4Frequency _frequency;
    private readonly string _dataPath;
    private readonly bool _autoDownload;
    private List<M4TimeSeries<T>>? _trainingSeries;
    private List<M4TimeSeries<T>>? _testSeries;
    private int _currentSeriesIndex;

    /// <summary>
    /// M4 Competition download URLs.
    /// </summary>
    /// <remarks>
    /// The base URL defaults to the official M4 GitHub repository but can be overridden
    /// at runtime by setting the <c>M4_DATASET_BASE_URL</c> environment variable. This
    /// allows users to point to a mirror or updated location if the upstream structure
    /// changes.
    /// </remarks>
    private static readonly Dictionary<M4Frequency, (string train, string test)> DatasetUrls = BuildDatasetUrls();

    /// <summary>
    /// Builds the M4 dataset URLs from a configurable base URL.
    /// </summary>
    /// <remarks>
    /// If the <c>M4_DATASET_BASE_URL</c> environment variable is set, its value is used
    /// as the base URL (trailing slashes are trimmed). Otherwise, the original GitHub
    /// <c>master</c>-branch URLs are used as defaults.
    /// </remarks>
    private static Dictionary<M4Frequency, (string train, string test)> BuildDatasetUrls()
    {
        // Default base points to the original M4 GitHub repository.
        const string defaultBaseUrl = "https://github.com/Mcompetitions/M4-methods/raw/master";

        var baseUrl = Environment.GetEnvironmentVariable("M4_DATASET_BASE_URL");
        if (string.IsNullOrWhiteSpace(baseUrl))
        {
            baseUrl = defaultBaseUrl;
        }
        else
        {
            // Normalize: remove any trailing slash to avoid double slashes in paths.
            baseUrl = baseUrl.TrimEnd('/');
        }

        return new Dictionary<M4Frequency, (string train, string test)>
        {
            [M4Frequency.Yearly] = (
                $"{baseUrl}/Dataset/Train/Yearly-train.csv",
                $"{baseUrl}/Dataset/Test/Yearly-test.csv"),
            [M4Frequency.Quarterly] = (
                $"{baseUrl}/Dataset/Train/Quarterly-train.csv",
                $"{baseUrl}/Dataset/Test/Quarterly-test.csv"),
            [M4Frequency.Monthly] = (
                $"{baseUrl}/Dataset/Train/Monthly-train.csv",
                $"{baseUrl}/Dataset/Test/Monthly-test.csv"),
            [M4Frequency.Weekly] = (
                $"{baseUrl}/Dataset/Train/Weekly-train.csv",
                $"{baseUrl}/Dataset/Test/Weekly-test.csv"),
            [M4Frequency.Daily] = (
                $"{baseUrl}/Dataset/Train/Daily-train.csv",
                $"{baseUrl}/Dataset/Test/Daily-test.csv"),
            [M4Frequency.Hourly] = (
                $"{baseUrl}/Dataset/Train/Hourly-train.csv",
                $"{baseUrl}/Dataset/Test/Hourly-test.csv")
        };
    }

    /// <summary>
    /// M4 Competition statistics per frequency.
    /// </summary>
    private static readonly Dictionary<M4Frequency, (int seriesCount, int forecastHorizon)> DatasetStats = new()
    {
        [M4Frequency.Yearly] = (23000, 6),
        [M4Frequency.Quarterly] = (24000, 8),
        [M4Frequency.Monthly] = (48000, 18),
        [M4Frequency.Weekly] = (359, 13),
        [M4Frequency.Daily] = (4227, 14),
        [M4Frequency.Hourly] = (414, 48)
    };

    /// <inheritdoc/>
    public override string Name => $"M4({_frequency})";

    /// <inheritdoc/>
    public override string Description => $"M4 Competition dataset loader for {_frequency} frequency time series";

    /// <inheritdoc/>
    public override int TotalCount => _trainingSeries?.Count ?? 0;

    /// <summary>
    /// Gets the number of time series in the dataset.
    /// </summary>
    public int SeriesCount => TotalCount;

    /// <summary>
    /// Gets the forecast horizon for this frequency.
    /// </summary>
    public int ForecastHorizon => DatasetStats[_frequency].forecastHorizon;

    /// <summary>
    /// Gets the frequency of the loaded time series.
    /// </summary>
    public M4Frequency Frequency => _frequency;

    /// <summary>
    /// Gets the training time series data.
    /// </summary>
    public IReadOnlyList<M4TimeSeries<T>> TrainingSeries => _trainingSeries ?? new List<M4TimeSeries<T>>();

    /// <summary>
    /// Gets the test time series data (ground truth for evaluation).
    /// </summary>
    public IReadOnlyList<M4TimeSeries<T>> TestSeries => _testSeries ?? new List<M4TimeSeries<T>>();

    /// <summary>
    /// Initializes a new instance of the <see cref="M4DatasetLoader{T}"/> class.
    /// </summary>
    /// <param name="frequency">The frequency of time series to load.</param>
    /// <param name="batchSize">Batch size for loading series (default: 32).</param>
    /// <param name="dataPath">Path to download/cache datasets (optional).</param>
    /// <param name="autoDownload">Whether to automatically download the dataset if not found locally.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Using M4 datasets:
    ///
    /// ```csharp
    /// // Load monthly time series
    /// var loader = new M4DatasetLoader&lt;double&gt;(
    ///     M4Frequency.Monthly,
    ///     batchSize: 32,
    ///     autoDownload: true);
    ///
    /// // Load the data
    /// await loader.LoadAsync();
    ///
    /// // Train your model on each series
    /// foreach (var series in loader.TrainingSeries)
    /// {
    ///     model.Train(series.Values);
    /// }
    ///
    /// // Evaluate using test data
    /// foreach (var (train, test) in loader.TrainingSeries.Zip(loader.TestSeries))
    /// {
    ///     var forecast = model.Forecast(train.Values, loader.ForecastHorizon);
    ///     var smape = CalculateSMAPE(forecast, test.Values);
    /// }
    /// ```
    /// </para>
    /// </remarks>
    public M4DatasetLoader(
        M4Frequency frequency,
        int batchSize = 32,
        string? dataPath = null,
        bool autoDownload = true) : base(batchSize)
    {
        _frequency = frequency;
        _dataPath = dataPath ?? GetDefaultDataPath();
        _autoDownload = autoDownload;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string datasetDir = Path.Combine(_dataPath, "M4", _frequency.ToString());

        // Ensure data exists (download if needed)
        await EnsureDataExistsAsync(datasetDir, cancellationToken);

        // Parse the training and test files
        string trainFile = Path.Combine(datasetDir, $"{_frequency}-train.csv");
        string testFile = Path.Combine(datasetDir, $"{_frequency}-test.csv");

        _trainingSeries = await ParseM4CsvAsync(trainFile, cancellationToken);
        _testSeries = await ParseM4CsvAsync(testFile, cancellationToken);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _trainingSeries = null;
        _testSeries = null;
    }

    /// <inheritdoc/>
    protected override void OnReset()
    {
        _currentSeriesIndex = 0;
    }

    /// <summary>
    /// Gets the next batch of time series for iteration.
    /// </summary>
    /// <returns>A list of training time series in the current batch.</returns>
    public List<M4TimeSeries<T>> GetNextBatch()
    {
        EnsureLoaded();

        var batch = new List<M4TimeSeries<T>>();
        int endIndex = Math.Min(_currentSeriesIndex + BatchSize, TotalCount);

        for (int i = _currentSeriesIndex; i < endIndex; i++)
        {
            batch.Add(_trainingSeries![i]);
        }

        _currentSeriesIndex = endIndex;
        CurrentIndex = _currentSeriesIndex;

        return batch;
    }

    /// <summary>
    /// Gets a specific time series by index.
    /// </summary>
    /// <param name="index">The index of the series to retrieve.</param>
    /// <returns>A tuple containing the training series and its corresponding test values.</returns>
    public (M4TimeSeries<T> train, M4TimeSeries<T> test) GetSeries(int index)
    {
        EnsureLoaded();

        if (index < 0 || index >= TotalCount)
            throw new ArgumentOutOfRangeException(nameof(index));

        return (_trainingSeries![index], _testSeries![index]);
    }

    /// <summary>
    /// Gets the default data path for caching datasets.
    /// </summary>
    private static string GetDefaultDataPath()
    {
        string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        return Path.Combine(appData, "AiDotNet", "Datasets");
    }

    /// <summary>
    /// Ensures the dataset files exist locally, downloading if necessary.
    /// </summary>
    private async Task EnsureDataExistsAsync(string datasetDir, CancellationToken cancellationToken)
    {
        string trainFile = Path.Combine(datasetDir, $"{_frequency}-train.csv");
        string testFile = Path.Combine(datasetDir, $"{_frequency}-test.csv");

        // Check if data files exist
        if (File.Exists(trainFile) && File.Exists(testFile))
        {
            return; // Data already exists
        }

        if (!_autoDownload)
        {
            throw new FileNotFoundException(
                $"Dataset files not found at {datasetDir}. " +
                $"Either provide the data files or set autoDownload=true to download automatically.");
        }

        // Download the dataset
        await DownloadDatasetAsync(datasetDir, cancellationToken);
    }

    /// <summary>
    /// Downloads the dataset from the M4 Competition GitHub repository.
    /// </summary>
    /// <remarks>
    /// Downloads to temporary files first and only moves them to the final location
    /// after successful download and basic validation to avoid partial files.
    /// </remarks>
    private async Task DownloadDatasetAsync(string datasetDir, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(datasetDir);

        var (trainUrl, testUrl) = DatasetUrls[_frequency];

        // Download training data with atomic write
        string trainFile = Path.Combine(datasetDir, $"{_frequency}-train.csv");
        await DownloadFileAtomicAsync(trainUrl, trainFile, "training", cancellationToken);

        // Download test data with atomic write
        string testFile = Path.Combine(datasetDir, $"{_frequency}-test.csv");
        await DownloadFileAtomicAsync(testUrl, testFile, "test", cancellationToken);
    }

    /// <summary>
    /// Downloads a file atomically by writing to a temp file first.
    /// </summary>
    private async Task DownloadFileAtomicAsync(string url, string targetPath, string dataType, CancellationToken cancellationToken)
    {
        string tempPath = targetPath + ".tmp";

        try
        {
            using var response = await SharedHttpClient.GetAsync(url, cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException(
                    $"Failed to download M4 {dataType} data from {url}. " +
                    $"HTTP status: {(int)response.StatusCode} {response.ReasonPhrase}. " +
                    $"You can manually download the file to: {targetPath} or set " +
                    $"M4_DATASET_BASE_URL environment variable to point to a mirror.");
            }

            var content = await response.Content.ReadAsStringAsync();

            // Basic validation - check we got CSV-like content
            if (string.IsNullOrWhiteSpace(content) || !content.Contains(','))
            {
                throw new InvalidDataException(
                    $"Downloaded M4 {dataType} data appears to be invalid (not CSV format). " +
                    $"URL: {url}. You may need to download the data manually.");
            }

            // Write to temp file first
            await FilePolyfill.WriteAllTextAsync(tempPath, content, cancellationToken);

            // Atomically move to final location
            if (File.Exists(targetPath))
            {
                File.Delete(targetPath);
            }
            File.Move(tempPath, targetPath);
        }
        finally
        {
            // Clean up temp file if it still exists (in case of error)
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); } catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Parses an M4 Competition CSV file.
    /// </summary>
    /// <remarks>
    /// M4 CSV format has a simple structure with quoted series IDs and unquoted numeric values:
    /// <c>"V1",12345,23456,...</c>
    /// Values never contain commas or quotes, so simple Split(',') is appropriate here.
    /// </remarks>
    private async Task<List<M4TimeSeries<T>>> ParseM4CsvAsync(string filePath, CancellationToken cancellationToken)
    {
        var series = new List<M4TimeSeries<T>>();
        var lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);

        // Skip header row
        for (int i = 1; i < lines.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var line = lines[i];
            if (string.IsNullOrWhiteSpace(line)) continue;

            // M4 format uses simple comma separation with quoted series ID
            var parts = line.Split(',');
            if (parts.Length < 2) continue;

            // Remove quotes from series ID (e.g., "Y1" -> Y1)
            var seriesId = parts[0].Trim('"');
            var values = new List<T>();

            for (int j = 1; j < parts.Length; j++)
            {
                var valueStr = parts[j].Trim();
                if (string.IsNullOrEmpty(valueStr) || valueStr == "NA") break;

                if (double.TryParse(valueStr, out double value))
                {
                    values.Add(NumOps.FromDouble(value));
                }
            }

            if (values.Count > 0)
            {
                series.Add(new M4TimeSeries<T>(seriesId, values, _frequency));
            }
        }

        return series;
    }

    /// <summary>
    /// Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) for M4 evaluation.
    /// </summary>
    /// <param name="forecast">The forecasted values.</param>
    /// <param name="actual">The actual test values.</param>
    /// <returns>The SMAPE score (0-200, lower is better).</returns>
    /// <remarks>
    /// <para>
    /// SMAPE is the official metric used in the M4 Competition.
    /// It's symmetric and bounded between 0% and 200%.
    /// </para>
    /// </remarks>
    public static T CalculateSMAPE(IReadOnlyList<T> forecast, IReadOnlyList<T> actual)
    {
        if (forecast.Count != actual.Count)
            throw new ArgumentException("Forecast and actual must have the same length.");

        var sum = NumOps.Zero;
        int n = forecast.Count;

        for (int i = 0; i < n; i++)
        {
            var absError = NumOps.Abs(NumOps.Subtract(forecast[i], actual[i]));
            var denominator = NumOps.Add(NumOps.Abs(forecast[i]), NumOps.Abs(actual[i]));

            if (NumOps.ToDouble(denominator) > 1e-10)
            {
                sum = NumOps.Add(sum, NumOps.Divide(absError, denominator));
            }
        }

        return NumOps.Multiply(NumOps.FromDouble(200.0 / n), sum);
    }

    /// <summary>
    /// Calculates the Mean Absolute Scaled Error (MASE) for M4 evaluation.
    /// </summary>
    /// <param name="forecast">The forecasted values.</param>
    /// <param name="actual">The actual test values.</param>
    /// <param name="trainingData">The training data used to compute the scaling factor.</param>
    /// <param name="seasonalPeriod">The seasonal period for the scaling factor (default: 1 for non-seasonal).</param>
    /// <returns>The MASE score (lower is better, 1.0 equals naive forecast).</returns>
    /// <remarks>
    /// <para>
    /// MASE is scale-independent and compares forecast accuracy against a naive seasonal forecast.
    /// A MASE of 1.0 means the forecast is as good as a seasonal naive forecast.
    /// Lower values indicate better performance.
    /// </para>
    /// </remarks>
    public static T CalculateMASE(IReadOnlyList<T> forecast, IReadOnlyList<T> actual, IReadOnlyList<T> trainingData, int seasonalPeriod = 1)
    {
        if (forecast.Count != actual.Count)
            throw new ArgumentException("Forecast and actual must have the same length.");

        // Calculate the scaling factor from training data (mean absolute error of seasonal naive forecast)
        var scalingSum = NumOps.Zero;
        int scalingCount = 0;

        for (int i = seasonalPeriod; i < trainingData.Count; i++)
        {
            scalingSum = NumOps.Add(scalingSum, NumOps.Abs(NumOps.Subtract(trainingData[i], trainingData[i - seasonalPeriod])));
            scalingCount++;
        }

        if (scalingCount == 0) return NumOps.FromDouble(double.PositiveInfinity);

        var scalingFactor = NumOps.Divide(scalingSum, NumOps.FromDouble(scalingCount));

        // Calculate MAE of forecast
        var maeSum = NumOps.Zero;
        for (int i = 0; i < forecast.Count; i++)
        {
            maeSum = NumOps.Add(maeSum, NumOps.Abs(NumOps.Subtract(forecast[i], actual[i])));
        }
        var mae = NumOps.Divide(maeSum, NumOps.FromDouble(forecast.Count));

        // MASE = MAE / scaling factor
        if (NumOps.ToDouble(scalingFactor) < 1e-10)
            return NumOps.FromDouble(double.PositiveInfinity);

        return NumOps.Divide(mae, scalingFactor);
    }
}

/// <summary>
/// Represents a single time series from the M4 Competition.
/// </summary>
/// <typeparam name="T">The numeric type used for values.</typeparam>
public class M4TimeSeries<T>
{
    /// <summary>
    /// Gets the unique identifier for this series (e.g., "Y1", "M1234", "H414").
    /// </summary>
    public string SeriesId { get; }

    /// <summary>
    /// Gets the time series values.
    /// </summary>
    public IReadOnlyList<T> Values { get; }

    /// <summary>
    /// Gets the frequency of this time series.
    /// </summary>
    public M4Frequency Frequency { get; }

    /// <summary>
    /// Gets the length of the time series.
    /// </summary>
    public int Length => Values.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="M4TimeSeries{T}"/> class.
    /// </summary>
    public M4TimeSeries(string seriesId, IReadOnlyList<T> values, M4Frequency frequency)
    {
        SeriesId = seriesId;
        Values = values;
        Frequency = frequency;
    }
}

/// <summary>
/// Frequency categories in the M4 Competition.
/// </summary>
public enum M4Frequency
{
    /// <summary>
    /// Yearly time series (23,000 series, forecast horizon: 6).
    /// </summary>
    Yearly,

    /// <summary>
    /// Quarterly time series (24,000 series, forecast horizon: 8).
    /// </summary>
    Quarterly,

    /// <summary>
    /// Monthly time series (48,000 series, forecast horizon: 18).
    /// </summary>
    Monthly,

    /// <summary>
    /// Weekly time series (359 series, forecast horizon: 13).
    /// </summary>
    Weekly,

    /// <summary>
    /// Daily time series (4,227 series, forecast horizon: 14).
    /// </summary>
    Daily,

    /// <summary>
    /// Hourly time series (414 series, forecast horizon: 48).
    /// </summary>
    Hourly
}
