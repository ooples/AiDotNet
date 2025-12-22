namespace AiDotNet.TrainingMonitoring.Dashboard;

/// <summary>
/// Interface for training dashboards that visualize metrics and training progress.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A training dashboard provides visual feedback about
/// your model's training progress. Think of it like TensorBoard - it shows
/// you graphs of loss, accuracy, and other metrics over time.
///
/// This library provides multiple dashboard implementations:
/// - HtmlDashboard: Generates interactive HTML reports
/// - LiveDashboard: Provides real-time updates via embedded web server
/// - ConsoleDashboard: Text-based visualization for terminal environments
///
/// Example usage:
/// <code>
/// using var dashboard = new HtmlDashboard("./training_logs");
///
/// // Log metrics during training
/// dashboard.LogScalar("loss", epoch, 0.5);
/// dashboard.LogScalar("accuracy", epoch, 0.85);
///
/// // Generate visualization report
/// dashboard.GenerateReport();
/// </code>
/// </remarks>
public interface ITrainingDashboard : IDisposable
{
    /// <summary>
    /// Gets the dashboard name.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the log directory.
    /// </summary>
    string LogDirectory { get; }

    /// <summary>
    /// Gets whether the dashboard is running.
    /// </summary>
    bool IsRunning { get; }

    /// <summary>
    /// Starts the dashboard.
    /// </summary>
    void Start();

    /// <summary>
    /// Stops the dashboard.
    /// </summary>
    void Stop();

    /// <summary>
    /// Logs a scalar metric.
    /// </summary>
    /// <param name="name">Metric name (e.g., "loss", "accuracy").</param>
    /// <param name="step">Training step or epoch.</param>
    /// <param name="value">Metric value.</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogScalar(string name, long step, double value, DateTime? wallTime = null);

    /// <summary>
    /// Logs multiple scalars at once.
    /// </summary>
    /// <param name="scalars">Dictionary of metric names to values.</param>
    /// <param name="step">Training step or epoch.</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogScalars(Dictionary<string, double> scalars, long step, DateTime? wallTime = null);

    /// <summary>
    /// Logs a histogram (distribution) of values.
    /// </summary>
    /// <param name="name">Histogram name (e.g., "weights", "gradients").</param>
    /// <param name="step">Training step or epoch.</param>
    /// <param name="values">The values to create a histogram from.</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogHistogram(string name, long step, double[] values, DateTime? wallTime = null);

    /// <summary>
    /// Logs an image.
    /// </summary>
    /// <param name="name">Image name/tag.</param>
    /// <param name="step">Training step.</param>
    /// <param name="imageData">Image data as PNG bytes.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogImage(string name, long step, byte[] imageData, int width, int height, DateTime? wallTime = null);

    /// <summary>
    /// Logs text.
    /// </summary>
    /// <param name="name">Text tag.</param>
    /// <param name="step">Training step.</param>
    /// <param name="text">Text content.</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogText(string name, long step, string text, DateTime? wallTime = null);

    /// <summary>
    /// Logs hyperparameters.
    /// </summary>
    /// <param name="hyperparams">Dictionary of hyperparameter names to values.</param>
    /// <param name="metrics">Optional final metrics to associate with these hyperparameters.</param>
    void LogHyperparameters(Dictionary<string, object> hyperparams, Dictionary<string, double>? metrics = null);

    /// <summary>
    /// Logs a confusion matrix.
    /// </summary>
    /// <param name="name">Matrix name.</param>
    /// <param name="step">Training step.</param>
    /// <param name="matrix">2D confusion matrix values.</param>
    /// <param name="labels">Class labels.</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogConfusionMatrix(string name, long step, int[,] matrix, string[] labels, DateTime? wallTime = null);

    /// <summary>
    /// Logs a precision-recall curve.
    /// </summary>
    /// <param name="name">Curve name.</param>
    /// <param name="step">Training step.</param>
    /// <param name="predictions">Predicted probabilities.</param>
    /// <param name="labels">True binary labels (0 or 1).</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogPRCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null);

    /// <summary>
    /// Logs an ROC curve.
    /// </summary>
    /// <param name="name">Curve name.</param>
    /// <param name="step">Training step.</param>
    /// <param name="predictions">Predicted probabilities.</param>
    /// <param name="labels">True binary labels (0 or 1).</param>
    /// <param name="wallTime">Optional wall clock time.</param>
    void LogROCCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null);

    /// <summary>
    /// Logs the model graph/architecture.
    /// </summary>
    /// <param name="modelDescription">Description of the model architecture.</param>
    void LogModelGraph(string modelDescription);

    /// <summary>
    /// Generates an HTML report of all logged data.
    /// </summary>
    /// <param name="outputPath">Path to save the HTML report.</param>
    /// <returns>Path to the generated report.</returns>
    string GenerateReport(string? outputPath = null);

    /// <summary>
    /// Exports data in TensorBoard-compatible format.
    /// </summary>
    /// <param name="outputDirectory">Directory to write TFEvents files.</param>
    void ExportTensorBoardFormat(string outputDirectory);

    /// <summary>
    /// Gets all logged scalar series.
    /// </summary>
    Dictionary<string, List<ScalarDataPoint>> GetScalarData();

    /// <summary>
    /// Gets all logged histogram series.
    /// </summary>
    Dictionary<string, List<HistogramDataPoint>> GetHistogramData();

    /// <summary>
    /// Clears all logged data.
    /// </summary>
    void Clear();

    /// <summary>
    /// Flushes any buffered data to storage.
    /// </summary>
    void Flush();
}

/// <summary>
/// Represents a scalar data point.
/// </summary>
public class ScalarDataPoint
{
    /// <summary>
    /// Gets or sets the training step.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the wall clock time.
    /// </summary>
    public DateTime WallTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the scalar value.
    /// </summary>
    public double Value { get; set; }
}

/// <summary>
/// Represents a histogram data point.
/// </summary>
public class HistogramDataPoint
{
    /// <summary>
    /// Gets or sets the training step.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the wall clock time.
    /// </summary>
    public DateTime WallTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the minimum value.
    /// </summary>
    public double Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum value.
    /// </summary>
    public double Max { get; set; }

    /// <summary>
    /// Gets or sets the number of values.
    /// </summary>
    public int Count { get; set; }

    /// <summary>
    /// Gets or sets the sum of values.
    /// </summary>
    public double Sum { get; set; }

    /// <summary>
    /// Gets or sets the sum of squared values.
    /// </summary>
    public double SumSquares { get; set; }

    /// <summary>
    /// Gets or sets the bucket limits.
    /// </summary>
    public double[] BucketLimits { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the bucket counts.
    /// </summary>
    public int[] BucketCounts { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets the mean value.
    /// </summary>
    public double Mean => Count > 0 ? Sum / Count : 0;

    /// <summary>
    /// Gets the variance.
    /// </summary>
    public double Variance => Count > 1 ? (SumSquares - (Sum * Sum / Count)) / (Count - 1) : 0;

    /// <summary>
    /// Gets the standard deviation.
    /// </summary>
    public double StdDev => Math.Sqrt(Math.Max(0, Variance));
}

/// <summary>
/// Represents an image data point.
/// </summary>
public class ImageDataPoint
{
    /// <summary>
    /// Gets or sets the training step.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the wall clock time.
    /// </summary>
    public DateTime WallTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the image data (PNG format).
    /// </summary>
    public byte[] Data { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Gets or sets the image width.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Gets or sets the image height.
    /// </summary>
    public int Height { get; set; }
}

/// <summary>
/// Represents a text data point.
/// </summary>
public class TextDataPoint
{
    /// <summary>
    /// Gets or sets the training step.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the wall clock time.
    /// </summary>
    public DateTime WallTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the text content.
    /// </summary>
    public string Text { get; set; } = string.Empty;
}

/// <summary>
/// Represents a confusion matrix data point.
/// </summary>
public class ConfusionMatrixDataPoint
{
    /// <summary>
    /// Gets or sets the training step.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the wall clock time.
    /// </summary>
    public DateTime WallTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the matrix values.
    /// </summary>
    public int[,] Matrix { get; set; } = new int[0, 0];

    /// <summary>
    /// Gets or sets the class labels.
    /// </summary>
    public string[] Labels { get; set; } = Array.Empty<string>();
}

/// <summary>
/// Represents a PR/ROC curve data point.
/// </summary>
public class CurveDataPoint
{
    /// <summary>
    /// Gets or sets the training step.
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Gets or sets the wall clock time.
    /// </summary>
    public DateTime WallTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the X values (e.g., recall for PR, FPR for ROC).
    /// </summary>
    public double[] XValues { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the Y values (e.g., precision for PR, TPR for ROC).
    /// </summary>
    public double[] YValues { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the thresholds.
    /// </summary>
    public double[] Thresholds { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the area under the curve.
    /// </summary>
    public double AUC { get; set; }
}
