namespace AiDotNet.Logging;

/// <summary>
/// PyTorch-compatible SummaryWriter for TensorBoard logging.
/// </summary>
/// <remarks>
/// <para>
/// This class provides an API similar to PyTorch's torch.utils.tensorboard.SummaryWriter,
/// making it easy to log training metrics, model weights, images, and more.
/// </para>
/// <para><b>For Beginners:</b> This is your interface to TensorBoard visualization.
///
/// During training, you use this writer to record:
/// - Loss values at each step (add_scalar)
/// - Model weight distributions (add_histogram)
/// - Sample outputs or feature maps (add_image)
/// - Model structure (add_graph)
///
/// Then you can visualize all this in TensorBoard by running:
/// tensorboard --logdir=your_log_directory
///
/// Example usage:
/// <code>
/// using var writer = new SummaryWriter("runs/experiment_1");
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     float loss = Train();
///     writer.AddScalar("loss/train", loss, epoch);
///     writer.AddHistogram("layer1/weights", model.Layer1.Weights, epoch);
/// }
/// </code>
/// </para>
/// </remarks>
public class SummaryWriter : IDisposable
{
    private readonly TensorBoardWriter _writer;
    private readonly string _logDir = string.Empty;
    private readonly string _comment = string.Empty;
    private long _defaultStep;
    private bool _disposed;

    /// <summary>
    /// Gets the log directory path.
    /// </summary>
    public string LogDir => _logDir;

    /// <summary>
    /// Gets the current default step number.
    /// </summary>
    public long DefaultStep => _defaultStep;

    /// <summary>
    /// Creates a new SummaryWriter.
    /// </summary>
    /// <param name="logDir">Directory to save event files. If null, uses 'runs/DATETIME_HOSTNAME'.</param>
    /// <param name="comment">Optional comment to append to the auto-generated logdir name.</param>
    /// <param name="purgeStep">Step at which to purge old data (not implemented yet).</param>
    /// <param name="maxQueue">Maximum number of pending events (not implemented yet).</param>
    /// <param name="flushSecs">How often to flush (not implemented yet).</param>
    /// <param name="filename">Optional filename suffix.</param>
    public SummaryWriter(
        string? logDir = null,
        string? comment = null,
        int purgeStep = 0,
        int maxQueue = 10,
        int flushSecs = 120,
        string? filename = null)
    {
        _comment = comment ?? "";

        // Generate log directory if not provided
        if (string.IsNullOrEmpty(logDir))
        {
            var timestamp = DateTime.Now.ToString("MMdd_HHmmss");
            var hostname = Environment.MachineName.ToLowerInvariant();
            _logDir = Path.Combine("runs", $"{timestamp}_{hostname}{(string.IsNullOrEmpty(_comment) ? "" : "_" + _comment)}");
        }
        else
        {
            _logDir = logDir!;
        }

        _writer = new TensorBoardWriter(_logDir!, filename);
        _defaultStep = 0;
    }

    /// <summary>
    /// Adds a scalar value to the summary.
    /// </summary>
    /// <param name="tag">Data identifier (e.g., "loss/train", "accuracy/val").</param>
    /// <param name="value">Scalar value to record.</param>
    /// <param name="step">Global step value. Uses auto-incremented default if not specified.</param>
    public void AddScalar(string tag, float value, long? step = null)
    {
        _writer.WriteScalar(tag, value, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds a scalar value (double precision).
    /// </summary>
    public void AddScalar(string tag, double value, long? step = null)
    {
        AddScalar(tag, (float)value, step);
    }

    /// <summary>
    /// Adds multiple scalars under a main tag.
    /// </summary>
    /// <param name="mainTag">Main tag prefix.</param>
    /// <param name="tagScalarDict">Dictionary mapping tag suffixes to values.</param>
    /// <param name="step">Global step value.</param>
    /// <remarks>
    /// Useful for comparing multiple runs. All scalars will be grouped together
    /// in TensorBoard under the main tag.
    /// </remarks>
    public void AddScalars(string mainTag, Dictionary<string, float> tagScalarDict, long? step = null)
    {
        _writer.WriteScalars(mainTag, tagScalarDict, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds a histogram of values.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="values">Array of values to build histogram from.</param>
    /// <param name="step">Global step value.</param>
    /// <param name="bins">Number of bins (not implemented, uses auto).</param>
    public void AddHistogram(string tag, float[] values, long? step = null, int bins = 64)
    {
        _writer.WriteHistogram(tag, values, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds a histogram from a 2D array (flattened).
    /// </summary>
    public void AddHistogram(string tag, float[,] values, long? step = null, int bins = 64)
    {
        int rows = values.GetLength(0);
        int cols = values.GetLength(1);
        var flat = new float[rows * cols];
        int idx = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                flat[idx++] = values[i, j];
            }
        }
        AddHistogram(tag, flat, step, bins);
    }

    /// <summary>
    /// Adds a histogram from a span of values.
    /// </summary>
    public void AddHistogram(string tag, ReadOnlySpan<float> values, long? step = null, int bins = 64)
    {
        _writer.WriteHistogram(tag, values, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds an image to the summary.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="imageData">Image data in CHW format (channels, height, width) normalized to [0, 1].</param>
    /// <param name="step">Global step value.</param>
    /// <param name="dataformats">Format of the image data: 'CHW' or 'HWC'. Default is 'CHW'.</param>
    public void AddImage(string tag, float[,,] imageData, long? step = null, string dataformats = "CHW")
    {
        int c, h, w;
        if (dataformats == "CHW")
        {
            c = imageData.GetLength(0);
            h = imageData.GetLength(1);
            w = imageData.GetLength(2);
        }
        else // HWC
        {
            h = imageData.GetLength(0);
            w = imageData.GetLength(1);
            c = imageData.GetLength(2);
        }

        // Convert to byte array in HWC format
        var pixels = new byte[h * w * c];
        int idx = 0;

        for (int row = 0; row < h; row++)
        {
            for (int col = 0; col < w; col++)
            {
                for (int ch = 0; ch < c; ch++)
                {
                    float val;
                    if (dataformats == "CHW")
                        val = imageData[ch, row, col];
                    else
                        val = imageData[row, col, ch];

                    // Clamp and convert to [0, 255]
                    pixels[idx++] = (byte)MathPolyfill.Clamp(val * 255, 0, 255);
                }
            }
        }

        _writer.WriteImageRaw(tag, pixels, h, w, c, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds an image from raw pixel data.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="pixels">Raw pixel data in HWC format, values in [0, 255].</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="channels">Number of channels (1, 3, or 4).</param>
    /// <param name="step">Global step value.</param>
    public void AddImageRaw(string tag, byte[] pixels, int height, int width, int channels, long? step = null)
    {
        _writer.WriteImageRaw(tag, pixels, height, width, channels, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds a grid of images.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="images">4D tensor of images in NCHW format.</param>
    /// <param name="step">Global step value.</param>
    /// <param name="nrow">Number of images per row in the grid.</param>
    /// <param name="padding">Padding between images.</param>
    /// <param name="normalize">Whether to normalize images to [0, 1].</param>
    public void AddImages(string tag, float[,,,] images, long? step = null, int nrow = 8, int padding = 2, bool normalize = false)
    {
        int n = images.GetLength(0);
        int c = images.GetLength(1);
        int h = images.GetLength(2);
        int w = images.GetLength(3);

        // Calculate grid dimensions
        int ncol = (n + nrow - 1) / nrow;
        int gridH = ncol * (h + padding) - padding;
        int gridW = nrow * (w + padding) - padding;

        // Create grid
        var grid = new float[c, gridH, gridW];

        // Fill with padding color (gray)
        for (int ch = 0; ch < c; ch++)
        {
            for (int row = 0; row < gridH; row++)
            {
                for (int col = 0; col < gridW; col++)
                {
                    grid[ch, row, col] = 0.5f;
                }
            }
        }

        // Place images in grid
        for (int i = 0; i < n; i++)
        {
            int gridRow = i / nrow;
            int gridCol = i % nrow;
            int startY = gridRow * (h + padding);
            int startX = gridCol * (w + padding);

            for (int ch = 0; ch < c; ch++)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        float val = images[i, ch, row, col];
                        if (normalize)
                        {
                            val = MathPolyfill.Clamp(val, 0, 1);
                        }
                        grid[ch, startY + row, startX + col] = val;
                    }
                }
            }
        }

        // Convert CHW to float[,,] and add
        var gridImage = new float[c, gridH, gridW];
        Array.Copy(grid, gridImage, grid.Length);
        AddImage(tag, gridImage, step, "CHW");
    }

    /// <summary>
    /// Adds text to the summary.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="text">Text string to record.</param>
    /// <param name="step">Global step value.</param>
    public void AddText(string tag, string text, long? step = null)
    {
        _writer.WriteText(tag, text, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds hyperparameters and associated metrics.
    /// </summary>
    /// <param name="hparams">Dictionary of hyperparameter names to values.</param>
    /// <param name="metrics">Dictionary of metric names to values.</param>
    /// <param name="hparamDomainDiscrete">Optional discrete domains for hyperparameters.</param>
    public void AddHparams(
        Dictionary<string, object> hparams,
        Dictionary<string, float> metrics,
        Dictionary<string, object[]>? hparamDomainDiscrete = null)
    {
        // Write hyperparameters as text for now (full HParam plugin support would require more protobuf work)
        var hparamText = string.Join("\n", hparams.Select(kv => $"{kv.Key}: {kv.Value}"));
        _writer.WriteText("hparams/config", hparamText, 0);

        // Write metrics as scalars
        foreach (var (name, value) in metrics)
        {
            _writer.WriteScalar($"hparams/{name}", value, 0);
        }
    }

    /// <summary>
    /// Adds an embedding with optional metadata and labels.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="embeddings">Embedding vectors (N x D).</param>
    /// <param name="metadata">Optional labels for each embedding point.</param>
    /// <param name="labelImg">Optional image for each point (N x C x H x W).</param>
    /// <param name="step">Global step value.</param>
    public void AddEmbedding(
        string tag,
        float[,] embeddings,
        string[]? metadata = null,
        float[,,,]? labelImg = null,
        long? step = null)
    {
        _writer.WriteEmbedding(tag, embeddings, metadata, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds a PR curve for binary classification evaluation.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="labels">Ground truth labels (0 or 1).</param>
    /// <param name="predictions">Prediction scores.</param>
    /// <param name="step">Global step value.</param>
    /// <param name="numThresholds">Number of thresholds for the curve.</param>
    public void AddPrCurve(string tag, int[] labels, float[] predictions, long? step = null, int numThresholds = 127)
    {
        // Calculate precision-recall at various thresholds
        var thresholds = Enumerable.Range(0, numThresholds)
            .Select(i => (float)i / (numThresholds - 1))
            .ToArray();

        var precisions = new List<float>();
        var recalls = new List<float>();

        foreach (var threshold in thresholds)
        {
            int tp = 0, fp = 0, fn = 0;
            for (int i = 0; i < labels.Length; i++)
            {
                bool predicted = predictions[i] >= threshold;
                bool actual = labels[i] == 1;

                if (predicted && actual) tp++;
                else if (predicted && !actual) fp++;
                else if (!predicted && actual) fn++;
            }

            float precision = tp + fp > 0 ? (float)tp / (tp + fp) : 1;
            float recall = tp + fn > 0 ? (float)tp / (tp + fn) : 0;

            precisions.Add(precision);
            recalls.Add(recall);
        }

        // Write as custom scalar for now
        var text = $"PR Curve - {tag}\n" +
                   string.Join("\n", Enumerable.Range(0, numThresholds)
                       .Select(i => $"Threshold: {thresholds[i]:F3}, Precision: {precisions[i]:F3}, Recall: {recalls[i]:F3}"));
        _writer.WriteText($"{tag}/pr_curve", text, step ?? _defaultStep++);
    }

    /// <summary>
    /// Adds a custom scalar with layout.
    /// </summary>
    /// <param name="tag">Data identifier.</param>
    /// <param name="value">Scalar value.</param>
    /// <param name="step">Global step value.</param>
    public void AddCustomScalar(string tag, float value, long? step = null)
    {
        AddScalar(tag, value, step);
    }

    /// <summary>
    /// Logs training metrics at the current step.
    /// </summary>
    /// <param name="loss">Training loss.</param>
    /// <param name="accuracy">Training accuracy (optional).</param>
    /// <param name="learningRate">Current learning rate (optional).</param>
    /// <param name="step">Global step.</param>
    public void LogTrainingStep(float loss, float? accuracy = null, float? learningRate = null, long? step = null)
    {
        var s = step ?? _defaultStep++;
        AddScalar("train/loss", loss, s);
        if (accuracy.HasValue)
            AddScalar("train/accuracy", accuracy.Value, s);
        if (learningRate.HasValue)
            AddScalar("train/learning_rate", learningRate.Value, s);
    }

    /// <summary>
    /// Logs validation metrics.
    /// </summary>
    /// <param name="loss">Validation loss.</param>
    /// <param name="accuracy">Validation accuracy (optional).</param>
    /// <param name="step">Global step.</param>
    public void LogValidationStep(float loss, float? accuracy = null, long? step = null)
    {
        var s = step ?? _defaultStep++;
        AddScalar("val/loss", loss, s);
        if (accuracy.HasValue)
            AddScalar("val/accuracy", accuracy.Value, s);
    }

    /// <summary>
    /// Logs model weight statistics.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <param name="weights">Weight array.</param>
    /// <param name="gradients">Gradient array (optional).</param>
    /// <param name="step">Global step.</param>
    public void LogWeights(string layerName, float[] weights, float[]? gradients = null, long? step = null)
    {
        var s = step ?? _defaultStep++;
        AddHistogram($"weights/{layerName}", weights, s);

        if (gradients != null)
        {
            AddHistogram($"gradients/{layerName}", gradients, s);

            // Log gradient magnitude
            float gradMag = (float)Math.Sqrt(gradients.Sum(g => g * g));
            AddScalar($"gradient_norm/{layerName}", gradMag, s);
        }

        // Log weight statistics
        float mean = weights.Average();
        float std = (float)Math.Sqrt(weights.Average(w => (w - mean) * (w - mean)));
        AddScalar($"weight_stats/{layerName}/mean", mean, s);
        AddScalar($"weight_stats/{layerName}/std", std, s);
    }

    /// <summary>
    /// Flushes pending writes to disk.
    /// </summary>
    public void Flush()
    {
        _writer.Flush();
    }

    /// <summary>
    /// Releases resources and closes the writer.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _writer.Dispose();
    }

    /// <summary>
    /// Closes the writer (alias for Dispose).
    /// </summary>
    public void Close()
    {
        Dispose();
    }
}

/// <summary>
/// Extension methods for easy TensorBoard logging.
/// </summary>
public static class TensorBoardExtensions
{
    /// <summary>
    /// Creates a SummaryWriter for the current run.
    /// </summary>
    /// <param name="experimentName">Name of the experiment.</param>
    /// <param name="runName">Optional run name (defaults to timestamp).</param>
    /// <returns>A new SummaryWriter instance.</returns>
    public static SummaryWriter CreateTensorBoardWriter(string experimentName, string? runName = null)
    {
        runName ??= DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var logDir = Path.Combine("runs", experimentName, runName);
        return new SummaryWriter(logDir);
    }

    /// <summary>
    /// Logs a dictionary of metrics to TensorBoard.
    /// </summary>
    /// <param name="writer">The summary writer.</param>
    /// <param name="metrics">Dictionary of metric names to values.</param>
    /// <param name="step">Global step.</param>
    /// <param name="prefix">Optional prefix for all metric tags.</param>
    public static void LogMetrics(this SummaryWriter writer, Dictionary<string, float> metrics, long step, string? prefix = null)
    {
        foreach (var (name, value) in metrics)
        {
            var tag = string.IsNullOrEmpty(prefix) ? name : $"{prefix}/{name}";
            writer.AddScalar(tag, value, step);
        }
    }
}

/// <summary>
/// Context manager for training runs with automatic TensorBoard logging.
/// </summary>
public class TensorBoardTrainingContext : IDisposable
{
    private readonly SummaryWriter _writer;
    private long _globalStep;
    private readonly DateTime _startTime;

    /// <summary>
    /// Gets the underlying SummaryWriter.
    /// </summary>
    public SummaryWriter Writer => _writer;

    /// <summary>
    /// Gets or sets the current global step.
    /// </summary>
    public long GlobalStep
    {
        get => _globalStep;
        set => _globalStep = value;
    }

    /// <summary>
    /// Creates a new training context.
    /// </summary>
    /// <param name="experimentName">Name of the experiment.</param>
    /// <param name="runName">Optional run name.</param>
    /// <param name="hparams">Optional hyperparameters to log.</param>
    public TensorBoardTrainingContext(
        string experimentName,
        string? runName = null,
        Dictionary<string, object>? hparams = null)
    {
        runName ??= DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var logDir = Path.Combine("runs", experimentName, runName);
        _writer = new SummaryWriter(logDir);
        _startTime = DateTime.Now;
        _globalStep = 0;

        // Log hyperparameters if provided
        if (hparams != null)
        {
            _writer.AddHparams(hparams, new Dictionary<string, float>());
        }

        // Log start time
        _writer.AddText("info/start_time", _startTime.ToString("yyyy-MM-dd HH:mm:ss"), 0);
    }

    /// <summary>
    /// Logs a training step with automatic step incrementing.
    /// </summary>
    public void LogTrainStep(float loss, float? accuracy = null, float? lr = null)
    {
        _writer.LogTrainingStep(loss, accuracy, lr, _globalStep++);
    }

    /// <summary>
    /// Logs a validation step (does not increment global step).
    /// </summary>
    public void LogValStep(float loss, float? accuracy = null)
    {
        _writer.LogValidationStep(loss, accuracy, _globalStep);
    }

    /// <summary>
    /// Logs model weights at current step.
    /// </summary>
    public void LogModelWeights(Dictionary<string, float[]> weights, Dictionary<string, float[]>? gradients = null)
    {
        foreach (var (name, w) in weights)
        {
            float[]? g = gradients?.GetValueOrDefault(name);
            _writer.LogWeights(name, w, g, _globalStep);
        }
    }

    /// <summary>
    /// Gets elapsed time since context creation.
    /// </summary>
    public TimeSpan Elapsed => DateTime.Now - _startTime;

    /// <summary>
    /// Logs elapsed time.
    /// </summary>
    public void LogElapsedTime()
    {
        _writer.AddScalar("info/elapsed_minutes", (float)Elapsed.TotalMinutes, _globalStep);
    }

    /// <summary>
    /// Releases resources.
    /// </summary>
    public void Dispose()
    {
        // Log final metrics
        _writer.AddText("info/end_time", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"), _globalStep);
        _writer.AddScalar("info/total_steps", _globalStep, _globalStep);
        _writer.AddScalar("info/total_minutes", (float)Elapsed.TotalMinutes, _globalStep);

        _writer.Dispose();
    }
}
