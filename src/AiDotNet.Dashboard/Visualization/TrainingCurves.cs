using SystemConsole = System.Console;

namespace AiDotNet.Dashboard.Visualization;

/// <summary>
/// Generates ASCII training curves for console visualization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Training curves show how your model's performance changes over time:
/// - Loss curves show the training and validation loss decreasing
/// - Accuracy curves show accuracy increasing
/// - Learning rate curves show scheduled changes
///
/// This class renders these curves as ASCII art in the console when a graphical
/// display isn't available.
///
/// Example:
/// <code>
/// var curves = new TrainingCurves();
/// curves.AddPoint("train_loss", epoch, trainLoss);
/// curves.AddPoint("val_loss", epoch, valLoss);
/// curves.Render();
/// </code>
/// </remarks>
public class TrainingCurves
{
    private readonly Dictionary<string, List<(double x, double y)>> _series;
    private readonly Dictionary<string, ConsoleColor> _colors;
    private readonly object _lock = new();

    private int _width;
    private int _height;
    private string _title;
    private string _xLabel;
    private string _yLabel;

    /// <summary>
    /// Gets or sets whether to auto-scale the Y axis.
    /// </summary>
    public bool AutoScaleY { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum Y value for the chart.
    /// </summary>
    public double? MinY { get; set; }

    /// <summary>
    /// Gets or sets the maximum Y value for the chart.
    /// </summary>
    public double? MaxY { get; set; }

    /// <summary>
    /// Initializes a new instance of the TrainingCurves class.
    /// </summary>
    /// <param name="width">Chart width in characters.</param>
    /// <param name="height">Chart height in characters.</param>
    /// <param name="title">Chart title.</param>
    public TrainingCurves(int width = 80, int height = 20, string title = "Training Progress")
    {
        _width = width;
        _height = height;
        _title = title;
        _xLabel = "Epoch";
        _yLabel = "Value";
        _series = new Dictionary<string, List<(double, double)>>();
        _colors = new Dictionary<string, ConsoleColor>
        {
            ["train_loss"] = ConsoleColor.Blue,
            ["val_loss"] = ConsoleColor.Yellow,
            ["train_acc"] = ConsoleColor.Green,
            ["val_acc"] = ConsoleColor.Cyan,
            ["learning_rate"] = ConsoleColor.Magenta
        };
    }

    /// <summary>
    /// Sets the chart title.
    /// </summary>
    public TrainingCurves WithTitle(string title)
    {
        _title = title;
        return this;
    }

    /// <summary>
    /// Sets the X axis label.
    /// </summary>
    public TrainingCurves WithXLabel(string label)
    {
        _xLabel = label;
        return this;
    }

    /// <summary>
    /// Sets the Y axis label.
    /// </summary>
    public TrainingCurves WithYLabel(string label)
    {
        _yLabel = label;
        return this;
    }

    /// <summary>
    /// Sets the chart dimensions.
    /// </summary>
    public TrainingCurves WithSize(int width, int height)
    {
        _width = width;
        _height = height;
        return this;
    }

    /// <summary>
    /// Adds a data point to a series.
    /// </summary>
    /// <param name="seriesName">The name of the series (e.g., "train_loss").</param>
    /// <param name="x">The X value (typically epoch or step).</param>
    /// <param name="y">The Y value (the metric value).</param>
    public void AddPoint(string seriesName, double x, double y)
    {
        if (string.IsNullOrWhiteSpace(seriesName))
            throw new ArgumentException("Series name cannot be null or empty.", nameof(seriesName));

        lock (_lock)
        {
            if (!_series.ContainsKey(seriesName))
            {
                _series[seriesName] = new List<(double, double)>();
            }
            _series[seriesName].Add((x, y));
        }
    }

    /// <summary>
    /// Adds multiple points to a series.
    /// </summary>
    public void AddPoints(string seriesName, IEnumerable<(double x, double y)> points)
    {
        if (string.IsNullOrWhiteSpace(seriesName))
            throw new ArgumentException("Series name cannot be null or empty.", nameof(seriesName));

        lock (_lock)
        {
            if (!_series.ContainsKey(seriesName))
            {
                _series[seriesName] = new List<(double, double)>();
            }
            _series[seriesName].AddRange(points);
        }
    }

    /// <summary>
    /// Sets a custom color for a series.
    /// </summary>
    public void SetSeriesColor(string seriesName, ConsoleColor color)
    {
        lock (_lock)
        {
            _colors[seriesName] = color;
        }
    }

    /// <summary>
    /// Clears all data from the chart.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _series.Clear();
        }
    }

    /// <summary>
    /// Clears data from a specific series.
    /// </summary>
    public void ClearSeries(string seriesName)
    {
        lock (_lock)
        {
            _series.Remove(seriesName);
        }
    }

    /// <summary>
    /// Renders the chart to the console.
    /// </summary>
    public void Render()
    {
        lock (_lock)
        {
            if (_series.Count == 0)
            {
                SystemConsole.WriteLine("No data to display.");
                return;
            }

            // Calculate bounds
            var allPoints = _series.Values.SelectMany(s => s).ToList();
            if (allPoints.Count == 0)
            {
                SystemConsole.WriteLine("No data points to display.");
                return;
            }

            double minX = allPoints.Min(p => p.x);
            double maxX = allPoints.Max(p => p.x);
            double minY = MinY ?? (AutoScaleY ? allPoints.Min(p => p.y) : 0);
            double maxY = MaxY ?? allPoints.Max(p => p.y);

            // Add padding to Y range
            var yRange = maxY - minY;
            if (yRange < 0.0001)
            {
                yRange = 1;
                minY -= 0.5;
                maxY += 0.5;
            }

            minY -= yRange * 0.05;
            maxY += yRange * 0.05;

            // Create the canvas
            var canvas = new char[_height, _width];
            var colorMap = new ConsoleColor?[_height, _width];

            // Initialize with spaces
            for (int row = 0; row < _height; row++)
            {
                for (int col = 0; col < _width; col++)
                {
                    canvas[row, col] = ' ';
                }
            }

            // Draw axes
            int yAxisCol = 8; // Leave room for Y labels
            int xAxisRow = _height - 3; // Leave room for X labels

            // Y axis
            for (int row = 0; row < xAxisRow; row++)
            {
                canvas[row, yAxisCol] = '|';
            }

            // X axis
            for (int col = yAxisCol; col < _width; col++)
            {
                canvas[xAxisRow, col] = '-';
            }

            canvas[xAxisRow, yAxisCol] = '+';

            // Plot data points
            int plotWidth = _width - yAxisCol - 2;
            int plotHeight = xAxisRow - 1;

            foreach (var kvp in _series)
            {
                var seriesName = kvp.Key;
                var points = kvp.Value;
                var color = _colors.GetValueOrDefault(seriesName, ConsoleColor.White);

                for (int i = 0; i < points.Count; i++)
                {
                    var (x, y) = points[i];

                    // Map to canvas coordinates
                    int col = yAxisCol + 1 + (int)((x - minX) / (maxX - minX + 0.0001) * (plotWidth - 1));
                    int row = xAxisRow - 1 - (int)((y - minY) / (maxY - minY + 0.0001) * (plotHeight - 1));

                    // Clamp to bounds
                    col = Math.Max(yAxisCol + 1, Math.Min(col, _width - 1));
                    row = Math.Max(0, Math.Min(row, xAxisRow - 1));

                    // Draw point
                    char marker = GetSeriesMarker(seriesName);
                    canvas[row, col] = marker;
                    colorMap[row, col] = color;

                    // Connect with previous point
                    if (i > 0)
                    {
                        var (prevX, prevY) = points[i - 1];
                        int prevCol = yAxisCol + 1 + (int)((prevX - minX) / (maxX - minX + 0.0001) * (plotWidth - 1));
                        int prevRow = xAxisRow - 1 - (int)((prevY - minY) / (maxY - minY + 0.0001) * (plotHeight - 1));

                        DrawLine(canvas, colorMap, prevRow, prevCol, row, col, color);
                    }
                }
            }

            // Render to console
            SystemConsole.WriteLine();
            SystemConsole.WriteLine($"  {_title}");
            SystemConsole.WriteLine();

            for (int row = 0; row < _height; row++)
            {
                // Y axis label
                if (row == 0)
                {
                    SystemConsole.Write($"{maxY,7:F3} ");
                }
                else if (row == xAxisRow)
                {
                    SystemConsole.Write($"{minY,7:F3} ");
                }
                else if (row == xAxisRow / 2)
                {
                    double midY = (minY + maxY) / 2;
                    SystemConsole.Write($"{midY,7:F3} ");
                }
                else
                {
                    SystemConsole.Write("        ");
                }

                // Draw row
                for (int col = 0; col < _width; col++)
                {
                    var color = colorMap[row, col];
                    if (color.HasValue)
                    {
                        var originalColor = SystemConsole.ForegroundColor;
                        SystemConsole.ForegroundColor = color.Value;
                        SystemConsole.Write(canvas[row, col]);
                        SystemConsole.ForegroundColor = originalColor;
                    }
                    else
                    {
                        SystemConsole.Write(canvas[row, col]);
                    }
                }
                SystemConsole.WriteLine();
            }

            // X axis labels
            SystemConsole.Write("        ");
            SystemConsole.Write($"{minX,-10:F0}");
            var midXPos = yAxisCol + plotWidth / 2 - 10;
            SystemConsole.Write(new string(' ', Math.Max(0, midXPos - 10)));
            SystemConsole.Write($"{(minX + maxX) / 2:F0}");
            SystemConsole.Write(new string(' ', Math.Max(0, plotWidth - midXPos - 10)));
            SystemConsole.WriteLine($"{maxX:F0}");

            SystemConsole.WriteLine($"        {new string(' ', plotWidth / 2 - _xLabel.Length / 2)}{_xLabel}");

            // Legend
            SystemConsole.WriteLine();
            SystemConsole.Write("  Legend: ");
            foreach (var seriesName in _series.Keys)
            {
                var color = _colors.GetValueOrDefault(seriesName, ConsoleColor.White);
                var marker = GetSeriesMarker(seriesName);
                var originalColor = SystemConsole.ForegroundColor;
                SystemConsole.ForegroundColor = color;
                SystemConsole.Write($"{marker} {seriesName}  ");
                SystemConsole.ForegroundColor = originalColor;
            }
            SystemConsole.WriteLine();
            SystemConsole.WriteLine();
        }
    }

    /// <summary>
    /// Gets the marker character for a series.
    /// </summary>
    private static char GetSeriesMarker(string seriesName)
    {
        return seriesName.ToLowerInvariant() switch
        {
            string s when s.Contains("train") && s.Contains("loss") => '*',
            string s when s.Contains("val") && s.Contains("loss") => 'o',
            string s when s.Contains("train") && s.Contains("acc") => '+',
            string s when s.Contains("val") && s.Contains("acc") => 'x',
            string s when s.Contains("lr") || s.Contains("learning") => '^',
            _ => '#'
        };
    }

    /// <summary>
    /// Draws a line between two points using Bresenham's algorithm.
    /// </summary>
    private void DrawLine(char[,] canvas, ConsoleColor?[,] colorMap, int r1, int c1, int r2, int c2, ConsoleColor color)
    {
        int dr = Math.Abs(r2 - r1);
        int dc = Math.Abs(c2 - c1);
        int sr = r1 < r2 ? 1 : -1;
        int sc = c1 < c2 ? 1 : -1;
        int err = dc - dr;

        int r = r1, c = c1;

        while (true)
        {
            if (r >= 0 && r < _height && c >= 0 && c < _width)
            {
                if (canvas[r, c] == ' ')
                {
                    canvas[r, c] = '.';
                    colorMap[r, c] = color;
                }
            }

            if (r == r2 && c == c2)
                break;

            int e2 = 2 * err;
            if (e2 > -dr)
            {
                err -= dr;
                c += sc;
            }
            if (e2 < dc)
            {
                err += dc;
                r += sr;
            }
        }
    }

    /// <summary>
    /// Generates an ASCII summary of the latest metrics.
    /// </summary>
    public string GetSummary()
    {
        lock (_lock)
        {
            var lines = new List<string> { $"=== {_title} ===" };

            foreach (var kvp in _series)
            {
                if (kvp.Value.Count > 0)
                {
                    var latest = kvp.Value.Last();
                    var min = kvp.Value.Min(p => p.y);
                    var max = kvp.Value.Max(p => p.y);
                    lines.Add($"  {kvp.Key}: {latest.y:F4} (min: {min:F4}, max: {max:F4})");
                }
            }

            return string.Join(Environment.NewLine, lines);
        }
    }
}
