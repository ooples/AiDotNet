using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using AiDotNet.Classification.Boosting;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PyTorchParity;

/// <summary>
/// The classical-ML rows (#2395): AiDotNet's trees, linear models and k-means against scikit-learn, XGBoost and
/// LightGBM on one dataset. This side writes the data as CSV (<c>train.csv</c>, <c>test.csv</c>) and
/// <c>pytorch/classical.py</c> reads it, so both fit exactly the same numbers. Each row reports the median and
/// interquartile range of the fit and predict times over the repeats, and the held-out quality, so a faster
/// fit that learned less is visible as such.
/// </summary>
internal static class ClassicalBench
{
    private const int Features = 20;
    private const int TrainRows = 20_000;
    private const int TestRows = 5_000;

    public static int Run(string outputDirectory, int repeats, int seed)
    {
        Directory.CreateDirectory(outputDirectory);
        var (train, test) = WriteData(outputDirectory, seed);

        var rows = new List<Dictionary<string, object>>
        {
            Classifier("random_forest", train, test, repeats,
                () => new RandomForestClassifier<double>(new RandomForestClassifierOptions<double> { NEstimators = 100, Seed = seed })),
            Classifier("hist_gradient_boosting", train, test, repeats,
                () => new HistGradientBoostingClassifier<double>(maxBins: 256, maxDepth: 6, nEstimators: 100,
                    learningRate: 0.1, minSamplesLeaf: 20, seed: seed)),
            Regressor("logistic_regression", train, test, repeats, classify: true,
                () => new LogisticRegression<double>(new LogisticRegressionOptions<double>())),
            Regressor("ridge", train, test, repeats, classify: false,
                () => new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1.0 })),
            KMeansRow(train, repeats, seed),
        };

        string path = Path.Combine(outputDirectory, "aidotnet_classical.json");
        File.WriteAllText(path, JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["framework"] = "AiDotNet",
            ["repeats"] = repeats,
            ["train_rows"] = TrainRows,
            ["test_rows"] = TestRows,
            ["features"] = Features,
            ["rows"] = rows,
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"[classical] report written to {path}");
        return 0;
    }

    private sealed record Split(Matrix<double> X, Vector<double> Class, Vector<double> Target);

    /// <summary>
    /// Gaussian features; a class label from a nonlinear rule (an interaction, a sine and a square, so trees beat a
    /// linear model on it) with 10% label noise, and a regression target linear in the features plus noise.
    /// </summary>
    private static (Split Train, Split Test) WriteData(string directory, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        double Gaussian() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());

        Split Make(int rows, string file)
        {
            var x = new Matrix<double>(rows, Features);
            var label = new Vector<double>(rows);
            var target = new Vector<double>(rows);
            var csv = new StringBuilder("class,target," + string.Join(",", Enumerable.Range(0, Features).Select(i => $"x{i}")) + "\n");
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < Features; c++) x[r, c] = Gaussian();
                double score = x[r, 0] * x[r, 1] + Math.Sin(2.0 * x[r, 2]) + 0.5 * x[r, 3] - 0.5 * x[r, 4] * x[r, 4] + 0.5;
                bool positive = score > 0;
                if (random.NextDouble() < 0.1) positive = !positive;
                label[r] = positive ? 1.0 : 0.0;
                double y = 0.0;
                for (int c = 0; c < Features; c++) y += (c % 3 - 1) * 0.5 * x[r, c];
                target[r] = y + 0.1 * Gaussian();

                csv.Append(label[r].ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(target[r].ToString("R", CultureInfo.InvariantCulture));
                for (int c = 0; c < Features; c++)
                    csv.Append(',').Append(x[r, c].ToString("R", CultureInfo.InvariantCulture));
                csv.Append('\n');
            }

            File.WriteAllText(Path.Combine(directory, file), csv.ToString());
            return new Split(x, label, target);
        }

        return (Make(TrainRows, "train.csv"), Make(TestRows, "test.csv"));
    }

    private static Dictionary<string, object> Classifier<TModel>(string name, Split train, Split test, int repeats,
        Func<TModel> create) where TModel : AiDotNet.Classification.ClassifierBase<double>
    {
        Console.WriteLine($"[classical] {name}: fitting {repeats} time(s)...");
        var fit = new List<double>();
        var predict = new List<double>();
        double accuracy = 0;
        for (int i = 0; i < repeats; i++)
        {
            var model = create();
            var clock = Stopwatch.StartNew();
            model.Train(train.X, train.Class);
            fit.Add(clock.Elapsed.TotalSeconds);
            clock.Restart();
            var predicted = model.Predict(test.X);
            predict.Add(clock.Elapsed.TotalSeconds);
            accuracy = Accuracy(predicted, test.Class);
        }

        return Row(name, fit, predict, "accuracy", accuracy);
    }

    private static Dictionary<string, object> Regressor<TModel>(string name, Split train, Split test, int repeats,
        bool classify, Func<TModel> create) where TModel : RegressionBase<double>
    {
        Console.WriteLine($"[classical] {name}: fitting {repeats} time(s)...");
        var fit = new List<double>();
        var predict = new List<double>();
        double quality = 0;
        for (int i = 0; i < repeats; i++)
        {
            var model = create();
            var clock = Stopwatch.StartNew();
            model.Train(train.X, classify ? train.Class : train.Target);
            fit.Add(clock.Elapsed.TotalSeconds);
            clock.Restart();
            var predicted = model.Predict(test.X);
            predict.Add(clock.Elapsed.TotalSeconds);
            if (classify)
            {
                // The logistic model predicts P(class 1); the class is the side of 0.5 it falls on.
                var labels = new Vector<double>(predicted.Length);
                for (int r = 0; r < predicted.Length; r++) labels[r] = predicted[r] >= 0.5 ? 1.0 : 0.0;
                quality = Accuracy(labels, test.Class);
            }
            else
            {
                quality = RSquared(predicted, test.Target);
            }
        }

        return Row(name, fit, predict, classify ? "accuracy" : "r2", quality);
    }

    private static Dictionary<string, object> KMeansRow(Split train, int repeats, int seed)
    {
        Console.WriteLine($"[classical] kmeans: fitting {repeats} time(s)...");
        var fit = new List<double>();
        double inertia = 0;
        for (int i = 0; i < repeats; i++)
        {
            var model = new KMeans<double>(new KMeansOptions<double> { NumClusters = 8, NumInitializations = 1, Seed = seed + i });
            var clock = Stopwatch.StartNew();
            model.Train(train.X);
            fit.Add(clock.Elapsed.TotalSeconds);
            inertia = model.Inertia;
        }

        return Row("kmeans", fit, [], "inertia", inertia);
    }

    private static double Accuracy(Vector<double> predicted, Vector<double> actual)
    {
        int correct = 0;
        for (int i = 0; i < actual.Length; i++)
            if (Math.Abs(predicted[i] - actual[i]) < 0.5) correct++;
        return (double)correct / actual.Length;
    }

    private static double RSquared(Vector<double> predicted, Vector<double> actual)
    {
        double mean = 0;
        for (int i = 0; i < actual.Length; i++) mean += actual[i];
        mean /= actual.Length;
        double residual = 0, total = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            residual += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
            total += (actual[i] - mean) * (actual[i] - mean);
        }

        return 1.0 - residual / total;
    }

    private static Dictionary<string, object> Row(string name, List<double> fit, List<double> predict, string metric, double value)
    {
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "[classical] {0}: fit median {1:F3} s, {2}{3} {4:F4}", name, Quantile(fit, 0.5),
            predict.Count > 0
                ? string.Format(CultureInfo.InvariantCulture, "predict median {0:F3} s, ", Quantile(predict, 0.5))
                : string.Empty,
            metric, value));
        var row = new Dictionary<string, object>
        {
            ["model"] = name,
            ["fit_seconds_median"] = Quantile(fit, 0.5),
            ["fit_seconds_iqr"] = Quantile(fit, 0.75) - Quantile(fit, 0.25),
            ["metric"] = metric,
            ["value"] = value,
        };
        if (predict.Count > 0)
        {
            row["predict_seconds_median"] = Quantile(predict, 0.5);
            row["predict_seconds_iqr"] = Quantile(predict, 0.75) - Quantile(predict, 0.25);
        }

        return row;
    }

    // Linear interpolation between order statistics, as benchmark.py's quantile and numpy's default.
    private static double Quantile(List<double> values, double q)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        double position = q * (sorted.Length - 1);
        int low = (int)position;
        int high = Math.Min(low + 1, sorted.Length - 1);
        return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
    }
}
