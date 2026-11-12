using System;
using AiDotNet.Classification.TimeSeries;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Classification;

/// <summary>
/// Reference checks for the Time Series Forest (Deng et al. 2013; sktime TimeSeriesForestClassifier).
/// </summary>
public class TimeSeriesForestReferenceTests
{
    private const double Tolerance = 1e-10;

    [Fact]
    [Trait("category", "unit")]
    public void Interval_features_are_mean_population_std_and_least_squares_slope()
    {
        const int samples = 3, length = 12;
        var sequences = Series(samples, length);
        var classifier = new TimeSeriesForestClassifier<double>(new TimeSeriesForestOptions<double> { NumTrees = 2 });
        classifier.TrainOnSequences(sequences, new Vector<double>(new[] { 0.0, 1, 0 }));

        const int start = 2, end = 9;
        var features = classifier.ExtractIntervalFeatures(sequences, start, end, 0);

        for (int s = 0; s < samples; s++)
        {
            // numpy: mean, std (ddof=0) and np.polyfit(arange(n), y, 1)[0] over the interval.
            int n = end - start;
            double mean = 0;
            for (int t = start; t < end; t++) mean += sequences[s, t];
            mean /= n;
            double variance = 0, sxy = 0, sxx = 0;
            for (int t = start; t < end; t++)
            {
                double x = t - start - ((n - 1) / 2.0);
                variance += (sequences[s, t] - mean) * (sequences[s, t] - mean);
                sxy += x * (sequences[s, t] - mean);
                sxx += x * x;
            }

            Assert.Equal(mean, features[s, 0], Tolerance);
            Assert.Equal(Math.Sqrt(variance / n), features[s, 1], Tolerance);
            Assert.Equal(sxy / sxx, features[s, 2], Tolerance);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Each_tree_summarizes_sqrt_m_intervals_and_probabilities_are_averaged_leaf_distributions()
    {
        const int samples = 8, length = 30;
        var sequences = Series(samples, length);
        var labels = new Vector<double>(new[] { 0.0, 1, 2, 0, 1, 2, 0, 1 });
        var classifier = new TimeSeriesForestClassifier<double>(new TimeSeriesForestOptions<double> { NumTrees = 7, Seed = 5 });
        classifier.TrainOnSequences(sequences, labels);

        // Deng et al. 2013: sqrt(m) intervals per tree, floor(sqrt(30)) = 5.
        Assert.Equal(5, classifier.IntervalsPerTree);

        var probabilities = classifier.PredictSequenceProbabilities(sequences);
        var predictions = classifier.PredictSequences(sequences);
        for (int s = 0; s < samples; s++)
        {
            double sum = 0;
            int best = 0;
            for (int c = 0; c < probabilities.Columns; c++)
            {
                sum += probabilities[s, c];
                if (probabilities[s, c] > probabilities[s, best]) best = c;
            }

            Assert.Equal(1.0, sum, Tolerance);
            Assert.Equal(best, predictions[s]);
        }
    }

    private static Tensor<double> Series(int samples, int length)
    {
        var sequences = new Tensor<double>(new[] { samples, length });
        for (int s = 0; s < samples; s++)
        {
            for (int t = 0; t < length; t++)
            {
                sequences[s, t] = Math.Sin((0.3 * t) + s) + (0.05 * s * t) + (s % 3);
            }
        }

        return sequences;
    }
}
