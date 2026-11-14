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

    [Fact]
    [Trait("category", "unit")]
    public void Probabilities_average_leaf_distributions_rather_than_votes()
    {
        // Four identical series cannot be split, so every tree is one leaf holding the class distribution 3:1. Averaged
        // leaf distributions give 0.75 / 0.25; averaged hard votes would give 1 / 0.
        const int samples = 4, length = 12;
        var sequences = new Tensor<double>(new[] { samples, length });
        for (int s = 0; s < samples; s++)
            for (int t = 0; t < length; t++)
                sequences[s, t] = Math.Sin(0.5 * t);

        var classifier = new TimeSeriesForestClassifier<double>(new TimeSeriesForestOptions<double> { NumTrees = 3, Seed = 2 });
        classifier.TrainOnSequences(sequences, new Vector<double>(new[] { 0.0, 0, 0, 1 }));

        var probabilities = classifier.PredictSequenceProbabilities(sequences);
        for (int s = 0; s < samples; s++)
        {
            Assert.Equal(0.75, probabilities[s, 0], Tolerance);
            Assert.Equal(0.25, probabilities[s, 1], Tolerance);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Single_channel_three_dimensional_input_gives_the_same_features()
    {
        const int samples = 3, length = 12;
        var flat = Series(samples, length);
        var cube = new Tensor<double>(new[] { samples, length, 1 });
        for (int s = 0; s < samples; s++)
            for (int t = 0; t < length; t++)
                cube[s, t, 0] = flat[s, t];

        var options = new TimeSeriesForestOptions<double> { NumTrees = 2, Seed = 4 };
        var labels = new Vector<double>(new[] { 0.0, 1, 0 });
        var fromFlat = new TimeSeriesForestClassifier<double>(options);
        fromFlat.TrainOnSequences(flat, labels);
        var fromCube = new TimeSeriesForestClassifier<double>(options);
        fromCube.TrainOnSequences(cube, labels);

        var expected = fromFlat.ExtractIntervalFeatures(flat, 2, 9, 0);
        var actual = fromCube.ExtractIntervalFeatures(cube, 2, 9, 0);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal(fromFlat.PredictSequences(flat).ToArray(), fromCube.PredictSequences(cube).ToArray());
    }

    [Fact]
    [Trait("category", "unit")]
    public void A_midpoint_that_rounds_onto_the_upper_value_still_splits()
    {
        // 1 + 2^-23 and 1 + 2^-22 are adjacent floats; their midpoint rounds (ties to even) onto the upper one, which
        // would send both series left. The threshold falls back to the lower value, as scikit-learn's splitter does.
        // BitConverter.Int32BitsToSingle/SingleToInt32Bits do not exist on net471, which this project also targets.
        int oneBits = BitConverter.ToInt32(BitConverter.GetBytes(1f), 0);
        float low = BitConverter.ToSingle(BitConverter.GetBytes(oneBits + 1), 0);
        float high = BitConverter.ToSingle(BitConverter.GetBytes(oneBits + 2), 0);
        const int length = 4;
        var sequences = new Tensor<float>(new[] { 2, length });
        for (int t = 0; t < length; t++)
        {
            sequences[0, t] = low;
            sequences[1, t] = high;
        }

        var classifier = new TimeSeriesForestClassifier<float>(new TimeSeriesForestOptions<float> { NumTrees = 1, Seed = 1 });
        var labels = new Vector<float>(new[] { 0f, 1f });
        classifier.TrainOnSequences(sequences, labels);

        Assert.Equal(labels.ToArray(), classifier.PredictSequences(sequences).ToArray());
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_leaf_distribution_counts_more_rows_than_half_can_count()
    {
        // MinSamplesSplit above the row count makes every tree one leaf holding all 3000 rows, 2500 : 500. Counting the
        // classes as a 0/1 sum in Half stopped at 2048, giving 2048 / 3000 = 0.683 for the larger class.
        const int samples = 3000, length = 4;
        var sequences = new Tensor<Half>(new[] { samples, length });
        var labels = new Vector<Half>(samples);
        for (int s = 0; s < samples; s++)
        {
            labels[s] = s < 2500 ? (Half)0 : (Half)1;
            for (int t = 0; t < length; t++)
                sequences[s, t] = (Half)((s % 7) + t);
        }

        var classifier = new TimeSeriesForestClassifier<Half>(new TimeSeriesForestOptions<Half>
        {
            NumTrees = 1, Seed = 3, MinSamplesSplit = samples + 1
        });
        classifier.TrainOnSequences(sequences, labels);

        var probabilities = classifier.PredictSequenceProbabilities(sequences);
        for (int s = 0; s < samples; s += 499)
        {
            Assert.Equal(2500.0 / 3000.0, (double)probabilities[s, 0], 1e-3);
            Assert.Equal(500.0 / 3000.0, (double)probabilities[s, 1], 1e-3);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_split_search_offers_no_threshold_between_equal_values()
    {
        // Feature 0 is constant, so it cannot split. The 1e-10 "distinct values" guard rounds to zero in Half, which let
        // every gap between its equal values score as a split (gain up to 1 bit, at least the 0.311 feature 1 offers)
        // and won on the earlier feature. Feature 1's only split, at 0.5, must be chosen.
        var x = new Tensor<Half>(new[] { 4, 2 });
        var oneHot = new Tensor<Half>(new[] { 4, 2 });
        Half[] second = { (Half)0, (Half)1, (Half)1, (Half)1 };
        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = (Half)1;
            x[i, 1] = second[i];
            oneHot[i, i < 2 ? 0 : 1] = (Half)1;
        }

        var (feature, threshold, found) = TimeSeriesForestClassifier<Half>.FindBestSplit(x, oneHot);

        Assert.True(found);
        Assert.Equal(1, feature);
        Assert.Equal(0.5, (double)threshold);
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
