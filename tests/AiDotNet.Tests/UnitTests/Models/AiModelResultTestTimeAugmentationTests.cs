using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Augmentation;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Test-time augmentation aggregation on <see cref="AiModelResult{T, TInput, TOutput}"/>, pinned where the
/// engine-op conversion (#2490) fixed a wrong result.
/// </summary>
public class AiModelResultTestTimeAugmentationTests
{
    [Fact]
    public void PredictWithTestTimeAugmentation_Half_AggregatesInT()
    {
        // y = 1 * x0 + 2 * x1. The original input [1, 1] predicts 3 and the augmented input [2, 2] predicts 6,
        // so the mean is 4.5 and the population standard deviation is 1.5. Half is not IConvertible, so the
        // former Convert.ToDouble aggregation threw InvalidCastException.
        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)1, (Half)2 }));
        var result = new AiModelResult<Half, Matrix<Half>, Vector<Half>>(
            new AiModelResultOptions<Half, Matrix<Half>, Vector<Half>>
            {
                OptimizationResult = new OptimizationResult<Half, Matrix<Half>, Vector<Half>> { BestSolution = model },
                AugmentationConfig = new AugmentationConfig
                {
                    EnableTTA = true,
                    TTAIncludeOriginal = true,
                    TTANumAugmentations = 1,
                    TTAAggregation = PredictionAggregationMethod.Mean
                }
            });
        var input = new Matrix<Half>(new[,] { { (Half)1, (Half)1 } });
        var augmented = new Matrix<Half>(new[,] { { (Half)2, (Half)2 } });

        var tta = result.PredictWithTestTimeAugmentation(input, new FixedPolicy(augmented));

        Assert.Equal(2, tta.IndividualPredictions.Count);
        Assert.Equal((Half)4.5, tta.AggregatedPrediction[0]);
        Assert.NotNull(tta.Confidence);
        Assert.Equal(1.0 / (1.0 + 1.5), tta.Confidence.Value, 1e-3);
    }

    [Fact]
    public void PredictWithTestTimeAugmentation_HalfWeightedMean_WeightsBeyondHalfRange()
    {
        // y = 1 * x0 + 2 * x1: [10000, 10000] predicts 30000 and [12000, 12000] predicts 36000. Each prediction is
        // weighted by its largest value, so the weighted mean is (30000^2 + 36000^2) / 66000 = 33272.7. The weight
        // total 66000 overflows Half (65504), which made every normalised weight 0 and the aggregate 0.
        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)1, (Half)2 }));
        var result = new AiModelResult<Half, Matrix<Half>, Vector<Half>>(
            new AiModelResultOptions<Half, Matrix<Half>, Vector<Half>>
            {
                OptimizationResult = new OptimizationResult<Half, Matrix<Half>, Vector<Half>> { BestSolution = model },
                AugmentationConfig = new AugmentationConfig
                {
                    EnableTTA = true,
                    TTAIncludeOriginal = true,
                    TTANumAugmentations = 1,
                    TTAAggregation = PredictionAggregationMethod.WeightedMean
                }
            });
        var input = new Matrix<Half>(new[,] { { (Half)10000, (Half)10000 } });
        var augmented = new Matrix<Half>(new[,] { { (Half)12000, (Half)12000 } });

        var tta = result.PredictWithTestTimeAugmentation(input, new FixedPolicy(augmented));

        Assert.Equal(33272.7, (double)tta.AggregatedPrediction[0], 64.0);
    }

    [Fact]
    public void ComputeMedian_ColumnWithNaN_IsNaN()
    {
        // numpy.median([[1, nan], [2, 5], [3, 6]], axis=0) is [2, nan]: any NaN in a column makes that element NaN.
        // Sorting NaN to one end and taking the middle returned a number (5 or 6) for the second column.
        using var model = new VectorModel<double>(new Vector<double>(new[] { 1.0 }));
        var result = new AiModelResult<double, Matrix<double>, Vector<double>>(
            new AiModelResultOptions<double, Matrix<double>, Vector<double>>
            {
                OptimizationResult = new OptimizationResult<double, Matrix<double>, Vector<double>> { BestSolution = model }
            });
        var predictions = new List<Vector<double>>
        {
            new Vector<double>(new[] { 1.0, double.NaN }),
            new Vector<double>(new[] { 2.0, 5.0 }),
            new Vector<double>(new[] { 3.0, 6.0 })
        };
        var method = result.GetType().GetMethod("ComputeMedian", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);

        var median = Assert.IsType<Vector<double>>(method?.Invoke(result, new object[] { predictions }));

        Assert.Equal(2.0, median[0]);
        Assert.True(double.IsNaN(median[1]));
    }

    /// <summary>An augmentation policy that always returns the same input.</summary>
    private sealed class FixedPolicy : IAugmentationPolicy<Half, Matrix<Half>>
    {
        private readonly Matrix<Half> _output;

        public FixedPolicy(Matrix<Half> output)
        {
            _output = output;
        }

        public string Name => "fixed";

        public int AugmentationCount => 0;

        public IList<string> AugmentationNames { get; } = new List<string>();

        public IList<IAugmentation<Half, Matrix<Half>>> Augmentations { get; } = new List<IAugmentation<Half, Matrix<Half>>>();

        public Matrix<Half> Apply(Matrix<Half> data, AugmentationContext<Half>? context = null) => _output;

        public IDictionary<string, object> GetConfiguration() => new Dictionary<string, object>();
    }
}
