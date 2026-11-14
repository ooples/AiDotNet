using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// TabNet's sparsity regularizer, Arik and Pfister (2021) eq. 3:
/// <c>L = lambda * sum_steps sum_b sum_j -M log(M + eps) / (N_steps * B)</c>.
/// </summary>
public class TabNetSparsityLossTests
{
    [Fact]
    public void SparsityLoss_IsTheStepAndBatchMeanMaskEntropy()
    {
        var options = new TabNetOptions<double>
        {
            NumDecisionSteps = 3,
            FeatureDimension = 8,
            OutputDimension = 4,
            SparsityCoefficient = 0.5
        };
        var model = new TabNetRegression<double>(4, options);
        var input = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[] { 0.4, -0.7, 1.1, 0.2, 0.6, 0.9, -0.3, 0.5 }));
        _ = model.Predict(input);

        // The learned masks: every step after the first, uniform one has an attentive transformer.
        var masks = model.GetStepAttentionMasks().Skip(1).ToArray();
        Assert.Equal(2, masks.Length);
        double entropy = 0;
        foreach (var mask in masks)
        {
            var values = mask.ToArray();
            entropy += values.Sum(m => -m * Math.Log(m + 1e-15));
        }

        double expected = 0.5 * entropy / (masks.Length * 2);
        Assert.Equal(expected, model.ComputeSparsityLoss(), 10);
    }
}
