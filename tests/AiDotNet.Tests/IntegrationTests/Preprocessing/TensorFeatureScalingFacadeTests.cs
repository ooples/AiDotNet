using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// TensorStandardScaler: feature scaling for Tensor-input facade builders. StandardScaler only speaks
/// Matrix, so neural-network (Tensor) builders previously had NO facade route for feature scaling —
/// every consumer hand-rolled it outside the facade (and typically fit the scaler on ALL rows, leaking
/// val/test statistics into training).
/// </summary>
public class TensorFeatureScalingFacadeTests
{
    [Fact]
    public void Scales_each_column_independently_and_round_trips()
    {
        // Two columns on wildly different scales (the RSI-vs-MACD shape of real indicator features).
        var data = new double[] { 10, 1000, 20, 2000, 30, 3000, 40, 4000 };
        var t = new Tensor<double>(new[] { 4, 2 }, new Vector<double>(data));

        var scaler = new TensorStandardScaler<double>();
        var scaled = scaler.FitTransform(t);

        for (int c = 0; c < 2; c++)
        {
            double mean = 0;
            for (int r = 0; r < 4; r++)
            {
                mean += scaled.Data.Span[(r * 2) + c];
            }

            Assert.True(Math.Abs(mean / 4) < 1e-9, $"column {c} not centered after scaling");
        }

        var back = scaler.InverseTransform(scaled);
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], back.Data.Span[i], 6);
        }
    }

    [Fact]
    public async Task Facade_tensor_builder_scales_features_and_predicts_on_raw_inputs()
    {
        // y depends on a feature with scale 1000: unscaled, gradient training diverges/stalls; with
        // facade-fitted scaling the net learns and Predict accepts RAW (unscaled) inputs because the
        // result applies the fitted pipeline internally.
        const int n = 100;
        var x = new double[n];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double raw = 1000.0 * i / n; // feature on a 0..1000 scale
            x[i] = raw;
            y[i] = (raw / 500.0) - 1.0;  // y in [-1, 1)
        }

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple,
                inputSize: 1,
                outputSize: 1)))
            .ConfigureDataLoader(DataLoaders.FromTensors(
                new Tensor<double>(new[] { n, 1 }, new Vector<double>(x)),
                new Tensor<double>(new[] { n, 1 }, new Vector<double>(y))))
            .ConfigurePreprocessing(new TensorStandardScaler<double>())
            .ConfigureOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                model: null,
                options: new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { MaxIterations = 250 }))
            .BuildAsync();

        var probes = new[] { 100.0, 300.0, 500.0, 700.0, 900.0 }; // RAW units
        var preds = probes
            .Select(p => result.Predict(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { p })))[0])
            .ToArray();

        int rising = 0;
        for (int i = 1; i < preds.Length; i++)
        {
            if (preds[i] > preds[i - 1])
            {
                rising++;
            }
        }

        Assert.True(preds.Max() - preds.Min() > 0.3 && rising >= 3,
            "Tensor feature scaling through the facade did not produce a usable fit on raw-unit inputs. " +
            $"preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");
    }
}
