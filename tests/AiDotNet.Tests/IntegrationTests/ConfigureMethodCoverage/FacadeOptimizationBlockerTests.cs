using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Regression tests for the three facade-optimization blockers:
/// <list type="number">
/// <item>ConfigureMemoryManagement stays concrete-only (documented compat policy); verify it chains fluently on
/// the concrete builder so consumers casting to it can chain the call.</item>
/// <item>Quantization must preserve a model's trained state for families (e.g. gradient-boosted trees) that keep
/// trained internals outside the parameter vector — previously Predict threw "Model must be trained".</item>
/// <item>The facade's supervised build must train a rank-3 tensor forecasting model (previously the optimizer's
/// "non-flat" guard misclassified Dense-embedding forecasters and crashed in GetColumnVectors).</item>
/// </list>
/// </summary>
public sealed class FacadeOptimizationBlockerTests
{
    private static bool IsFiniteValue(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void ConfigureMemoryManagement_chains_fluently_on_the_concrete_builder()
    {
        // ConfigureMemoryManagement is intentionally concrete-only (see the IAiModelBuilder note + the
        // completeness test's allowlist). Verify it honours the fluent contract — returns the SAME builder
        // instance — so a consumer that casts to the concrete type can chain it.
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var chained = builder.ConfigureMemoryManagement(
            global::AiDotNet.Training.Memory.TrainingMemoryConfig.ForTransformers());
        Assert.Same(builder, chained);
    }

    [Fact(Timeout = 120_000)]
    [Trait("category", "integration-configure-method")]
    public async Task Quantization_preserves_trained_state_for_tree_models()
    {
        // #53: gradient-boosted trees keep their trained state (_trees / _binThresholds) OUTSIDE the parameter
        // vector, so quantizing via WithParameters used to drop it and Predict threw "Model must be trained".
        var rng = new Random(11);
        var rows = 80;
        var cols = 3;
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (var i = 0; i < rows; i++)
        {
            double a = rng.NextDouble(), b = rng.NextDouble(), c = rng.NextDouble();
            x[i, 0] = a; x[i, 1] = b; x[i, 2] = c;
            y[i] = 2.0 * a - 1.0 * b + 0.5 * c;   // a learnable linear-ish target
        }

        var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y);
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new HistGradientBoostingRegression<double>())
            .ConfigureDataLoader(loader)
            .ConfigureQuantization(new QuantizationConfig { Mode = QuantizationMode.Float16 })
            .BuildAsync();

        var predictions = result.Predict(x);   // must NOT throw "Model must be trained before making predictions"
        Assert.Equal(rows, predictions.Length);
        Assert.All(Enumerable.Range(0, rows), i => Assert.True(IsFiniteValue(predictions[i])));
        Assert.True(predictions.Distinct().Count() > 1, "quantized tree model collapsed to a constant");
    }

    [Fact(Timeout = 120_000)]
    [Trait("category", "integration-configure-method")]
    public async Task Facade_trains_a_rank3_tensor_forecasting_model()
    {
        // #54: a sequence forecaster consumes rank-3 [batch, seq, features]. The optimizer's non-flat guard used
        // to misclassify Dense-embedding forecasters (PatchTST/iTransformer) as columnar and crash in
        // GetColumnVectors. It must now train + predict through the facade.
        const int n = 48;
        const int lookback = 16;
        var rng = new Random(5);
        var xin = new double[n * lookback];
        var yin = new double[n];
        for (var s = 0; s < n; s++)
        {
            double level = 0.0;
            for (var t = 0; t < lookback; t++)
            {
                level = 0.6 * level + (rng.NextDouble() - 0.5) * 0.1;
                xin[s * lookback + t] = level;
            }

            yin[s] = 0.6 * level;   // next-step continuation
        }

        var xTrain = new Tensor<double>([n, lookback, 1], new Vector<double>(xin));
        var yTrain = new Tensor<double>([n, 1, 1], new Vector<double>(yin));

        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: lookback, outputSize: 1);
        var model = new global::AiDotNet.Finance.Forecasting.Transformers.PatchTST<double>(
            arch, sequenceLength: lookback, predictionHorizon: 1, numFeatures: 1,
            patchSize: 8, stride: 4, numLayers: 1, numHeads: 2, modelDimension: 16, feedForwardDimension: 16);

        var loader = new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(xTrain, yTrain);
        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .BuildAsync();   // previously threw ArgumentException "Number of indices must match the tensor's rank."

        var probeFlat = new double[lookback];
        Array.Copy(xin, probeFlat, lookback);
        var prediction = result.Predict(new Tensor<double>([1, lookback, 1], new Vector<double>(probeFlat)));
        var flat = prediction.ToVector();
        Assert.True(flat.Length >= 1);
        Assert.True(IsFiniteValue(flat[0]), $"rank-3 forecast produced non-finite {flat[0]}");
    }
}
