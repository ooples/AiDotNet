using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Regression.MultiOutput;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// End-to-end facade tests for multi-output (n×H) regression: the model is trained through the full
/// <see cref="AiModelBuilder{T, TInput, TOutput}"/> pipeline (optimizer-driven selection + StandardScaler feature
/// preprocessing), NOT by calling Train/Predict directly. This is the regression test for the optimizer bug where
/// an n×H target scored as garbage (ConvertToVector could not flatten a Matrix), causing the trained model to lose
/// selection to the default untrained model and predict ≈0. A correct pipeline recovers each horizon.
/// </summary>
[Trait("Category", "Integration")]
public class MultiOutputRegressorFacadeTests
{
    private static (Matrix<double> x, Matrix<double> y) CreateMultiHorizonData(int n, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(n, 2);
        var y = new Matrix<double>(n, 3);
        for (var i = 0; i < n; i++)
        {
            var a = random.NextDouble() * 10.0;
            var b = random.NextDouble() * 10.0;
            x[i, 0] = a;
            x[i, 1] = b;
            y[i, 0] = 2.0 * a + 1.0 * b + 3.0;
            y[i, 1] = -1.0 * a + 4.0 * b - 2.0;
            y[i, 2] = 0.5 * a - 0.5 * b + 1.0;
        }

        return (x, y);
    }

    [Fact]
    public async Task Facade_TrainsAndRecoversEachHorizon()
    {
        var (x, y) = CreateMultiHorizonData(200, seed: 42);
        var model = new MultiOutputRegressor<double>(() => new MultipleRegression<double>());

        var result = await new AiModelBuilder<double, Matrix<double>, Matrix<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Matrix<double>>(x, y))
            .ConfigurePreprocessing(new StandardScaler<double>())
            .BuildAsync();

        var pred = result.Predict(x);

        Assert.Equal(x.Rows, pred.Rows);
        Assert.Equal(3, pred.Columns);

        // A correctly trained pipeline recovers each (clean linear) horizon. Loose tolerance because the facade
        // standardizes features; the point is the model TRAINED and was SELECTED (not the ≈0 default).
        for (var j = 0; j < 3; j++)
        {
            var maxError = 0.0;
            for (var i = 0; i < x.Rows; i++)
            {
                maxError = Math.Max(maxError, Math.Abs(pred[i, j] - y[i, j]));
            }

            Assert.True(maxError < 0.5, $"horizon {j} max abs error was {maxError} (model was not trained/selected)");
        }

        // Horizons must be genuinely distinct — not one scalar broadcast across columns.
        var distinct = false;
        for (var i = 0; i < x.Rows; i++)
        {
            if (Math.Abs(pred[i, 0] - pred[i, 1]) > 1.0)
            {
                distinct = true;
                break;
            }
        }

        Assert.True(distinct, "horizons collapsed to the same value — multi-output selection is broken");
    }

    [Fact]
    public async Task Facade_WithPerColumnTargetScaling_RecoversEachHorizon()
    {
        // Exercises the full default path: per-column target standardization (ConfigureTargetScaling) + inverse-
        // transform on the way out. Horizons here are on deliberately DIFFERENT scales so a working per-column
        // target scaler matters. Recovery in native units proves the inverse-transform is wired for a Matrix target.
        var random = new Random(7);
        const int n = 200;
        var x = new Matrix<double>(n, 2);
        var y = new Matrix<double>(n, 3);
        for (var i = 0; i < n; i++)
        {
            var a = random.NextDouble() * 10.0;
            var b = random.NextDouble() * 10.0;
            x[i, 0] = a;
            x[i, 1] = b;
            y[i, 0] = 2.0 * a + b;             // ~O(10)
            y[i, 1] = 100.0 * a - 50.0 * b;    // ~O(1000)
            y[i, 2] = 0.01 * a + 0.02 * b;     // ~O(0.1)
        }

        var model = new MultiOutputRegressor<double>(() => new MultipleRegression<double>());
        var result = await new AiModelBuilder<double, Matrix<double>, Matrix<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Matrix<double>>(x, y))
            .ConfigurePreprocessing(new StandardScaler<double>())
            .ConfigureTargetScaling()
            .BuildAsync();

        var pred = result.Predict(x);

        Assert.Equal(3, pred.Columns);
        for (var j = 0; j < 3; j++)
        {
            // Relative error per horizon (scales differ by 4 orders of magnitude), so a per-column scaler is essential.
            var maxRel = 0.0;
            for (var i = 0; i < n; i++)
            {
                var denom = Math.Max(1e-9, Math.Abs(y[i, j]));
                maxRel = Math.Max(maxRel, Math.Abs(pred[i, j] - y[i, j]) / denom);
            }

            Assert.True(maxRel < 0.05, $"horizon {j} max relative error {maxRel} — target inverse-transform likely broken");
        }
    }
}
