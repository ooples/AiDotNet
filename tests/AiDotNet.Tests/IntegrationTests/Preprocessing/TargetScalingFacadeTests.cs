using AiDotNet.Data.Loaders;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// ConfigureTargetScaling end-to-end: the facade fits the target scaler on the TRAINING split, trains in
/// scaled space, and Predict returns values in the ORIGINAL target units (via the previously-orphaned
/// PreprocessingInfo.TargetPipeline / InverseTransformPredictions plumbing — now populated and exercised).
/// </summary>
public class TargetScalingFacadeTests
{
    [Fact]
    public async Task Predictions_come_back_in_original_target_units()
    {
        // y = 1000·x1 + 5000 — a target whose scale (thousands) is far from the unit-scaled features.
        const int n = 120;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double xi = i / (double)n;
            x[i, 0] = xi;
            y[i] = (1000.0 * xi) + 5000.0;
        }

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .ConfigurePreprocessing(new StandardScaler<double>())
            .ConfigureTargetScaling()
            .BuildAsync();

        Assert.True(result.PreprocessingInfo is not null, "PreprocessingInfo missing on the result");
        Assert.True(result.PreprocessingInfo!.IsTargetFitted,
            $"Target pipeline not fitted/carried: TargetPipeline={(result.PreprocessingInfo.TargetPipeline is null ? "null" : "set-unfitted")}");

        var probe = new Matrix<double>(2, 1);
        probe[0, 0] = 0.25; // expect ≈ 5250
        probe[1, 0] = 0.75; // expect ≈ 5750
        var pred = result.Predict(probe);

        // If the inverse transform were missing, predictions would sit near 0 (z-score space, |v| < ~3).
        Assert.True(Math.Abs(pred[0] - 5250.0) < 150.0, $"pred[0]={pred[0]} — expected ≈5250 in ORIGINAL units");
        Assert.True(Math.Abs(pred[1] - 5750.0) < 150.0, $"pred[1]={pred[1]} — expected ≈5750 in ORIGINAL units");
    }

    [Fact]
    public void TargetStandardScaler_round_trips_vectors_and_tensors()
    {
        var v = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });
        var sv = new TargetStandardScaler<double, Vector<double>>();
        var scaled = sv.FitTransform(v);
        Assert.True(Math.Abs(scaled.Average()) < 1e-9, "scaled mean should be ~0");
        var back = sv.InverseTransform(scaled);
        for (int i = 0; i < v.Length; i++)
        {
            Assert.Equal(v[i], back[i], 6);
        }

        var t = new Tensor<double>(new[] { 4, 1 }, new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 }));
        var st = new TargetStandardScaler<double, Tensor<double>>();
        var backT = st.InverseTransform(st.FitTransform(t));
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(t.Data.Span[i], backT.Data.Span[i], 6);
        }
    }
}
