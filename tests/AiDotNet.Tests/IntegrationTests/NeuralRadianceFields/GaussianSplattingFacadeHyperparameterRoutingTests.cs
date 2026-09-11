using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralRadianceFields;

/// <summary>
/// End-to-end regression test for #1833.
///
/// The existing unit tests (<c>GaussianSplattingHyperparameterAwareTests</c>) call
/// <c>ApplyOptimizerHyperparameters</c> directly, which pins the derivation maths but NOT
/// the thing #1833 actually reported: that going through the <c>AiModelBuilder</c> facade
/// silently dropped the caller's <c>ConfigureOptimizer</c> settings. Those tests would keep
/// passing even if the builder stopped invoking the hook entirely.
///
/// This test closes that gap by driving the real facade path — <c>ConfigureDataLoader</c> →
/// <c>ConfigureModel</c> → <c>ConfigureOptimizer</c> → <c>BuildAsync</c> — and asserting the
/// caller's learning rate reached the model's per-attribute schedule.
/// </summary>
public class GaussianSplattingFacadeHyperparameterRoutingTests
{
    // Deliberately distinct from every default in play, so a pass cannot be a coincidence:
    //   - AdamOptimizerOptions default InitialLearningRate = 1e-3
    //   - OptimizationAlgorithmOptions base default        = 0.01
    //   - the 3DGS paper's canonical position anchor       = 1.6e-4
    private const double CallerLearningRate = 8.0e-4;

    // Kerbl et al. 2023 ratio: color = position * (2.5e-3 / 1.6e-4) = position * 15.625.
    private const double ExpectedColorLearningRate = CallerLearningRate * 15.625;   // 0.0125

    // Seeded into the model's options so they differ from BOTH the caller-derived values
    // above AND the library defaults. If the facade never invokes the hook, these survive
    // untouched and the assertions below fail — which is what makes this a real control.
    private const double SeededPositionLearningRate = 0.05;
    private const double SeededColorLearningRate = 0.2;

    private static GaussianSplatting<double> BuildSeededModel()
    {
        var options = new GaussianSplattingOptions
        {
            EnableDensification = false,
            EnableSpatialIndex = false,
            UseSphericalHarmonics = false,
            MaxGaussians = 8,
            ShDegree = 0,
            PositionLearningRate = SeededPositionLearningRate,
            ColorLearningRate = SeededColorLearningRate,
        };

        var points = new Matrix<double>(4, 3);
        var colors = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                points[i, j] = 0.1 * (i + j);
                colors[i, j] = 0.5;
            }
        }

        return new GaussianSplatting<double>(options, points, colors);
    }

    [Fact(Timeout = 120000)]
    public async Task BuildAsync_RoutesConfiguredLearningRate_IntoPerAttributeSchedule()
    {
        var model = BuildSeededModel();

        // Control arm: before the build, the model still holds the seeded values. If this
        // were already the derived value the test could not discriminate.
        Assert.Equal(SeededPositionLearningRate, model.PositionLearningRate, precision: 12);
        Assert.Equal(SeededColorLearningRate, model.ColorLearningRate, precision: 12);

        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = CallerLearningRate,
                MaxIterations = 1,
            });

        // GaussianSplatting's ray contract is [N, 6] in (3D position + 3D view direction)
        // and [N, 4] out (RGBA). 30 rows so the loader's train/validation/test split leaves
        // no empty shard — the input contract rejects a zero-length axis.
        const int Rows = 30;
        var features = new Tensor<double>([Rows, 6]);
        var targets = new Tensor<double>([Rows, 6]);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                features[i, j] = 0.01 * (i + j);
            }

            for (int j = 0; j < 6; j++)
            {
                targets[i, j] = 0.02 * (i + j);
            }
        }

        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(features, targets))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .BuildAsync();

        // The caller's base LR became the position LR verbatim, and the remaining attributes
        // were re-derived from it using the paper's ratios.
        Assert.Equal(CallerLearningRate, model.PositionLearningRate, precision: 12);
        Assert.Equal(ExpectedColorLearningRate, model.ColorLearningRate, precision: 10);

        // Explicitly assert the seeded values did NOT survive — the failure mode #1833
        // reported was the configured value being silently ignored.
        Assert.NotEqual(SeededPositionLearningRate, model.PositionLearningRate);
        Assert.NotEqual(SeededColorLearningRate, model.ColorLearningRate);
    }
}
