using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;
using System.Threading.Tasks;
using AiDotNetVector = AiDotNet.Tensors.LinearAlgebra.Vector<float>;

namespace AiDotNetTests.UnitTests.MixedPrecision;

/// <summary>
/// Mixed precision trains on the fused compiled path, as PyTorch's autocast + GradScaler do: the same compiled plan as an
/// FP32 run (no eager fallback, so no extra per-step launches or host round trips), FP32 master weights, the loss scaled
/// into the backward and unscaled inside the plan. The loss trajectory then matches FP32 training within a stated
/// tolerance, and an overflowing step is skipped with the scale backed off.
/// </summary>
public class MixedPrecisionFusedPathTests
{
    private const int InputSize = 8;
    private const int Hidden = 16;
    private const int OutputSize = 4;
    private const int Rows = 16;
    private const int Steps = 30;

    private static NeuralNetwork<float> BuildNetwork()
    {
        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(InputSize),
            new DenseLayer<float>(Hidden, activationFunction: new TanhActivation<float>()),
            new DenseLayer<float>(OutputSize, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: InputSize,
            outputSize: OutputSize,
            layers: layers);
        var model = new NeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());
        var p = model.GetParameters();
        var det = new float[p.Length];
        for (int i = 0; i < det.Length; i++) det[i] = (((i * 37) % 23) - 11) * 0.02f;
        model.UpdateParameters(new AiDotNetVector(det));
        return model;
    }

    private static (Tensor<float> X, Tensor<float> Y) MakeBatch(int step)
    {
        var x = new float[Rows * InputSize];
        var y = new float[Rows * OutputSize];
        for (int i = 0; i < x.Length; i++) x[i] = (float)System.Math.Sin(0.37 * i + 0.11 * step);
        for (int r = 0; r < Rows; r++)
            for (int o = 0; o < OutputSize; o++)
                y[r * OutputSize + o] = (float)System.Math.Cos(0.5 * x[r * InputSize + o] + 0.2 * o);
        return (new Tensor<float>(new[] { Rows, InputSize }, new AiDotNetVector(x)),
                new Tensor<float>(new[] { Rows, OutputSize }, new AiDotNetVector(y)));
    }

    private static (double[] Losses, long FusedSteps) Train(MixedPrecisionConfig? config)
    {
        var model = BuildNetwork();
        if (config is not null) model.EnableMixedPrecision(config);
        var losses = new double[Steps];
        for (int s = 0; s < Steps; s++)
        {
            var (x, y) = MakeBatch(s);
            model.Train(x, y);
            losses[s] = System.Convert.ToDouble(model.GetLastLoss());
        }
        // The calling thread's counter for the state this model's last step ran on (a fresh model: all its steps).
        return (losses, CompiledTapeTrainingStep<float>.GetFusedStepCount());
    }

    [Theory(Timeout = 120000)]
    [InlineData(MixedPrecisionType.FP16)]
    [InlineData(MixedPrecisionType.BF16)]
    public async Task MixedPrecision_TakesTheSameFusedPlan_AndMatchesFp32Loss(MixedPrecisionType precision)
    {
        await Task.Yield();
        var (fp32, fp32Fused) = Train(null);
        var config = precision == MixedPrecisionType.BF16
            ? MixedPrecisionConfig.ForBF16()
            : new MixedPrecisionConfig { PrecisionType = MixedPrecisionType.FP16 };   // the library default: scale 65536, dynamic
        var (mp, mpFused) = Train(config);

        // Same route: every step fused, exactly as many as FP32 (launch count per step equal by construction).
        Assert.True(fp32Fused > 0, "the FP32 reference must take the fused compiled path");
        Assert.Equal(fp32Fused, mpFused);

        // Same trajectory. A power-of-two loss scale is exact in FP32 and matmuls on the CPU stay FP32, so the only
        // difference is rounding at overflow-free steps; on a GPU the FP16/BF16 matmul rounding adds ~1e-3 relative.
        const double relativeTolerance = 2e-2;
        for (int s = 0; s < Steps; s++)
        {
            Assert.True(double.IsFinite(mp[s]), $"step {s}: mixed-precision loss {mp[s]} is not finite");
            double rel = System.Math.Abs(mp[s] - fp32[s]) / System.Math.Max(1e-6, System.Math.Abs(fp32[s]));
            Assert.True(rel <= relativeTolerance, $"step {s}: FP32 loss {fp32[s]} vs mixed {mp[s]} (relative {rel:E2})");
        }
        Assert.True(mp[Steps - 1] < mp[0], "mixed-precision training must reduce the loss");
    }

    [Fact(Timeout = 120000)]
    public async Task MixedPrecision_Fp16_ScalerStateReadsBackFromThePlan()
    {
        await Task.Yield();
        var model = BuildNetwork();
        model.EnableMixedPrecision(new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = 1024.0,
            ScaleGrowthInterval = 4,
        });
        for (int s = 0; s < 10; s++)
        {
            var (x, y) = MakeBatch(s);
            model.Train(x, y);
        }
        var scaler = model.GetMixedPrecisionContext()!.LossScaler;
        // Ten finite steps with growth every four: 1024 -> 2048 -> 4096, and the counts follow the plan's scaler.
        Assert.Equal(4096.0, scaler.Scale);
        Assert.Equal(10, scaler.TotalUpdates);
        Assert.Equal(0, scaler.SkippedUpdates);
    }
}
