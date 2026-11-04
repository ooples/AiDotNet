using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// AdamW with compressed moments (<see cref="AdamWOptimizerOptions{T, TInput, TOutput}.MomentStorage"/>): bf16 and
/// 8-bit block-wise. The reference is Tensors' host twin of the device kernels with decoupled decay, so the expected
/// numbers are bitsandbytes' AdamW8bit / bf16-state AdamW, and the same state must give the same bits on the device,
/// on the host, across a device-to-host move and across a checkpoint.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class AdamWCompressedMomentsTests
{
    private const int Length = 8192;                 // >= the 8-bit minimum (4096), so the moments really quantize
    private const float Lr = 1e-2f, Beta1 = 0.9f, Beta2 = 0.99f, Eps = 1e-8f, WeightDecay = 0.05f;
    private const int BlockSize = 256;

    [Fact]
    public void CompressedMomentLayout_DoesNotPreventLoadingOptimizerTypes()
    {
        var optimizerType = typeof(AdamWOptimizer<,,>);
        Assert.Contains(optimizerType, optimizerType.Assembly.GetTypes());
    }


    [Fact]
    public void OptionsCopy_PreservesEveryMomentSetting()
    {
        var original = new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>
        {
            MomentStorage = OptimizerMomentStorage.Int8Blockwise,
            Int8MomentBlockSize = 64,
            Int8MinQuantizedLength = 128,
        };
        var copy = new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>(original);
        Assert.Equal(original.MomentStorage, copy.MomentStorage);
        Assert.Equal(original.Int8MomentBlockSize, copy.Int8MomentBlockSize);
        Assert.Equal(original.Int8MinQuantizedLength, copy.Int8MinQuantizedLength);
    }

    [Theory]
    [InlineData("block-size")]
    [InlineData("minimum-length")]
    [InlineData("bf16")]
    [InlineData("fp32")]
    public void Checkpoint_RejectsIncompatibleMomentFormatsWithoutChangingTarget(string mismatch)
    {
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var source = Optimizer(OptimizerMomentStorage.Int8Blockwise);
            var param = new Tensor<float>(new[] { Length }, new Vector<float>(Rand(Length, 1)));
            Step(source, param, Grad(1));
            byte[] checkpoint = source.Serialize();
            var options = new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>
            {
                MomentStorage = mismatch == "bf16" ? OptimizerMomentStorage.BFloat16
                    : mismatch == "fp32" ? OptimizerMomentStorage.Float32 : OptimizerMomentStorage.Int8Blockwise,
                Int8MomentBlockSize = mismatch == "block-size" ? 64 : BlockSize,
                Int8MinQuantizedLength = mismatch == "minimum-length" ? 128 : 4096,
                EnableGradientClipping = false,
            };
            var target = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null, options);
            byte[] before = target.Serialize();
            Assert.Throws<InvalidOperationException>(() => target.Deserialize(checkpoint));
            Assert.Equal(before, target.Serialize());
            // Rejected restore must leave the source usable and lossless on a compatible target.
            var compatible = Optimizer(OptimizerMomentStorage.Int8Blockwise);
            compatible.Deserialize(checkpoint);
            var resumed = new Tensor<float>(new[] { Length }, new Vector<float>(param.ToArray()));
            Step(compatible, resumed, Grad(2));
            Step(source, param, Grad(2));
            Assert.Equal(param.ToArray(), resumed.ToArray());
        }
        finally { AiDotNetEngine.Current = saved; }
    }

    [Fact]
    public void CompressedStep_RejectsMismatchedGradientLength()
    {
        var optimizer = Optimizer(OptimizerMomentStorage.BFloat16);
        Assert.Throws<ArgumentException>(() => Step(optimizer,
            new Tensor<float>(new[] { 4 }), new Tensor<float>(new[] { 3 })));
    }

    [Fact]
    public void FusedConfig_DeclinesUnsupportedCompressedRequests()
    {
        var nonFloat = new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null,
            new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
            { MomentStorage = OptimizerMomentStorage.BFloat16, UseAdaptiveLearningRate = false, EnableGradientClipping = false });
        Assert.False(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)nonFloat).TryGetFusedOptimizerConfig(out _));
        var float32Control = new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null,
            new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
            { MomentStorage = OptimizerMomentStorage.Float32, UseAdaptiveLearningRate = false, EnableGradientClipping = false });
        Assert.True(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)float32Control).TryGetFusedOptimizerConfig(out _));
        var zeroBlock = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null,
            new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>
            { MomentStorage = OptimizerMomentStorage.Int8Blockwise, Int8MomentBlockSize = 0,
                UseAdaptiveLearningRate = false, EnableGradientClipping = false });
        Assert.False(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)zeroBlock).TryGetFusedOptimizerConfig(out _));
        var positiveBlockControl = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null,
            new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>
            { MomentStorage = OptimizerMomentStorage.Int8Blockwise, Int8MomentBlockSize = 64,
                UseAdaptiveLearningRate = false, EnableGradientClipping = false });
        Assert.True(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)positiveBlockControl).TryGetFusedOptimizerConfig(out _));
    }

    private static float[] Rand(int n, int seed, double scale = 1.0)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    private static AdamWOptimizer<float, Matrix<float>, Vector<float>> Optimizer(OptimizerMomentStorage storage)
        => new(null, new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>
        {
            InitialLearningRate = Lr,
            Beta1 = Beta1,
            Beta2 = Beta2,
            Epsilon = Eps,
            WeightDecay = WeightDecay,
            MomentStorage = storage,
            Int8MomentBlockSize = BlockSize,
            // The reference takes the raw gradients; default global-norm clipping would rescale them first.
            EnableGradientClipping = false,
        });

    private static void Step(AdamWOptimizer<float, Matrix<float>, Vector<float>> optimizer, Tensor<float> param, Tensor<float> grad)
        => optimizer.Step(new TapeStepContext<float>(
            parameters: new[] { param },
            gradients: new Dictionary<Tensor<float>, Tensor<float>> { [param] = grad },
            loss: 0f));

    /// <summary>The host twin's trajectory: <paramref name="steps"/> AdamW steps from <c>Rand(Length, 1)</c>.</summary>
    private static float[] Reference(OptimizerMomentStorage storage, int steps)
    {
        var p = Rand(Length, 1);
        int blocks = HostCompressedMomentAdam.BlockCount(Length, BlockSize);
        var m = new byte[storage == OptimizerMomentStorage.Int8Blockwise ? Length : 2 * Length];
        var v = new byte[m.Length];
        var ms = new float[blocks];
        var vs = new float[blocks];
        for (int step = 1; step <= steps; step++)
        {
            var g = Rand(Length, 100 + step, 0.5);
            if (storage == OptimizerMomentStorage.Int8Blockwise)
                HostCompressedMomentAdam.Int8Step(p, g, m, v, ms, vs, Length, BlockSize, Lr, Beta1, Beta2, Eps, WeightDecay, step, true);
            else
                HostCompressedMomentAdam.Bf16Step(p, g, m, v, Length, Lr, Beta1, Beta2, Eps, WeightDecay, step, true);
        }
        return p;
    }

    private static Tensor<float> Grad(int step) => new(new[] { Length }, new Vector<float>(Rand(Length, 100 + step, 0.5)));

    private static void AssertSame(float[] want, float[] got, string label)
    {
        for (int i = 0; i < want.Length; i++)
            Assert.True(want[i] == got[i], $"{label}: element {i} is {got[i]}, expected {want[i]}");
    }

    /// <summary>
    /// Device kernels round the update (fused multiply-add, sqrt) within an ulp of the host; Tensors pins the moment
    /// bytes exactly and the weights to this tolerance, so the same bound holds here.
    /// </summary>
    private static void AssertClose(float[] want, float[] got, string label, float tolerance = 1e-5f)
    {
        for (int i = 0; i < want.Length; i++)
            Assert.True(Math.Abs(want[i] - got[i]) < tolerance, $"{label}: element {i} is {got[i]}, expected {want[i]}");
    }

    /// <summary>How many of the optimizer's compressed slots hold their state on the device.</summary>
    private static int DeviceSlotCount(AdamWOptimizer<float, Matrix<float>, Vector<float>> optimizer)
    {
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public;
        var slots = (System.Collections.IDictionary)typeof(AdamWOptimizer<float, Matrix<float>, Vector<float>>)
            .GetField("_compressedSlots", flags)!.GetValue(optimizer)!;
        int onDevice = 0;
        foreach (var slot in slots.Values)
            if ((bool)slot!.GetType().GetProperty("OnDevice", flags)!.GetValue(slot)!) onDevice++;
        return onDevice;
    }

    private static int Fp32MomentCount(AdamWOptimizer<float, Matrix<float>, Vector<float>> optimizer)
    {
        var field = typeof(AdamWOptimizer<float, Matrix<float>, Vector<float>>).GetField("_tapeM",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        return ((System.Collections.ICollection)field.GetValue(optimizer)!).Count;
    }

    [Theory]
    [InlineData(OptimizerMomentStorage.Int8Blockwise)]
    [InlineData(OptimizerMomentStorage.BFloat16)]
    public void HostSteps_FollowTheReference_WithoutFp32Moments(OptimizerMomentStorage storage)
    {
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var optimizer = Optimizer(storage);
            var param = new Tensor<float>(new[] { Length }, new Vector<float>(Rand(Length, 1)));
            for (int step = 1; step <= 4; step++) Step(optimizer, param, Grad(step));
            AssertSame(Reference(storage, 4), param.ToArray(), $"{storage} host");
            Assert.Equal(0, Fp32MomentCount(optimizer));

            // The 32-bit control lands elsewhere: the storage really changed the moments.
            var fp32 = Optimizer(OptimizerMomentStorage.Float32);
            var control = new Tensor<float>(new[] { Length }, new Vector<float>(Rand(Length, 1)));
            for (int step = 1; step <= 4; step++) Step(fp32, control, Grad(step));
            Assert.Equal(1, Fp32MomentCount(fp32));
            float drift = param.ToArray().Zip(control.ToArray(), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(drift < 1e-2f, $"{storage} drifted {drift} from fp32 AdamW in 4 steps");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Theory]
    [InlineData(OptimizerMomentStorage.Int8Blockwise)]
    [InlineData(OptimizerMomentStorage.BFloat16)]
    public void Checkpoint_RestoresTheCompressedState_Exactly(OptimizerMomentStorage storage)
    {
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var first = Optimizer(storage);
            var param = new Tensor<float>(new[] { Length }, new Vector<float>(Rand(Length, 1)));
            for (int step = 1; step <= 2; step++) Step(first, param, Grad(step));
            byte[] state = first.Serialize();

            var resumed = Optimizer(storage);
            resumed.Deserialize(state);
            var resumedParam = new Tensor<float>(new[] { Length }, new Vector<float>(param.ToArray()));
            for (int step = 3; step <= 4; step++) Step(resumed, resumedParam, Grad(step));
            AssertSame(Reference(storage, 4), resumedParam.ToArray(), $"{storage} resumed");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    [Fact]
    public void ShortTensors_KeepFp32Moments_UnderInt8()
    {
        var optimizer = Optimizer(OptimizerMomentStorage.Int8Blockwise);
        var bias = new Tensor<float>(new[] { 64 }, new Vector<float>(Rand(64, 3)));
        var control = new Tensor<float>(new[] { 64 }, new Vector<float>(Rand(64, 3)));
        var fp32 = Optimizer(OptimizerMomentStorage.Float32);
        var g = new Tensor<float>(new[] { 64 }, new Vector<float>(Rand(64, 4)));
        Step(optimizer, bias, g);
        Step(fp32, control, g);
        AssertSame(control.ToArray(), bias.ToArray(), "a 64-element tensor under Int8Blockwise");
    }

    [Fact]
    public void AmsGrad_WithCompressedMoments_IsRefused()
    {
        var optimizer = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null,
            new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>>
            {
                UseAMSGrad = true,
                MomentStorage = OptimizerMomentStorage.BFloat16,
                EnableGradientClipping = false,
            });
        var param = new Tensor<float>(new[] { Length }, new Vector<float>(Rand(Length, 1)));
        Assert.Throws<NotSupportedException>(() => Step(optimizer, param, Grad(1)));
    }

    /// <summary>
    /// On a GPU engine the step runs on the device, with the same bits as the host twin. Moving to the CPU halfway
    /// downloads the state and continues on the host, still on the reference trajectory.
    /// </summary>
    [SkippableTheory]
    [InlineData(OptimizerMomentStorage.Int8Blockwise)]
    [InlineData(OptimizerMomentStorage.BFloat16)]
    public void DeviceSteps_MatchTheHost_AndTheStateMovesToTheHost(OptimizerMomentStorage storage)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or InvalidOperationException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu!;
            var optimizer = Optimizer(storage);
            // Device-resident parameter and gradients: results of device ops.
            var param = gpu!.TensorMultiplyScalar(new Tensor<float>(new[] { Length }, new Vector<float>(Rand(Length, 1))), 1f);
            for (int step = 1; step <= 2; step++)
                Step(optimizer, param, gpu.TensorMultiplyScalar(Grad(step), 1f));
            AssertClose(Reference(storage, 2), param.ToArray(), $"{storage} device");
            Assert.True(DeviceSlotCount(optimizer) == 1, $"{storage}: the steps ran on the host, not the device");
            Assert.Equal(0, Fp32MomentCount(optimizer));

            AiDotNetEngine.Current = new CpuEngine();
            var hostParam = new Tensor<float>(new[] { Length }, new Vector<float>(param.ToArray()));
            // The same optimizer continues on the host: rebind its state to the host copy of the parameter.
            var resumed = Optimizer(storage);
            AiDotNetEngine.Current = gpu;
            byte[] state = optimizer.Serialize();       // downloads the device state
            AiDotNetEngine.Current = new CpuEngine();
            resumed.Deserialize(state);
            for (int step = 3; step <= 4; step++) Step(resumed, hostParam, Grad(step));
            AssertClose(Reference(storage, 4), hostParam.ToArray(), $"{storage} device then host");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu?.Dispose();
        }
    }

    /// <summary>The compiled (fused) plan gets the same storage request, and AMSGrad declines as the eager step refuses.</summary>
    [Theory]
    [InlineData(OptimizerMomentStorage.Int8Blockwise)]
    [InlineData(OptimizerMomentStorage.BFloat16)]
    public void FusedConfig_CarriesTheMomentStorage(OptimizerMomentStorage storage)
    {
        var optimizer = Optimizer(storage);
        Assert.True(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out var config));
        Assert.Equal(storage, config.MomentStorage);
        Assert.Equal(BlockSize, config.Int8MomentBlockSize);
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdamW, config.Type);

        var amsGrad = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null,
            new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>> { UseAMSGrad = true, MomentStorage = storage });
        Assert.False(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)amsGrad).TryGetFusedOptimizerConfig(out _));
    }
}
