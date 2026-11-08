using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// The "ddpm" row: one DDPM training step of the shipped <see cref="DDPMModel{T}"/> (Ho et al. 2020's CIFAR-10 U-Net,
/// 128 base channels, multipliers [1, 2, 2, 2], two residual blocks per level, attention at 16x16) on [3, 32, 32]
/// images. A step draws a timestep and Gaussian noise per image, noises the batch on the linear schedule
/// (1e-4 to 0.02 over 1000 steps) and regresses the noise with mean squared error. Inference is one denoiser
/// evaluation of the whole batch at timestep 500. benchmark.py's <c>DDPMUNet</c> is a layer-for-layer twin of the
/// U-Net as AiDotNet builds it, so the two sides time the same function.
/// </summary>
internal sealed class AiDotNetDdpmModel : IBenchmarkModel
{
    internal const int ImageSize = 32;
    internal const int Channels = 3;
    internal const int InferenceTimestep = 500;

    private readonly DDPMModel<float> _model;
    private readonly Random _random;
    private Tensor<float> _input = Tensor<float>.Empty();
    private bool _hasTrainingLoss;

    public AiDotNetDdpmModel(int seed)
    {
        _random = new Random(seed);
        _model = new DDPMModel<float>(
            options: new DiffusionModelOptions<float>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear,
                // The optimizer every row uses (torch.optim.AdamW defaults, lr 1e-3), with the same global-norm clip
                // at 1.0 the PyTorch loop applies.
                OptimizerFactory = () => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 1e-3, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
                        EnableGradientClipping = true, MaxGradientNorm = 1.0,
                    }),
            },
            channels: Channels,
            imageSize: ImageSize,
            seed: seed);

        // The U-Net's layers are lazy: one forward allocates them, so the count is the real size.
        LoadSyntheticBatch(1);
        Forward();
        ParameterCount = _model.ParameterCount;
    }

    public long ParameterCount { get; }

    /// <summary>The batch is the clean images; DDPM draws its own noise and timesteps inside each step.</summary>
    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _input);
        set => _input = value.Input;
    }

    /// <summary>Clean images uniform in [-1, 1], the range DDPM trains on (benchmark.py's ddpm_batch).</summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        var pixels = new float[batchSize * Channels * ImageSize * ImageSize];
        for (var i = 0; i < pixels.Length; i++) pixels[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        _input = new Tensor<float>(pixels, [batchSize, Channels, ImageSize, ImageSize]);
    }

    public void Forward() => _ = _model.PredictNoise(_input, InferenceTimestep);

    public void Backward()
    {
        // DiffusionModelBase.Train is the whole step: noise, forward, MSE, backward and the optimizer update.
        _model.Train(_input, _input);
        _hasTrainingLoss = true;
    }

    public void Step()
    {
        // The optimizer step ran inside Backward's Train call.
    }

    public double? LastLoss => _hasTrainingLoss ? _model.GetLastLoss() : null;
}
