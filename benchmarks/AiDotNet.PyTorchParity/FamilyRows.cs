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
                // Ho et al. 2020's optimizer: Adam at 2e-4 with the global gradient norm clipped to 1, no weight decay.
                // The other rows' AdamW at 1e-3 is too large for this U-Net without warmup: a step that size took the
                // loss from 2 to 50 on both sides at the paper's initial weight scale. The twin uses the same settings.
                OptimizerFactory = () => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        InitialLearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
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

/// <summary>
/// The "whisper-tiny" row: the shipped <see cref="AiDotNet.Audio.Whisper.WhisperModel{T}"/> at the tiny size (width
/// 384, 4 encoder and 4 decoder layers, 6 heads) on 30-second [3000, 80] log-mel clips. A training step is
/// teacher-forced cross-entropy over a 16-token transcript; inference is the encoder plus one decoder step from
/// <c>&lt;|startoftranscript|&gt;</c>. benchmark.py's <c>WhisperTiny</c> is transformers'
/// WhisperForConditionalGeneration at the same configuration; it has 9,216 more parameters, the q and v attention
/// biases AiDotNet's attention layers do not have.
/// </summary>
internal sealed class AiDotNetWhisperTinyModel : IBenchmarkModel
{
    internal const int Mels = 80;
    internal const int Frames = 3000;
    internal const int Tokens = 16;
    internal const int StartOfTranscript = 50258;

    private readonly AiDotNet.Audio.Whisper.WhisperModel<float> _model;
    private readonly Random _random;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _label = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetWhisperTinyModel(int seed)
    {
        _random = new Random(seed);
        _model = new AiDotNet.Audio.Whisper.WhisperModel<float>(
            new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<float>(
                inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
                inputHeight: Frames, inputWidth: Mels, inputDepth: 1, outputSize: 51865),
            new AiDotNet.Audio.Whisper.WhisperOptions { ModelSize = AiDotNet.Audio.Whisper.WhisperModelSize.Tiny },
            // The optimizer every row uses (torch.optim.AdamW defaults, lr 1e-3) with the same global-norm clip.
            new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 1e-3, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
                    EnableGradientClipping = true, MaxGradientNorm = 1.0,
                }));

        LoadSyntheticBatch(1);
        Forward();
        ParameterCount = _model.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _label);
        set => (_input, _label) = value;
    }

    /// <summary>Log-mel features in [-1, 1] and random transcript tokens below the special-token range.</summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        var features = new float[batchSize * Frames * Mels];
        for (var i = 0; i < features.Length; i++) features[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        _input = new Tensor<float>(features, [batchSize, Frames, Mels]);
        var tokens = new float[batchSize * Tokens];
        for (var i = 0; i < tokens.Length; i++) tokens[i] = _random.Next(0, StartOfTranscript);
        _label = new Tensor<float>(tokens, [batchSize, Tokens]);
    }

    public void Forward() => _ = _model.Predict(_input);

    public void Backward()
    {
        // WhisperModel.Train is the whole step: teacher-forced forward, cross-entropy, backward and the update.
        _model.Train(_input, _label);
        _trained = true;
    }

    public void Step()
    {
        // The optimizer step ran inside Backward's Train call.
    }

    public double? LastLoss => _trained ? _model.GetLastLoss() : null;
}

/// <summary>
/// The "faster-rcnn" row: the shipped <see cref="AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN.FasterRCNN{T}"/>
/// (ResNet-50 + FPN, 90 classes plus background) on 320x320 images. A training step is the RPN plus RoI-head loss for
/// one image with two boxes, under the model's paper SGD (lr 1e-3, momentum 0.9); inference is the full Detect pipeline.
/// benchmark.py's <c>FasterRcnnR50Fpn</c> is torchvision's fasterrcnn_resnet50_fpn at the same size. The trainable
/// layouts match layer for layer except the RPN objectness: Ren et al.'s 2k-way softmax here, torchvision's k sigmoid
/// scores there (771 parameters).
/// </summary>
internal sealed class AiDotNetFasterRcnnModel : IBenchmarkModel
{
    internal const int ImageSize = 320;
    internal const int Classes = 90;
    internal const int BoxesPerImage = 2;

    private readonly AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN.FasterRCNN<float> _model;
    private readonly Random _random;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _label = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetFasterRcnnModel(int seed)
    {
        _random = new Random(seed);
        _model = new AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN.FasterRCNN<float>(
            new ObjectDetectionOptions<float>
            {
                Size = ModelSize.Medium, NumClasses = Classes, InputSize = [ImageSize, ImageSize], Seed = seed,
            });

        LoadSyntheticBatch(1);
        Forward();
        // Trainable parameters, as benchmark.py counts them: ParameterCount also includes batch normalization's
        // running statistics, which no optimizer step touches.
        ParameterCount = _model.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    /// <summary>The label holds, per image, rows of (class, x1, y1, x2, y2) in pixels, as benchmark.py's make_batch.</summary>
    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _label);
        set => (_input, _label) = value;
    }

    public void LoadSyntheticBatch(int batchSize)
    {
        var pixels = new float[batchSize * 3 * ImageSize * ImageSize];
        for (var i = 0; i < pixels.Length; i++) pixels[i] = (float)_random.NextDouble();
        _input = new Tensor<float>(pixels, [batchSize, 3, ImageSize, ImageSize]);

        var boxes = new float[batchSize * BoxesPerImage * 5];
        for (var b = 0; b < batchSize * BoxesPerImage; b++)
        {
            double cx = _random.NextDouble() * 0.5 + 0.25, cy = _random.NextDouble() * 0.5 + 0.25;
            double w = _random.NextDouble() * 0.3 + 0.1, h = _random.NextDouble() * 0.3 + 0.1;
            boxes[b * 5] = _random.Next(0, Classes);
            boxes[b * 5 + 1] = (float)((cx - w / 2) * ImageSize);
            boxes[b * 5 + 2] = (float)((cy - h / 2) * ImageSize);
            boxes[b * 5 + 3] = (float)((cx + w / 2) * ImageSize);
            boxes[b * 5 + 4] = (float)((cy + h / 2) * ImageSize);
        }
        _label = new Tensor<float>(boxes, [batchSize, BoxesPerImage, 5]);
    }

    public void Forward() => _ = _model.Detect(_input);

    public void Backward()
    {
        if (_input.Shape[0] != 1)
            throw new InvalidOperationException(
                "FasterRCNN trains one image per step (its detection head classifies one image's proposals); run the " +
                "faster-rcnn row with --batch-size 1.");

        var targets = new List<AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<float>>();
        for (var b = 0; b < BoxesPerImage; b++)
        {
            float x1 = _label[0, b, 1] / ImageSize, y1 = _label[0, b, 2] / ImageSize;
            float x2 = _label[0, b, 3] / ImageSize, y2 = _label[0, b, 4] / ImageSize;
            targets.Add(new AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<float>(
                (int)_label[0, b, 0], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1));
        }

        // TrainDetections is the whole step: forward, RPN and RoI losses, backward and the SGD update.
        _model.TrainDetections(_input, new AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<float>(new[] { targets }));
        _trained = true;
    }

    public void Step()
    {
        // The optimizer step ran inside Backward's TrainDetections call.
    }

    public double? LastLoss => _trained ? _model.GetLastLoss() : null;
}
