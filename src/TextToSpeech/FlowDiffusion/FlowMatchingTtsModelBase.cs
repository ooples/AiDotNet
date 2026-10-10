using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// The text-guided speech-infilling conditional flow matching shared by E2 TTS and F5-TTS (Chen et al. 2024, §2–3;
/// reference SWivid/F5-TTS <c>model/cfm.py</c>): characters padded with filler tokens to the mel length, a masked copy of
/// the mel as the audio condition, a backbone predicting the flow <c>x₁ − x₀</c>, classifier-free guidance and sway
/// sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Training draws x₀ ~ N(0, I) and t ~ U[0, 1], forms <c>ψ_t = (1 − t) x₀ + t x₁</c>, zeroes a random contiguous span of
/// 70–100 % of the frames in the audio condition, drops the audio condition with probability 0.3 and both audio and text
/// with probability 0.2 (the unconditional model for CFG), and minimizes the mean squared error to the flow over the
/// masked span. Synthesis integrates the predicted field with Euler steps over sway-sampled flow steps
/// <c>t = u + s (cos(π u / 2) − 1 + u)</c> (Eq. 7), with CFG <c>v + α (v − v_uncond)</c> (Eq. 6), keeping the prompt
/// frames. With <see cref="TtsModelBase{T}.Voice"/> carrying a reference mel and its transcript tokens the reference
/// prefixes both the condition and the text and sets the speaking rate; otherwise the condition is silent and the length
/// is the character count times <c>FramesPerCharacter</c>.
/// </para>
/// </remarks>
public abstract class FlowMatchingTtsModelBase<T> : TtsModelBase<T>, ITrainingObjectiveProvider<T>
{
    private Random _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);

    /// <summary>Creates the model.</summary>
    protected FlowMatchingTtsModelBase(NeuralNetworkArchitecture<T> architecture) : base(architecture) { }

    /// <summary>The flow-matching hyperparameters of the concrete model's options.</summary>
    protected abstract FlowMatchingSettings Settings { get; }

    /// <summary>The optimizer training steps use.</summary>
    protected abstract IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer { get; }

    /// <summary>Whether the paper's backbone was built (false for caller-supplied layers).</summary>
    protected abstract bool HasPaperBackbone { get; }

    /// <summary>The backbone's predicted flow <c>[frames, mel]</c> for the noisy mel, the audio condition, the
    /// character tokens and the flow step.</summary>
    protected abstract Tensor<T> PredictFlow(Tensor<T> noisy, Tensor<T> condition, Tensor<T> tokens, double t, bool dropAudio, bool dropText);

    /// <inheritdoc />
    /// <remarks>Characters are vocabulary ids in [0, V); the embedding shifts them by one to keep row 0 for the filler.</remarks>
    public override AiDotNet.NeuralNetworks.Layers.LayerInputDomain GetInputDomain(int[]? inputShape)
        => HasPaperBackbone ? AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Indices(Settings.VocabularySize) : base.GetInputDomain(inputShape);

    /// <summary>Re-seeds the training draws.</summary>
    protected void SeedTraining(int seed) => _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    // ---------------------------------------------------------------- training

    private sealed record Draw(double Time, Tensor<T> Noise, int SpanStart, int SpanLength, bool DropAudio, bool DropText);

    private Draw DrawTraining(int frames, Random random)
    {
        var s = Settings;
        double time = random.NextDouble();
        var noise = Gaussian(new[] { frames, s.MelChannels }, random);
        double fraction = s.MaskFractionMin + random.NextDouble() * (s.MaskFractionMax - s.MaskFractionMin);
        int length = Math.Max(1, Math.Min(frames, (int)(fraction * frames)));
        int start = (int)(random.NextDouble() * (frames - length + 1));
        bool dropAudio = random.NextDouble() < s.AudioDropProbability;
        bool dropBoth = random.NextDouble() < s.ConditionDropProbability;
        return new Draw(time, noise, Math.Min(start, frames - length), length, dropAudio || dropBoth, dropBoth);
    }

    private (Tensor<T> Tokens, Tensor<T> Mel) Prepare(Tensor<T> input, Tensor<T> target)
    {
        var tokens = input.Rank == 2 && input.Shape[0] == 1 ? Engine.Reshape(input, new[] { input.Shape[1] }) : input;
        var mel = target.Rank == 3 ? Engine.Reshape(target, new[] { target.Shape[1], target.Shape[2] }) : target;
        if (tokens.Rank != 1 || mel.Rank != 2 || mel.Shape[1] != Settings.MelChannels)
            throw new ArgumentException(
                $"Expected characters [tokens] and a mel spectrogram [frames, {Settings.MelChannels}], got [{string.Join(", ", input.Shape)}] and [{string.Join(", ", target.Shape)}].");
        return (tokens, mel);
    }

    /// <summary>The infilling flow-matching loss (reference <c>CFM.forward</c>): MSE between the predicted and the true
    /// flow over the masked span.</summary>
    private Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel, Draw draw)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        var span = new Tensor<T>(new[] { frames, channels });
        for (int f = draw.SpanStart; f < draw.SpanStart + draw.SpanLength; f++)
            for (int c = 0; c < channels; c++) span[f, c] = NumOps.One;
        var keep = Engine.TensorAddScalar(Engine.TensorNegate(span), NumOps.One);
        var condition = Engine.TensorMultiply(mel, keep);
        var noisy = Engine.TensorAdd(Engine.TensorMultiplyScalar(draw.Noise, NumOps.FromDouble(1 - draw.Time)),
            Engine.TensorMultiplyScalar(mel, NumOps.FromDouble(draw.Time)));
        var flow = Engine.TensorSubtract(mel, draw.Noise);
        var predicted = PredictFlow(noisy, condition, tokens, draw.Time, draw.DropAudio, draw.DropText);
        var error = Engine.TensorMultiply(Engine.TensorSubtract(predicted, flow), span);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(error, error), new[] { 0, 1 }, keepDims: false),
            NumOps.FromDouble(1.0 / (draw.SpanLength * (double)channels)));
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperBackbone)
        {
            TrainWithTape(input, expectedOutput, TrainingOptimizer);
            return;
        }
        var (tokens, mel) = Prepare(input, expectedOutput);
        var draw = DrawTraining(mel.Shape[0], _trainingRandom);
        TrainWithCustomObjective(tokens, mel, (x, y) => Objective(x, y, draw), TrainingOptimizer);
    }

    /// <inheritdoc />
    /// <remarks>The model needs nothing beyond characters and the recording's mel spectrogram.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        Train(sample.Tokens, DeriveAcousticTargets(sample).Mel);
        return LastLoss ?? NumOps.Zero;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
        => ((ITrainingObjectiveProvider<T>)this).EvaluateTrainingObjective(sample.Tokens, DeriveAcousticTargets(sample).Mel);

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <remarks>The flow step, the noise, the span and the drops are fixed by the sampling seed so the same parameters
    /// always score the same.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var (tokens, mel) = Prepare(input, target);
        var draw = DrawTraining(mel.Shape[0], AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Settings.SamplingSeed));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(tokens, mel, draw)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- inference

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperBackbone)
        {
            var x = input;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        if (input.Rank == 2 && input.Shape[0] == 1)
            input = Engine.Reshape(input, new[] { input.Shape[1] });
        if (input.Rank != 1)
            throw new ArgumentException($"Expected characters [tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));
        return Generate(input);
    }

    /// <summary>
    /// Sampling (reference <c>CFM.sample</c>): the prompt (if any) prefixes text and condition; x₀ ~ N(0, I) over the
    /// total length; Euler steps over sway-sampled t with CFG; the prompt frames are restored and cut off.
    /// </summary>
    private Tensor<T> Generate(Tensor<T> tokens)
    {
        var s = Settings;
        using var _ = new NoGradScope<T>();
        var reference = Voice?.Reference;
        var referenceTokens = Voice?.ReferenceTokens;
        bool prompted = reference is not null && referenceTokens is not null && referenceTokens.Length > 0;
        int promptFrames = prompted ? reference!.Shape[0] : 0;
        var text = prompted ? Engine.TensorConcatenate(new[] { referenceTokens!, tokens }, 0) : tokens;
        int generated = prompted
            ? (int)(promptFrames / (double)referenceTokens!.Length * tokens.Length / s.Speed)
            : (int)Math.Ceiling(tokens.Length * s.FramesPerCharacter / s.Speed);
        int frames = Math.Max(promptFrames + generated, Math.Max(text.Length, promptFrames) + 1);

        var condition = new Tensor<T>(new[] { frames, s.MelChannels });
        for (int f = 0; f < promptFrames; f++)
            for (int c = 0; c < s.MelChannels; c++) condition[f, c] = reference![f, c];

        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(s.SamplingSeed);
        var x = Gaussian(new[] { frames, s.MelChannels }, random);
        int steps = Math.Max(1, s.Steps);
        var times = new double[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double u = (double)i / steps;
            times[i] = u + s.SwayCoefficient * (Math.Cos(Math.PI / 2 * u) - 1 + u);
        }
        for (int i = 0; i < steps; i++)
        {
            var v = PredictFlow(x, condition, text, times[i], false, false);
            if (s.CfgStrength > 1e-5)
            {
                var unconditional = PredictFlow(x, condition, text, times[i], true, true);
                v = Engine.TensorAdd(v, Engine.TensorMultiplyScalar(Engine.TensorSubtract(v, unconditional), NumOps.FromDouble(s.CfgStrength)));
            }
            x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(v, NumOps.FromDouble(times[i + 1] - times[i])));
        }
        return Engine.TensorSlice(x, new[] { promptFrames, 0 }, new[] { frames - promptFrames, s.MelChannels });
    }

    /// <summary>The flow-matching hyperparameters.</summary>
    protected sealed record FlowMatchingSettings(
        int VocabularySize, int MelChannels, double MaskFractionMin, double MaskFractionMax, double AudioDropProbability,
        double ConditionDropProbability, int Steps, double CfgStrength, double SwayCoefficient, double FramesPerCharacter,
        double Speed, int SamplingSeed);
}
