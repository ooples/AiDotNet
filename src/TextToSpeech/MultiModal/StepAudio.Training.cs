using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class StepAudio<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;
    private string _trainedGroup = "tts";
    private LlamaNetwork<T>? _reward, _critic, _reference;
    private DenseLayer<T>? _rewardHead, _criticHead;
    private int _ppoSteps;

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer("tts", _layers.Tts) : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    private void RequireNative()
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
    }

    // ---------------------------------------------------------------- optimizers

    // The stage's rate, its floor and whether it decays (cosine): §4.2, §5.1.2, §5.2.2, §5.2.4, §5.2.6, all scaled by
    // LearningRate / 2e-5 (the paper's base rate, the default).
    private (double Rate, double Floor, bool Cosine) StageSchedule()
    {
        var (rate, floor, cosine) = PaperSchedule();
        double scale = _options.LearningRate / 2e-5;
        return (rate * scale, floor * scale, cosine);
    }

    private (double Rate, double Floor, bool Cosine) PaperSchedule() => _options.Stage switch
    {
        StepAudioTrainingStage.PreTraining1 or StepAudioTrainingStage.PreTraining2 => (2e-5, 2e-5, false),
        StepAudioTrainingStage.PreTraining3 => (2e-5, 5e-6, true),
        StepAudioTrainingStage.TtsFineTuning => (2e-5, 2e-6, true),
        StepAudioTrainingStage.ChatFineTuning => (5.656e-5, 5.656e-6, true),
        StepAudioTrainingStage.RewardModel => (1.24e-5, 6e-6, true),
        _ => (1e-6, 2e-7, true),
    };

    // AdamW (the paper names no optimizer; β (0.9, 0.95) and weight decay 0.1 are Step-1's family defaults, not stated) on
    // the stage's schedule; in pre-training stages 1 and 2 the token table and head at five times the rate.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer(string group, IEnumerable<LayerBase<T>> layers, LlamaNetwork<T>? boosted = null)
    {
        string key = $"{group}:{_options.Stage}";
        if (_optimizers.TryGetValue(key, out var existing)) return existing;
        var o = _options;
        var (rate, floor, cosine) = StageSchedule();
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-8, WeightDecay = o.WeightDecay,
            UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
            LearningRateScheduler = cosine
                ? new LinearWarmupScheduler(rate, o.WarmupSteps, Math.Max(o.TotalSteps, o.WarmupSteps + 1), 0.0, LinearWarmupScheduler.DecayMode.Cosine, floor)
                : null,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        bool boost = boosted is not null && o.Stage is StepAudioTrainingStage.PreTraining1 or StepAudioTrainingStage.PreTraining2;
        if (boost)
        {
            var fast = new HashSet<LayerBase<T>>(new[] { boosted!.Embeddings, boosted.LmHead }.Where(l => l is not null).Cast<LayerBase<T>>());
            var parameters = new List<Tensor<T>>();
            var rates = new List<T>();
            foreach (var layer in layers)
                foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { layer }, -1))
                {
                    parameters.Add(p);
                    for (int i = 0; i < p.Length; i++) rates.Add(Ops.FromDouble(fast.Contains(layer) ? 5.0 : 1.0));
                }
            var options = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
            options.LearningRateMask = new Vector<T>(rates.ToArray());
            options.MaskParameters = parameters;
        }
        return _optimizers[key] = optimizer;
    }

    // The layers a group names: the TTS or chat model, the reward model, the critic.
    private IEnumerable<LayerBase<T>> GroupLayers(string group) => group switch
    {
        "chat" or "policy" => _layers.Chat,
        "reward" => _layers.Reward,
        "critic" => _layers.Critic,
        _ => _layers.Tts,
    };

    private T Step(string group, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer, Func<Tensor<T>> objective)
    {
        _trainedGroup = group;
        var marker = IndexTtsOps.Ids<T>(new[] { 0 });
        return TrainWithCustomObjective(marker, marker, (_, _) => objective(), optimizer);
    }

    // ---------------------------------------------------------------- language-model objectives

    // The next-token cross-entropy of ids after the first `scoredFrom` (those are context only).
    private Tensor<T> SequenceLoss(LlamaNetwork<T> model, IReadOnlyList<int> ids, int scoredFrom)
    {
        if (ids.Count < 2) throw new ArgumentException("A training sequence needs at least two tokens.");
        var grid = new int[1, ids.Count];
        var labels = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            grid[0, i] = ids[i];
            labels[0, i] = i < scoredFrom ? -100 : ids[i];
        }
        var (_, logits) = model.Forward(model.Embed(grid));
        return model.CausalLmLoss(logits!, labels);
    }

    private (LlamaNetwork<T> Model, List<LayerBase<T>> Layers, string Group) PreTrainingTarget(bool tts) =>
        tts ? (Core.Tts, _layers.Tts, "tts") : (Core.ChatModel, _layers.Chat, "chat");

    private void RequirePreTraining()
    {
        if (_options.Stage is not (StepAudioTrainingStage.PreTraining1 or StepAudioTrainingStage.PreTraining2 or StepAudioTrainingStage.PreTraining3))
            throw new InvalidOperationException($"Pre-training data trains at the pre-training stages; Stage is {_options.Stage}.");
    }

    /// <summary>
    /// One pre-training step (§4.2) on a sequence of text and &lt;audio_N&gt; tokens (pure audio continuation:
    /// <see cref="AudioTokenString"/>; interleaved, ASR and TTS data: the caller's layout, whose format the paper does not
    /// state), on the chat model or, with <paramref name="tts"/>, the TTS model ("the same pre-training strategy across
    /// models of varying parameter scales"). Every token is scored. Returns the loss.
    /// </summary>
    public T TrainSequence(string sequence, bool tts = false)
    {
        RequireNative();
        RequirePreTraining();
        if (sequence is null) throw new ArgumentNullException(nameof(sequence));
        var (model, layers, group) = PreTrainingTarget(tts);
        var ids = Core.Tokens.Encode(sequence);
        return Step(group, Optimizer(group, layers, model), () => SequenceLoss(model, ids, 1));
    }

    /// <summary>A pre-training step on a recording's audio continuation (its dual-codebook tokens).</summary>
    public T TrainContinuation(Tensor<T> recording, int sampleRate, bool tts = false) =>
        TrainSequence(AudioTokenString(recording ?? throw new ArgumentNullException(nameof(recording)), sampleRate), tts);

    /// <summary>
    /// One TTS fine-tuning step (§5.1.2) on the TTS model: the two-turn prompt with the speaker (their transcript and
    /// audio tokens) and the human turn's text, then the recording's dual-codebook tokens and &lt;|EOT|&gt;, which alone are
    /// scored. Returns the loss.
    /// </summary>
    public T TrainTts(string text, Tensor<T> recording, int sampleRate, StepAudioSpeaker<T> speaker)
    {
        RequireNative();
        if (_options.Stage != StepAudioTrainingStage.TtsFineTuning) throw new InvalidOperationException($"TTS data trains at TtsFineTuning; Stage is {_options.Stage}.");
        if (text is null || recording is null || speaker is null) throw new ArgumentNullException(text is null ? nameof(text) : recording is null ? nameof(recording) : nameof(speaker));
        var prompt = Core.Tokens.TtsPrompt(text, speaker.PromptText, speaker.Name, speaker.PromptCode);
        var (audio, _, _) = Core.AudioTokens(recording, sampleRate);
        var ids = prompt.Concat(audio).Concat(new[] { StepAudioTokens.Eos }).ToArray();
        return Step("tts", Optimizer("tts", _layers.Tts), () => SequenceLoss(Core.Tts, ids, prompt.Length));
    }

    /// <summary>
    /// One AQTA fine-tuning step (§5.2.2) on the chat model: the chat prompt with earlier turns and the spoken question,
    /// then the answer's tokens and &lt;|EOT|&gt;, which alone are scored. Returns the loss.
    /// </summary>
    public T TrainChat(Tensor<T> question, int sampleRate, string answer, IReadOnlyList<(string Role, string Content)>? history = null)
    {
        RequireNative();
        if (_options.Stage != StepAudioTrainingStage.ChatFineTuning) throw new InvalidOperationException($"Chat data trains at ChatFineTuning; Stage is {_options.Stage}.");
        if (question is null || answer is null) throw new ArgumentNullException(question is null ? nameof(question) : nameof(answer));
        var (prompt, ids) = ChatExample(question, sampleRate, answer, history);
        return Step("chat", Optimizer("chat", _layers.Chat), () => SequenceLoss(Core.ChatModel, ids, prompt.Length));
    }

    private (int[] Prompt, int[] Ids) ChatExample(Tensor<T> question, int sampleRate, string answer, IReadOnlyList<(string Role, string Content)>? history)
    {
        var turns = new List<(string, string)>(history ?? Array.Empty<(string, string)>()) { ("human", AudioTokenString(question, sampleRate)) };
        var prompt = Core.Tokens.ChatPrompt(turns);
        var ids = prompt.Concat(Core.Tokens.Encode(answer, addBos: false)).Concat(new[] { StepAudioTokens.Eos }).ToArray();
        return (prompt, ids);
    }

    // The sample's objective: its tokens' text spoken by its recording, in the speaker the recording itself gives (the
    // TTS fine-tuning format; at a pre-training stage the recording's continuation).
    private Func<Tensor<T>> SampleObjective(TtsTrainingSample<T> sample, out IEnumerable<LayerBase<T>> layers, out string group)
    {
        var core = Core;
        if (sample.Audio is null) throw new ArgumentException("Step-Audio trains on the recording; set the sample's Audio.", nameof(sample));
        int rate = _options.SampleRate;
        if (_options.Stage is StepAudioTrainingStage.PreTraining1 or StepAudioTrainingStage.PreTraining2 or StepAudioTrainingStage.PreTraining3)
        {
            var sequence = core.Tokens.Encode(AudioTokenString(sample.Audio, rate));
            layers = _layers.Tts;
            group = "tts";
            return () => SequenceLoss(core.Tts, sequence, 1);
        }
        var textIds = IdsOf(sample.Tokens);
        string text = core.Tokens.Decode(textIds);
        var speaker = SpeakerVoice ?? core.CreateSpeaker(sample.Audio, rate, text, string.Empty);
        var prompt = core.Tokens.TtsPrompt(text, textIds, speaker.PromptText, speaker.Name, speaker.PromptCode);
        var (audio, _, _) = core.AudioTokens(sample.Audio, rate);
        var ids = prompt.Concat(audio).Concat(new[] { StepAudioTokens.Eos }).ToArray();
        layers = _layers.Tts;
        group = "tts";
        return () => SequenceLoss(core.Tts, ids, prompt.Length);
    }

    /// <inheritdoc />
    /// <remarks>A TTS step on the TTS model: the sample's text spoken by its recording, in <see cref="SpeakerVoice"/> (or the
    /// recording itself as the prompt speaker); at a pre-training stage, the recording's audio continuation.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequireNative();
        var objective = SampleObjective(sample, out var layers, out var group);
        return Step(group, Optimizer(group, layers, group == "tts" ? Core.Tts : null), objective);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequireNative();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var objective = SampleObjective(sample, out _, out _);
            using var scope = new NoGradScope<T>();
            return objective().GetFlat(0);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- reward model and PPO

    private void BuildRlhfModels()
    {
        var o = _options;
        LlamaConfiguration Config(bool head) => new()
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.ChatLayers, Heads = o.NumHeads, KeyValueHeads = o.ChatGroups,
            Intermediate = o.ChatIntermediate, RmsEpsilon = o.RmsEpsilon, SqrtAlibi = true, IncludeHead = head,
        };
        if (o.ChatLayers <= 0) throw new ArgumentException("RLHF trains the chat model; ChatLayers must be positive.");
        _reward = new LlamaNetwork<T>(Engine, _layers.Reward, Config(head: false));
        _rewardHead = ScalarHead(_layers.Reward);
        _critic = new LlamaNetwork<T>(Engine, _layers.Critic, Config(head: false));
        _criticHead = ScalarHead(_layers.Critic);
        _reference = new LlamaNetwork<T>(Engine, _layers.Reference, Config(head: true));
        CopyInto(_layers.Chat, _layers.Reward);
        CopyInto(_layers.Chat, _layers.Critic);
        CopyInto(_layers.Chat, _layers.Reference);
    }

    private DenseLayer<T> ScalarHead(List<LayerBase<T>> layers)
    {
        var head = new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>());
        TorchParameters.Linear(Engine, head, _options.HiddenDim, 1, new double[_options.HiddenDim], new double[1]);
        layers.Add(head);
        return head;
    }

    // Copies the first layers of `to` from `from` in order (the Step-1 layers are built in the same order; a backbone
    // without a head stops before it).
    private void CopyInto(IReadOnlyList<LayerBase<T>> from, IReadOnlyList<LayerBase<T>> to)
    {
        for (int i = 0; i < to.Count && i < from.Count; i++)
        {
            if (to[i] is DenseLayer<T> && from[i] is not DenseLayer<T>) break;
            var source = from[i].GetParameters();
            if (source.Length != to[i].ParameterCount) break;
            to[i].SetParameters(source);
        }
    }

    private void RequireRlhf()
    {
        if (_reward is null) throw new InvalidOperationException("RLHF needs StepAudioOptions.IncludeRlhfModels.");
    }

    /// <summary>Starts the reward model from the current chat model (§5.2.4: "initializes from a SFT model") and the
    /// critic and the frozen reference policy likewise.</summary>
    public void InitializeRlhfFromChat()
    {
        RequireNative();
        RequireRlhf();
        CopyInto(_layers.Chat, _layers.Reward);
        CopyInto(_layers.Chat, _layers.Critic);
        CopyInto(_layers.Chat, _layers.Reference);
        _ppoSteps = 0;
    }

    /// <summary>Starts the critic from the trained reward model.</summary>
    public void InitializeCriticFromReward()
    {
        RequireNative();
        RequireRlhf();
        CopyInto(_layers.Reward, _layers.Critic);
    }

    // The scalar head's value at every position of ids [L] ([L]).
    private Tensor<T> Values(LlamaNetwork<T> backbone, DenseLayer<T> head, IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        var (hidden, _) = backbone.Forward(backbone.Embed(grid));
        return Engine.Reshape(head.Forward(Engine.Reshape(hidden, new[] { ids.Count, _options.HiddenDim })), new[] { ids.Count });
    }

    private Tensor<T> Last(Tensor<T> values) => Engine.TensorSlice(values, new[] { values.Length - 1 }, new[] { 1 });

    /// <summary>The reward model's score of an answer to a spoken question (its value at the last token).</summary>
    public double Reward(Tensor<T> question, int sampleRate, string answer)
    {
        RequireNative();
        RequireRlhf();
        var (_, ids) = ChatExample(question, sampleRate, answer, null);
        using var scope = new NoGradScope<T>();
        return Ops.ToDouble(Last(Values(_reward!, _rewardHead!, ids))[0]);
    }

    /// <summary>
    /// One reward-model step (§5.2.4): the Bradley–Terry loss −log σ(r(chosen) − r(rejected)) on two answers to the same
    /// spoken question. Returns the loss.
    /// </summary>
    public T TrainReward(Tensor<T> question, int sampleRate, string chosen, string rejected)
    {
        RequireNative();
        RequireRlhf();
        if (_options.Stage != StepAudioTrainingStage.RewardModel) throw new InvalidOperationException($"Preferences train at RewardModel; Stage is {_options.Stage}.");
        var (_, good) = ChatExample(question, sampleRate, chosen, null);
        var (_, bad) = ChatExample(question, sampleRate, rejected, null);
        return Step("reward", Optimizer("reward", _layers.Reward), () =>
        {
            var margin = Engine.TensorSubtract(Last(Values(_reward!, _rewardHead!, good)), Last(Values(_reward!, _rewardHead!, bad)));
            // −log σ(m) = softplus(−m) = log(1 + exp(−m)).
            return Engine.TensorLog(Engine.TensorAddScalar(Engine.TensorExp(Engine.TensorNegate(margin)), Ops.One));
        });
    }

    // Log-probabilities [n] of the response tokens ids[start..] under a model with a head.
    private Tensor<T> ResponseLogProbabilities(LlamaNetwork<T> model, IReadOnlyList<int> ids, int start)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        var (_, logits) = model.Forward(model.Embed(grid));
        int vocabulary = model.Configuration.VocabSize, n = ids.Count - start;
        var log = Engine.TensorLogSoftmax(Engine.TensorSlice(Engine.Reshape(logits!, new[] { ids.Count, vocabulary }), new[] { start - 1, 0 }, new[] { n, vocabulary }), axis: 1);
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * vocabulary + ids[start + i]).ToArray()));
        return Engine.Reshape(Engine.TensorIndexSelect(Engine.Reshape(log, new[] { n * vocabulary, 1 }), picks, 0), new[] { n });
    }

    /// <summary>
    /// One PPO step (§5.2.6) on a spoken question: the policy (the chat model) samples an answer; the per-token reward is
    /// −β (log π − log π_ref) with the reward model's score added at the last token; GAE (γ, λ) gives the advantages
    /// (whitened) and returns; the critic minimizes ½ (V − R)², and after its warm-up steps the policy minimizes the
    /// clipped surrogate −min(ρA, clip(ρ, 1 ± ε)A). Returns the policy loss (the value loss during warm-up).
    /// </summary>
    public T TrainPpo(Tensor<T> question, int sampleRate)
    {
        RequireNative();
        RequireRlhf();
        if (_options.Stage != StepAudioTrainingStage.Ppo) throw new InvalidOperationException($"PPO trains at Ppo; Stage is {_options.Stage}.");
        var o = _options;
        var core = Core;
        var prompt = core.Tokens.ChatPrompt(new[] { ("human", AudioTokenString(question, sampleRate)) });
        var response = core.Generate(core.ChatModel, prompt, o.PpoResponseLength, o.ChatTemperature, o.ChatTopP, TrainingRandom, repetitionAware: false);
        if (response.Count < o.PpoResponseLength) response.Add(StepAudioTokens.Eos);
        var ids = prompt.Concat(response).ToArray();
        int start = prompt.Length, n = response.Count;
        double[] old, reference, values;
        double score;
        using (new NoGradScope<T>())
        {
            old = ResponseLogProbabilities(core.ChatModel, ids, start).ToArray().Select(v => Ops.ToDouble(v)).ToArray();
            reference = ResponseLogProbabilities(_reference!, ids, start).ToArray().Select(v => Ops.ToDouble(v)).ToArray();
            var all = Values(_critic!, _criticHead!, ids);
            values = Enumerable.Range(0, n).Select(i => Ops.ToDouble(all[start - 1 + i])).ToArray();
            score = Ops.ToDouble(Last(Values(_reward!, _rewardHead!, ids))[0]);
        }
        var rewards = Enumerable.Range(0, n).Select(i => -o.KlCoefficient * (old[i] - reference[i])).ToArray();
        rewards[n - 1] += score;
        var advantages = new double[n];
        double running = 0;
        for (int t = n - 1; t >= 0; t--)
        {
            double next = t + 1 < n ? values[t + 1] : 0;
            double delta = rewards[t] + o.Gamma * next - values[t];
            running = delta + o.Gamma * o.Lambda * running;
            advantages[t] = running;
        }
        var returns = advantages.Select((a, t) => a + values[t]).ToArray();
        double mean = advantages.Average(), std = Math.Sqrt(advantages.Select(a => (a - mean) * (a - mean)).Average() + 1e-8);
        var whitened = advantages.Select(a => (a - mean) / std).ToArray();
        Tensor<T> Vector(double[] v) => new(new[] { v.Length }, new Vector<T>(v.Select(x => Ops.FromDouble(x)).ToArray()));

        var valueLoss = Step("critic", Optimizer("critic", _layers.Critic), () =>
        {
            var all = Values(_critic!, _criticHead!, ids);
            var predicted = Engine.TensorSlice(all, new[] { start - 1 }, new[] { n });
            var error = Engine.TensorSubtract(predicted, Vector(returns));
            return Engine.TensorMultiplyScalar(Engine.ReduceMean(Engine.TensorMultiply(error, error), new[] { 0 }, false), Ops.FromDouble(0.5));
        });
        _ppoSteps++;
        if (_ppoSteps <= o.CriticWarmupSteps) return valueLoss;
        var oldTensor = Vector(old);
        var advantage = Vector(whitened);
        return Step("policy", Optimizer("policy", _layers.Chat), () =>
        {
            var ratio = Engine.TensorExp(Engine.TensorSubtract(ResponseLogProbabilities(core.ChatModel, ids, start), oldTensor));
            var unclipped = Engine.TensorMultiply(ratio, advantage);
            var clipped = Engine.TensorMultiply(Engine.TensorClamp(ratio, Ops.FromDouble(1 - o.PpoClip), Ops.FromDouble(1 + o.PpoClip)), advantage);
            return Engine.TensorNegate(Engine.ReduceMean(Engine.TensorMin(unclipped, clipped), new[] { 0 }, false));
        });
    }

    /// <inheritdoc />
    /// <remarks>Only the layers the step's objective trains: the TTS or chat model, the reward model, the critic.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(GroupLayers(_trainedGroup).Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
