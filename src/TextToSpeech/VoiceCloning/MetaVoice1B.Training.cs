using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class MetaVoice1B<T>
{
    internal double? TrainingSpeakerDrawForAudit { get; set; }
    protected override TtsSupervision RequiredSupervision => HasPaperPipeline
        ? TtsSupervision.CodecTokens | TtsSupervision.ReferenceRecording : TtsSupervision.None;
    public override bool TrainsOnSampleObjective => HasPaperPipeline;
    public override int CodecTokenCodebooks => 8;

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
        => _first is null ? parameters : TapeTrainingStep<T>.CollectParameters(
            _first.FineTuneLayers(_options.FineTuneLastBlocks).Cast<ILayer<T>>().ToList());

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        if (_first is null || _optimizer is null) throw new NotSupportedException("MetaVoice fine-tuning requires the native causal GPT.");
        if (_usesDefaultOptimizer && _optimizer is AdamWOptimizer<T, Tensor<T>, Tensor<T>> adam)
        {
            var parameters = TapeTrainingStep<T>.CollectParameters(_first.FineTuneLayers(_options.FineTuneLastBlocks).Cast<ILayer<T>>().ToList());
            var options = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)adam.GetOptions();
            options.MaskParameters = parameters;
            options.WeightDecayMask = new Vector<T>(parameters.SelectMany(parameter =>
                Enumerable.Repeat(parameter.Rank >= 2 ? NumOps.One : NumOps.Zero, parameter.Length)));
        }
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildFirstStageObjective(sample), _optimizer);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        bool previous = IsTrainingMode; SetTrainingMode(false);
        try { using var noGrad = new NoGradScope<T>(); return BuildFirstStageObjective(sample)[0]; }
        finally { SetTrainingMode(previous); }
    }

    private Tensor<T> BuildFirstStageObjective(TtsTrainingSample<T> sample)
    {
        var model = _first ?? throw new NotSupportedException("The native causal GPT is unavailable.");
        int[] text = sample.Text is { } rawText ? EncodeText(rawText) : IntegerIds(sample.Tokens);
        if (text.Length == 0 || text.Any(token => token < 2049 || token > 2561))
            throw new ArgumentException("Supply the first-stage byte-BPE text IDs or the original Text.", nameof(sample));
        var codes = TrainingCodes(sample);
        var sequence = new List<int>(text);
        for (int frame = 0; frame < codes.GetLength(1); frame++)
        {
            sequence.Add(codes[0, frame]); sequence.Add(codes[1, frame] + 1024);
        }
        // Original combine_tokens_flattened_interleaved -> pad_tokens(ctx+1,2048) -> shift.
        while (sequence.Count < _options.FirstStageContext + 1) sequence.Add(2048);
        var input = sequence.Take(_options.FirstStageContext).ToArray();
        var target = sequence.Skip(1).Take(_options.FirstStageContext).ToArray();
        Tensor<T> speaker;
        using (new NoGradScope<T>())
        {
            var reference = sample.SpeakerReference;
            if (reference is not null && reference.Length == SpeakerEmbeddingDim)
                speaker = new Tensor<T>(new[] { 1, SpeakerEmbeddingDim }, new Vector<T>(reference.ToArray()));
            else
            {
                var audio = reference ?? sample.Audio ?? throw new ArgumentException("Supply a speaker recording or embedding.", nameof(sample));
                speaker = Engine.Reshape(ExtractSpeakerEmbedding(audio, reference is null ? _options.SampleRate : _options.ReferenceSampleRate), new[] { 1, SpeakerEmbeddingDim });
            }
        }
        var logits = model.Forward(new[] { input }, speaker, training: IsTrainingMode,
            speakerDropoutDraw: TrainingSpeakerDrawForAudit).Logits[0];
        var weights = new Tensor<T>(logits.Shape.ToArray());
        for (int i = 0; i < target.Length; i++) weights[i, target[i]] = NumOps.FromDouble(-1.0 / target.Length);
        // Text and repeated EOA padding remain in the original fine-tuning objective.
        return Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(logits, 1), weights), new[] { 0, 1 }, true), new[] { 1 });
    }

    private int[,] TrainingCodes(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is not { } tensor)
        {
            if (_codec is null || sample.Audio is null) throw new ArgumentException("Supply codec tokens or load EnCodec and supply audio.", nameof(sample));
            using var noGrad = new NoGradScope<T>();
            return _codec.Encode(sample.Audio);
        }
        var values = IntegerIds(tensor); int rows, frames;
        // TtsTrainingSample uses frame-major codec tensors. Continue accepting
        // the author's codebook-major arrays when they are unambiguous; eight
        // columns take precedence as the library's canonical layout.
        bool frameMajor = tensor.Rank == 2 && tensor.Shape[1] == 8;
        if (frameMajor)
        { rows = 8; frames = tensor.Shape[0]; }
        else if (tensor.Rank == 2 && tensor.Shape[0] >= 2 && tensor.Shape[0] <= 8)
        { rows = tensor.Shape[0]; frames = tensor.Shape[1]; }
        else if (tensor.Rank == 1 && values.Length % 8 == 0)
        { rows = 8; frames = values.Length / 8; }
        else throw new ArgumentException("Codec tokens must be [frames,8], unambiguous [codebook,frames], or frame-major eight-codebook values.", nameof(sample));
        if (frames == 0 || values.Any(value => value >= 1024)) throw new ArgumentException("Codec IDs must be in0..1023 and nonempty.", nameof(sample));
        var codes = new int[rows, frames];
        for (int q = 0; q < rows; q++) for (int frame = 0; frame < frames; frame++)
            codes[q, frame] = values[frameMajor || tensor.Rank == 1 ? frame * rows + q : q * frames + frame];
        return codes;
    }
}
