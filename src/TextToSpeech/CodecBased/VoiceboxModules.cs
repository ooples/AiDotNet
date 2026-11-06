using System.Text.RegularExpressions;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

internal static class VoiceboxOps<T>
{
    internal static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    {
        owner.Add(layer);
        return layer;
    }

    internal static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    internal static int[] ToIds(Tensor<T> tensor, int vocabulary)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
            if (ids[i] < 0 || ids[i] >= vocabulary)
                throw new ArgumentOutOfRangeException(nameof(tensor), $"Voicebox phone ids lie in [0, {vocabulary}).");
        }
        return ids;
    }
}

/// <summary>Voicebox's OT-CFM equations and Appendix A.3 duration transformation.</summary>
internal static class VoiceboxObjectives<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    internal static double TransformDuration(int frames, double uniformDraw)
    {
        if (frames < 0 || uniformDraw < 0 || uniformDraw >= 1) throw new ArgumentOutOfRangeException(nameof(frames));
        // A zero-duration phone can have a negative dequantized draw; log1p remains defined down to -0.5.
        return Math.Log(1.0 + frames + uniformDraw - .5);
    }

    internal static (Tensor<T> State, Tensor<T> Velocity) Flow(IEngine engine, Tensor<T> target, Tensor<T> prior,
        double time, double sigmaMinimum)
    {
        var state = engine.TensorAdd(engine.TensorMultiplyScalar(prior, Ops.FromDouble(1 - (1 - sigmaMinimum) * time)),
            engine.TensorMultiplyScalar(target, Ops.FromDouble(time)));
        var velocity = engine.TensorSubtract(target, engine.TensorMultiplyScalar(prior, Ops.FromDouble(1 - sigmaMinimum)));
        return (state, velocity);
    }

    internal static Tensor<T> MaskedMse(IEngine engine, Tensor<T> difference, IReadOnlyList<bool> frameMask)
    {
        int channels = difference.Shape[1];
        var mask = new Tensor<T>(difference.Shape.ToArray());
        int count = 0;
        for (int f = 0; f < frameMask.Count; f++)
            if (frameMask[f])
            {
                count++;
                for (int c = 0; c < channels; c++) mask[f, c] = Ops.One;
            }
        return engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(
            engine.TensorMultiply(difference, difference), mask), new[] { 0, 1 }, false),
            Ops.FromDouble(1.0 / Math.Max(1, count * channels)));
    }
}

/// <summary>
/// English fallback for the paper's unpublished MFA-derived phone inventory. It preserves the two published
/// representation rules: a potentially zero-duration SIL between every pair of words and B/I/E/S word-position
/// suffixes on every lexical phone. Converted checkpoints must use this stable table or provide pre-tokenized ids.
/// </summary>
internal sealed class VoiceboxPhoneTokenizer
{
    private static readonly Regex Words = new(@"[\p{L}\p{N}']+", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly string[] Vowels =
        ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"];
    private static readonly string[] Consonants =
        ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"];
    private static readonly string[] Positions = ["_B", "_I", "_E", "_S"];
    private readonly Dictionary<string, int> _ids;
    private readonly int _maximum;
    private readonly bool _customInventory;

    internal VoiceboxPhoneTokenizer(int vocabularySize, IReadOnlyDictionary<string, int>? vocabulary = null)
    {
        _maximum = vocabularySize;
        _customInventory = vocabulary is not null;
        _ids = vocabulary is null ? BuildIds() : vocabulary.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        if (!_ids.ContainsKey("SIL") || _ids.Keys.Any(string.IsNullOrWhiteSpace) || _ids.Values.Any(id => id < 0 || id >= vocabularySize) ||
            _ids.Values.Distinct().Count() != _ids.Count)
            throw new ArgumentException("Phone IDs must be unique, in range, and include SIL.", nameof(vocabulary));
        if (_ids.Count > vocabularySize)
            throw new ArgumentException($"The fixed Voicebox English phone table needs {_ids.Count} rows, not {vocabularySize}.", nameof(vocabularySize));
    }

    internal int SilenceId => _ids["SIL"];
    internal int TableSize => _ids.Count;

    internal IReadOnlyList<string> Phonemize(string text,
        Func<string, IReadOnlyList<IReadOnlyList<string>>>? wordPhonemizer = null, bool allowNativeFallback = true)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must contain a word.", nameof(text));
        if (_customInventory || wordPhonemizer is not null)
        {
            if (!_customInventory || wordPhonemizer is null)
                throw new InvalidOperationException("MFA text processing requires both per-word pronunciations and their matching phone vocabulary.");
            var pronunciations = wordPhonemizer(text);
            if (pronunciations is null || pronunciations.Count == 0 || pronunciations.Any(word => word is null || word.Count == 0 || word.Any(string.IsNullOrWhiteSpace)))
                throw new ArgumentException("MFA pronunciations must contain nonempty words and phone symbols.", nameof(wordPhonemizer));
            var positioned = new List<string> { "SIL" };
            foreach (var word in pronunciations)
            {
                for (int i = 0; i < word.Count; i++)
                {
                    string suffix = word.Count == 1 ? "_S" : i == 0 ? "_B" : i == word.Count-1 ? "_E" : "_I";
                    positioned.Add(word[i] + suffix);
                }
                positioned.Add("SIL");
            }
            return positioned;
        }
        if (!allowNativeFallback)
            throw new InvalidOperationException("Voicebox raw text requires an MFA-compatible per-word phonemizer and PhoneVocabulary. UseNativeEnglishFallback explicitly selects ARPAbet for compatible independently trained models.");
        var words = Words.Matches(text).Cast<Match>().Select(match => match.Value).ToArray();
        if (words.Length == 0) throw new ArgumentException("Text must contain a pronounceable word.", nameof(text));
        var result = new List<string> { "SIL" };
        for (int w = 0; w < words.Length; w++)
        {
            if (w > 0) result.Add("SIL");
            var phones = G2pEn.Default.Word(words[w])
                .Where(phone => phone != " " && phone.Any(char.IsLetter))
                .ToArray();
            if (phones.Length == 0) phones = ["<unk>"];
            for (int i = 0; i < phones.Length; i++)
            {
                string suffix = phones.Length == 1 ? "_S" : i == 0 ? "_B" : i == phones.Length - 1 ? "_E" : "_I";
                result.Add(phones[i].ToUpperInvariant() + suffix);
            }
        }
        result.Add("SIL");
        return result;
    }

    internal int[] Encode(string text, int limit,
        Func<string, IReadOnlyList<IReadOnlyList<string>>>? wordPhonemizer = null, bool allowNativeFallback = true)
    {
        var phones = Phonemize(text, wordPhonemizer, allowNativeFallback);
        if (_customInventory && phones.Count > limit) throw new ArgumentException("The complete MFA phone sequence exceeds MaxTextLength.", nameof(text));
        int count = Math.Min(phones.Count, limit);
        var result = new int[count];
        for (int i = 0; i < count; i++)
        {
            if (_ids.TryGetValue(phones[i], out int id) && id < _maximum) result[i] = id;
            else if (_customInventory) throw new KeyNotFoundException("The configured MFA inventory lacks " + phones[i] + ".");
            else result[i] = 1;
        }
        return result;
    }

    private static Dictionary<string, int> BuildIds()
    {
        var ids = new Dictionary<string, int>(StringComparer.Ordinal) { ["SIL"] = 0, ["<unk>"] = 1 };
        void Add(string phone)
        {
            foreach (string position in Positions) ids[phone + position] = ids.Count;
        }
        foreach (string vowel in Vowels)
            for (int stress = 0; stress <= 2; stress++) Add(vowel + stress);
        foreach (string consonant in Consonants) Add(consonant);
        return ids;
    }
}

/// <summary>Two grouped same-width convolutions that form Voicebox's convolutional positional embedding.</summary>
internal sealed class VoiceboxConvPosition<T>
{
    private readonly IEngine _engine;
    private readonly List<NormedConv1DLayer<T>> _convolutions = new();

    internal VoiceboxConvPosition(IEngine engine, List<LayerBase<T>> owner, int dim, int width, int groups, int depth)
    {
        _engine = engine;
        for (int i = 0; i < depth; i++)
        {
            _convolutions.Add(VoiceboxOps<T>.Own(owner,
                new NormedConv1DLayer<T>(dim, dim, width, 1, 1, groups, width / 2, false, ConvolutionNormalization.Weight)));
        }
    }

    internal Tensor<T> Forward(Tensor<T> input)
    {
        int length = input.Shape[0], dim = input.Shape[1];
        var x = _engine.Reshape(_engine.TensorTranspose(input), new[] { 1, dim, length });
        for (int i = 0; i < _convolutions.Count; i++) x = ActivationHelper.ExactGelu(_convolutions[i].Forward(x), _engine);
        return _engine.TensorAdd(input, _engine.TensorTranspose(_engine.Reshape(x, new[] { dim, length })));
    }
}

/// <summary>A pre-norm bidirectional Transformer block with Voicebox's symmetric ALiBi.</summary>
internal sealed class VoiceboxTransformerBlock<T>
{
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly int _heads;
    private readonly double[] _slopes;
    private readonly DropoutLayer<T>? _attentionDropout;
    private readonly DropoutLayer<T>? _activationDropout;
    private readonly LayerNormalizationLayer<T> _attentionNorm;
    private readonly DenseLayer<T> _query;
    private readonly DenseLayer<T> _key;
    private readonly DenseLayer<T> _value;
    private readonly DenseLayer<T> _attentionOut;
    private readonly LayerNormalizationLayer<T> _feedForwardNorm;
    private readonly DenseLayer<T> _feedForwardIn;
    private readonly DenseLayer<T> _feedForwardOut;

    internal VoiceboxTransformerBlock(IEngine engine, List<LayerBase<T>> owner, int dim, int heads, int feedForward,
        double attentionDropout, double activationDropout)
    {
        _engine = engine;
        _dim = dim;
        _heads = heads;
        _slopes = Slopes(heads);
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _attentionNorm = VoiceboxOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        _query = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        _key = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        _value = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        _attentionOut = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        if (attentionDropout > 0) _attentionDropout = VoiceboxOps<T>.Own(owner, new DropoutLayer<T>(attentionDropout));
        _feedForwardNorm = VoiceboxOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        _feedForwardIn = VoiceboxOps<T>.Own(owner,
            new DenseLayer<T>(feedForward, identity, biasMode: BiasMode.Always));
        if (activationDropout > 0) _activationDropout = VoiceboxOps<T>.Own(owner, new DropoutLayer<T>(activationDropout));
        _feedForwardOut = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
    }

    internal Tensor<T> Forward(Tensor<T> input, bool lastIsFlowStep)
    {
        var normalized = _attentionNorm.Forward(input);
        int length = normalized.Shape[0], headDim = _dim / _heads;
        Tensor<T> Heads(Tensor<T> projected) => _engine.TensorPermute(
            _engine.Reshape(projected, new[] { length, _heads, headDim }), new[] { 1, 0, 2 });
        var q = _engine.TensorMultiplyScalar(Heads(_query.Forward(normalized)),
            VoiceboxOps<T>.NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));
        var k = Heads(_key.Forward(normalized));
        var v = Heads(_value.Forward(normalized));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));
        var bias = new Tensor<T>(new[] { _heads, length, length });
        int flow = lastIsFlowStep ? length - 1 : -1;
        for (int h = 0; h < _heads; h++)
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                    bias[h, i, j] = VoiceboxOps<T>.NumOps.FromDouble(i == flow || j == flow ? 0 : -_slopes[h] * Math.Abs(i - j));
        var weights = _engine.Softmax(_engine.TensorAdd(scores, bias), 2);
        if (_attentionDropout is not null) weights = _attentionDropout.Forward(weights);
        var attended = _engine.BatchMatMul(weights, v);
        attended = _engine.Reshape(_engine.TensorPermute(attended, new[] { 1, 0, 2 }), new[] { length, _dim });
        var x = _engine.TensorAdd(input, _attentionOut.Forward(attended));
        var hidden = ActivationHelper.ExactGelu(_feedForwardIn.Forward(_feedForwardNorm.Forward(x)), _engine);
        if (_activationDropout is not null) hidden = _activationDropout.Forward(hidden);
        return _engine.TensorAdd(x, _feedForwardOut.Forward(hidden));
    }

    private static double[] Slopes(int heads)
    {
        int n = 1 << (int)Math.Floor(Math.Log(heads, 2) + 1e-9);
        var result = new double[heads];
        double baseSlope = Math.Pow(2.0, -8.0 / n);
        for (int i = 0; i < n; i++) result[i] = Math.Pow(baseSlope, i + 1);
        if (n < heads)
        {
            double extra = Math.Pow(2.0, -4.0 / n);
            for (int i = n; i < heads; i++) result[i] = Math.Pow(extra, 2.0 * (i - n) + 1);
        }
        return result;
    }
}

/// <summary>Voicebox's Transformer encoder with symmetric U-Net skip concatenations.</summary>
internal sealed class VoiceboxTransformer<T>
{
    private readonly IEngine _engine;
    private readonly VoiceboxConvPosition<T> _position;
    private readonly List<VoiceboxTransformerBlock<T>> _blocks = new();
    private readonly List<DenseLayer<T>> _skipProjections = new();
    private readonly LayerNormalizationLayer<T> _finalNorm;

    internal VoiceboxTransformer(IEngine engine, List<LayerBase<T>> owner, int dim, int heads, int feedForward, int layers,
        int convWidth, int convGroups, int convDepth, double attentionDropout, double activationDropout)
    {
        _engine = engine;
        _position = new VoiceboxConvPosition<T>(engine, owner, dim, convWidth, convGroups, convDepth);
        for (int i = 0; i < layers; i++)
            _blocks.Add(new VoiceboxTransformerBlock<T>(engine, owner, dim, heads, feedForward,
                attentionDropout, activationDropout));
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        for (int i = 0; i < layers / 2; i++)
            _skipProjections.Add(VoiceboxOps<T>.Own(owner,
                new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always)));
        _finalNorm = VoiceboxOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
    }

    internal int LayerCount => _blocks.Count;

    internal Tensor<T> Forward(Tensor<T> input, bool lastIsFlowStep)
    {
        var x = _position.Forward(input);
        int half = _blocks.Count / 2;
        var skips = new List<Tensor<T>>(half);
        for (int i = 0; i < _blocks.Count; i++)
        {
            if (i >= half)
            {
                var skip = skips[_blocks.Count - 1 - i];
                x = _skipProjections[i - half].Forward(_engine.TensorConcatenate(new[] { x, skip }, 1));
            }
            x = _blocks[i].Forward(x, lastIsFlowStep);
            if (i < half) skips.Add(x);
        }
        return _finalNorm.Forward(x);
    }
}

/// <summary>The conditional audio vector field v_t(x_t, x_context, phones).</summary>
internal sealed class VoiceboxAudioModel<T>
{
    private readonly IEngine _engine;
    private readonly VoiceboxOptions _options;
    private readonly EmbeddingLayer<T> _phones;
    private readonly DenseLayer<T> _input;
    private readonly VoiceboxTransformer<T> _transformer;
    private readonly DenseLayer<T> _output;

    internal VoiceboxAudioModel(IEngine engine, List<LayerBase<T>> owner, VoiceboxOptions options)
    {
        _engine = engine;
        _options = options;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _phones = VoiceboxOps<T>.Own(owner, new EmbeddingLayer<T>(options.PhoneVocabularySize, options.PhoneEmbeddingDim));
        _input = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(options.AudioModelDim, identity, biasMode: BiasMode.Always));
        _transformer = new VoiceboxTransformer<T>(engine, owner, options.AudioModelDim, options.AudioHeads,
            options.AudioFeedForwardDim, options.AudioLayers, options.AudioConvPositionWidth,
            options.AudioConvPositionGroups, options.AudioConvPositionDepth, options.AudioAttentionDropout,
            options.AudioActivationDropout);
        _output = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(options.MelChannels, identity, biasMode: BiasMode.Always));
    }

    internal int LayerCount => _transformer.LayerCount;

    internal Tensor<T> Forward(Tensor<T> state, Tensor<T> context, IReadOnlyList<int> phoneIds, double time,
        bool conditioned = true)
    {
        int frames = state.Shape[0];
        if (context.Shape[0] != frames || phoneIds.Count != frames)
            throw new ArgumentException("Voicebox audio state, context, and frame-level phones must have the same length.");
        var embedded = conditioned ? _phones.Forward(VoiceboxOps<T>.Ids(phoneIds))
            : new Tensor<T>(new[] { frames, _options.PhoneEmbeddingDim });
        var audioContext = conditioned ? context : new Tensor<T>(context.Shape.ToArray());
        var projected = _input.Forward(_engine.TensorConcatenate(new[] { state, audioContext, embedded }, 1));
        var timeRow = new Tensor<T>(new[] { 1, _options.AudioModelDim });
        double scaled = time * _options.FlowTimeScale;
        for (int i = 0; i < _options.AudioModelDim; i += 2)
        {
            double frequency = Math.Exp(-Math.Log(10000.0) * i / _options.AudioModelDim);
            timeRow[0, i] = VoiceboxOps<T>.NumOps.FromDouble(Math.Sin(scaled * frequency));
            if (i + 1 < _options.AudioModelDim)
                timeRow[0, i + 1] = VoiceboxOps<T>.NumOps.FromDouble(Math.Cos(scaled * frequency));
        }
        var sequence = _engine.TensorConcatenate(new[] { projected, timeRow }, 0);
        var hidden = _transformer.Forward(sequence, lastIsFlowStep: true);
        return _output.Forward(_engine.TensorSlice(hidden, new[] { 0, 0 }, new[] { frames, _options.AudioModelDim }));
    }
}

/// <summary>The paper's default regression duration model g(l_context, phones).</summary>
internal sealed class VoiceboxDurationModel<T>
{
    private readonly IEngine _engine;
    private readonly VoiceboxOptions _options;
    private readonly EmbeddingLayer<T> _phones;
    private readonly DenseLayer<T> _input;
    private readonly VoiceboxTransformer<T> _transformer;
    private readonly DenseLayer<T> _output;

    internal VoiceboxDurationModel(IEngine engine, List<LayerBase<T>> owner, VoiceboxOptions options)
    {
        _engine = engine;
        _options = options;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _phones = VoiceboxOps<T>.Own(owner, new EmbeddingLayer<T>(options.PhoneVocabularySize, options.PhoneEmbeddingDim));
        _input = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(options.DurationModelDim, identity, biasMode: BiasMode.Always));
        _transformer = new VoiceboxTransformer<T>(engine, owner, options.DurationModelDim, options.DurationHeads,
            options.DurationFeedForwardDim, options.DurationLayers, options.DurationConvPositionWidth,
            options.DurationConvPositionGroups, options.DurationConvPositionDepth, options.DurationAttentionDropout,
            options.DurationActivationDropout);
        _output = VoiceboxOps<T>.Own(owner, new DenseLayer<T>(1, identity, biasMode: BiasMode.Always));
    }

    internal int LayerCount => _transformer.LayerCount;

    internal Tensor<T> Forward(IReadOnlyList<int> phoneIds, Tensor<T> transformedContext, bool conditioned = true)
    {
        int count = phoneIds.Count;
        if (transformedContext.Shape[0] != count)
            throw new ArgumentException("Voicebox duration context and phones must have the same length.");
        var embedded = conditioned ? _phones.Forward(VoiceboxOps<T>.Ids(phoneIds))
            : new Tensor<T>(new[] { count, _options.PhoneEmbeddingDim });
        var context = conditioned ? _engine.Reshape(transformedContext, new[] { count, 1 })
            : new Tensor<T>(new[] { count, 1 });
        var input = _input.Forward(_engine.TensorConcatenate(new[] { context, embedded }, 1));
        return _output.Forward(_transformer.Forward(input, lastIsFlowStep: false));
    }
}
