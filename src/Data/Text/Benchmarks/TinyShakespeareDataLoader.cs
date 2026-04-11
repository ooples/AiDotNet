using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the Tiny Shakespeare character-level language modeling dataset —
/// a small public-domain corpus standard for validating language model
/// training at low cost. Produces sequences of byte tokens (256-entry
/// vocabulary) paired with next-byte targets for causal language modeling.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This loader gives you training data for a character-
/// level language model. Each training example is a sequence of N bytes from
/// Shakespeare's works, and the target is the same sequence shifted forward
/// by one position — so the model learns "given the last N characters, what
/// comes next?" The standard benchmark from Karpathy's char-rnn tutorial.
/// </para>
/// <para>
/// When no data file is supplied, the loader uses a ~2KB public-domain
/// Shakespeare excerpt bundled directly in the source, which lets tests
/// run with zero network or filesystem dependencies. For real experiments,
/// set <see cref="TinyShakespeareDataLoaderOptions.DataPath"/> to point at
/// the full 1MB tinyshakespeare.txt file.
/// </para>
/// </remarks>
public class TinyShakespeareDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly TinyShakespeareDataLoaderOptions _options;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "TinyShakespeare";

    /// <inheritdoc/>
    public override string Description => "Karpathy-style character-level Shakespeare corpus (byte vocabulary)";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _options.SequenceLength;

    /// <inheritdoc/>
    public override int OutputDimension => _options.SequenceLength;

    /// <summary>Creates a new Tiny Shakespeare data loader.</summary>
    public TinyShakespeareDataLoader(TinyShakespeareDataLoaderOptions? options = null)
    {
        _options = options ?? new TinyShakespeareDataLoaderOptions();
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string text;
        if (!string.IsNullOrEmpty(_options.DataPath) && File.Exists(_options.DataPath))
        {
            text = await FilePolyfill.ReadAllTextAsync(_options.DataPath, cancellationToken);
        }
        else
        {
            text = BundledShakespeareExcerpt;
        }

        byte[] corpus = System.Text.Encoding.UTF8.GetBytes(text);
        if (corpus.Length < _options.SequenceLength + 2)
        {
            throw new InvalidOperationException(
                $"TinyShakespeare corpus is too short ({corpus.Length} bytes) for " +
                $"sequence length {_options.SequenceLength}. Need at least " +
                $"{_options.SequenceLength + 2} bytes.");
        }

        // Split the byte stream into train / validation regions.
        int trainEnd = (int)(corpus.Length * _options.TrainFraction);
        int splitStart, splitEnd;
        switch (_options.Split)
        {
            case Geometry.DatasetSplit.Validation:
            case Geometry.DatasetSplit.Test:
                splitStart = trainEnd;
                splitEnd = corpus.Length - 1;
                break;
            case Geometry.DatasetSplit.Train:
            default:
                splitStart = 0;
                splitEnd = trainEnd;
                break;
        }

        int seqLen = _options.SequenceLength;
        int numSequences = Math.Max(0, (splitEnd - splitStart - 1) / seqLen);
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < numSequences)
            numSequences = _options.MaxSamples.Value;

        _sampleCount = numSequences;
        var featuresData = new T[numSequences * seqLen];
        var labelsData = new T[numSequences * seqLen];

        for (int i = 0; i < numSequences; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int start = splitStart + i * seqLen;
            int featureOffset = i * seqLen;
            for (int j = 0; j < seqLen; j++)
            {
                featuresData[featureOffset + j] = NumOps.FromDouble(corpus[start + j]);
                labelsData[featureOffset + j] = NumOps.FromDouble(corpus[start + j + 1]);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { numSequences, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { numSequences, seqLen });
        InitializeIndices(numSequences);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (
            TextLoaderHelper.ExtractTensorBatch(features, indices),
            TextLoaderHelper.ExtractTensorBatch(labels, indices));
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7, double validationRatio = 0.15, int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);
        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        var features = LoadedFeatures ?? throw new InvalidOperationException("Not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                TextLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()),
                TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                TextLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                TextLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()),
                TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    /// <summary>
    /// A short Shakespeare excerpt from <i>Coriolanus</i> (public domain) used as
    /// the default corpus when no external file is provided. Lets unit tests run
    /// with zero external dependencies.
    /// </summary>
    private const string BundledShakespeareExcerpt =
        "First Citizen:\n" +
        "Before we proceed any further, hear me speak.\n\n" +
        "All:\n" +
        "Speak, speak.\n\n" +
        "First Citizen:\n" +
        "You are all resolved rather to die than to famish?\n\n" +
        "All:\n" +
        "Resolved. resolved.\n\n" +
        "First Citizen:\n" +
        "First, you know Caius Marcius is chief enemy to the people.\n\n" +
        "All:\n" +
        "We know't, we know't.\n\n" +
        "First Citizen:\n" +
        "Let us kill him, and we'll have corn at our own price.\n" +
        "Is't a verdict?\n\n" +
        "All:\n" +
        "No more talking on't; let it be done: away, away!\n\n" +
        "Second Citizen:\n" +
        "One word, good citizens.\n\n" +
        "First Citizen:\n" +
        "We are accounted poor citizens, the patricians good.\n" +
        "What authority surfeits on would relieve us: if they\n" +
        "would yield us but the superfluity, while it were\n" +
        "wholesome, we might guess they relieved us humanely;\n" +
        "but they think we are too dear: the leanness that\n" +
        "afflicts us, the object of our misery, is as an\n" +
        "inventory to particularise their abundance; our\n" +
        "sufferance is a gain to them Let us revenge this with\n" +
        "our pikes, ere we become rakes: for the gods know I\n" +
        "speak this in hunger for bread, not in thirst for revenge.\n\n" +
        "Second Citizen:\n" +
        "Would you proceed especially against Caius Marcius?\n\n" +
        "All:\n" +
        "Against him first: he's a very dog to the commonalty.\n\n" +
        "Second Citizen:\n" +
        "Consider you what services he has done for his country?\n\n" +
        "First Citizen:\n" +
        "Very well; and could be content to give him good\n" +
        "report fort, but that he pays himself with being proud.\n\n" +
        "Second Citizen:\n" +
        "Nay, but speak not maliciously.\n\n" +
        "First Citizen:\n" +
        "I say unto you, what he hath done famously, he did\n" +
        "it to that end: though soft-conscienced men can be\n" +
        "content to say it was for his country he did it to\n" +
        "please his mother and to be partly proud; which he\n" +
        "is, even till the altitude of his virtue.\n\n" +
        "Second Citizen:\n" +
        "What he cannot help in his nature, you account a\n" +
        "vice in him. You must in no way say he is covetous.\n\n" +
        "First Citizen:\n" +
        "If I must not, I need not be barren of accusations;\n" +
        "he hath faults, with surplus, to tire in repetition.\n" +
        "What shouts are these? The other side o' the city\n" +
        "is risen: why stay we prating here? to the Capitol!\n";
}
