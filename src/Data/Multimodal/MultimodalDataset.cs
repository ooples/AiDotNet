using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Multimodal;

/// <summary>
/// A dataset of multimodal samples for training models that process multiple data types simultaneously.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// MultimodalDataset stores collections of multimodal samples and provides batching, shuffling,
/// and splitting capabilities. Each sample can contain any combination of modalities
/// (image, text, audio, etc.).
/// </para>
/// <para><b>For Beginners:</b> Use this when your model needs to process multiple types of data
/// together, like image captioning (image + text) or audio-visual tasks (audio + video).
/// <code>
/// var dataset = new MultimodalDataset&lt;float&gt;();
/// dataset.Add(new MultimodalSample&lt;float&gt;(
///     new ModalitySample&lt;float&gt;(ModalityType.Image, imageTensor),
///     new ModalitySample&lt;float&gt;(ModalityType.Text, tokenTensor)
/// ));
///
/// // Get batched data by modality
/// var (imageBatch, textBatch) = dataset.GetBatch(0, 32, "image", "text");
/// </code>
/// </para>
/// </remarks>
public class MultimodalDataset<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<MultimodalSample<T>> _samples = new List<MultimodalSample<T>>();
    private int[]? _shuffledIndices;

    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    public int Count => _samples.Count;

    /// <summary>
    /// Gets a sample at the specified index (respects shuffle order).
    /// </summary>
    /// <param name="index">The index of the sample.</param>
    public MultimodalSample<T> this[int index]
    {
        get
        {
            int actualIndex = _shuffledIndices is not null ? _shuffledIndices[index] : index;
            return _samples[actualIndex];
        }
    }

    /// <summary>
    /// Adds a multimodal sample to the dataset.
    /// </summary>
    /// <param name="sample">The multimodal sample to add.</param>
    public void Add(MultimodalSample<T> sample)
    {
        if (sample is null)
            throw new ArgumentNullException(nameof(sample));

        _samples.Add(sample);
        _shuffledIndices = null; // Invalidate shuffle
    }

    /// <summary>
    /// Adds multiple multimodal samples to the dataset.
    /// </summary>
    /// <param name="samples">The samples to add.</param>
    public void AddRange(IEnumerable<MultimodalSample<T>> samples)
    {
        if (samples is null)
            throw new ArgumentNullException(nameof(samples));

        foreach (var sample in samples)
        {
            Add(sample);
        }
    }

    /// <summary>
    /// Shuffles the dataset using the specified random seed.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public void Shuffle(int? seed = null)
    {
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        _shuffledIndices = Enumerable.Range(0, _samples.Count)
            .OrderBy(_ => random.Next())
            .ToArray();
    }

    /// <summary>
    /// Gets a batch of modality tensors by key, stacked along a new batch dimension.
    /// </summary>
    /// <param name="startIndex">The starting sample index.</param>
    /// <param name="batchSize">Number of samples in the batch.</param>
    /// <param name="modalityKey">The modality key to extract.</param>
    /// <returns>A tensor with shape [batchSize, ...modalityShape].</returns>
    /// <remarks>
    /// All samples in the batch must have the specified modality key and matching tensor shapes.
    /// </remarks>
    public Tensor<T> GetModalityBatch(int startIndex, int batchSize, string modalityKey)
    {
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        if (startIndex < 0 || startIndex >= _samples.Count)
            throw new ArgumentOutOfRangeException(nameof(startIndex));

        int endIndex = Math.Min(startIndex + batchSize, _samples.Count);
        int actualBatchSize = endIndex - startIndex;

        if (actualBatchSize == 0)
            throw new InvalidOperationException("Batch size is zero.");

        // Get the first sample to determine shape
        var firstSample = this[startIndex];
        if (!firstSample.HasKey(modalityKey))
        {
            throw new KeyNotFoundException(
                $"Modality key '{modalityKey}' not found in sample at index {startIndex}.");
        }

        int[] sampleShape = firstSample[modalityKey].Shape;
        int elementsPerSample = 1;
        for (int d = 0; d < sampleShape.Length; d++)
        {
            elementsPerSample *= sampleShape[d];
        }

        // Build batch shape: [batchSize, ...sampleShape]
        int[] batchShape = new int[sampleShape.Length + 1];
        batchShape[0] = actualBatchSize;
        Array.Copy(sampleShape, 0, batchShape, 1, sampleShape.Length);

        var result = new Tensor<T>(batchShape);
        var resultSpan = result.Data.Span;

        for (int i = 0; i < actualBatchSize; i++)
        {
            var sample = this[startIndex + i];
            if (!sample.HasKey(modalityKey))
            {
                throw new KeyNotFoundException(
                    $"Modality key '{modalityKey}' not found in sample at index {startIndex + i}.");
            }

            var modalitySample = sample[modalityKey];
            if (!modalitySample.Shape.SequenceEqual(sampleShape))
            {
                throw new InvalidOperationException(
                    $"Modality '{modalityKey}' shape mismatch at index {startIndex + i}. " +
                    $"Expected [{string.Join(", ", sampleShape)}] but got [{string.Join(", ", modalitySample.Shape)}].");
            }

            var modalityData = modalitySample.Data;
            if (modalityData.Data.Length != elementsPerSample)
            {
                throw new InvalidOperationException(
                    $"Modality '{modalityKey}' data length mismatch at index {startIndex + i}. " +
                    $"Expected {elementsPerSample} elements but got {modalityData.Data.Length}.");
            }

            var srcSpan = modalityData.Data.Span;

            int dstOffset = i * elementsPerSample;
            srcSpan.Slice(0, elementsPerSample).CopyTo(resultSpan.Slice(dstOffset, elementsPerSample));
        }

        return result;
    }

    /// <summary>
    /// Gets a batch of label tensors.
    /// </summary>
    /// <param name="startIndex">The starting sample index.</param>
    /// <param name="batchSize">Number of samples in the batch.</param>
    /// <returns>A tensor of labels for the batch.</returns>
    public Tensor<T>? GetLabelBatch(int startIndex, int batchSize)
    {
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        if (startIndex < 0 || startIndex >= _samples.Count)
            throw new ArgumentOutOfRangeException(nameof(startIndex));

        int endIndex = Math.Min(startIndex + batchSize, _samples.Count);
        int actualBatchSize = endIndex - startIndex;

        var firstSample = this[startIndex];
        if (firstSample.Label is null) return null;

        int[] labelShape = firstSample.Label.Shape;
        int elementsPerLabel = 1;
        for (int d = 0; d < labelShape.Length; d++)
        {
            elementsPerLabel *= labelShape[d];
        }

        int[] batchShape = new int[labelShape.Length + 1];
        batchShape[0] = actualBatchSize;
        Array.Copy(labelShape, 0, batchShape, 1, labelShape.Length);

        var result = new Tensor<T>(batchShape);
        var resultSpan = result.Data.Span;

        for (int i = 0; i < actualBatchSize; i++)
        {
            var sample = this[startIndex + i];
            if (sample.Label is null)
                throw new InvalidOperationException(
                    $"Sample at index {startIndex + i} has null label but batch expects labels.");

            var srcSpan = sample.Label.Data.Span;
            if (srcSpan.Length != elementsPerLabel)
            {
                throw new InvalidOperationException(
                    $"Label at index {startIndex + i} has {srcSpan.Length} elements, expected {elementsPerLabel}.");
            }

            int dstOffset = i * elementsPerLabel;
            srcSpan.CopyTo(resultSpan.Slice(dstOffset, elementsPerLabel));
        }

        return result;
    }

    /// <summary>
    /// Splits the dataset into train, validation, and test sets.
    /// </summary>
    /// <param name="trainRatio">Fraction of data for training (e.g., 0.7).</param>
    /// <param name="validationRatio">Fraction of data for validation (e.g., 0.15).</param>
    /// <param name="seed">Optional random seed for reproducible splits.</param>
    /// <returns>Three MultimodalDataset instances for train, validation, and test.</returns>
    public (MultimodalDataset<T> Train, MultimodalDataset<T> Validation, MultimodalDataset<T> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        if (trainRatio < 0 || trainRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(trainRatio), "Must be between 0 and 1.");
        if (validationRatio < 0 || validationRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(validationRatio), "Must be between 0 and 1.");
        if (trainRatio + validationRatio > 1.0)
            throw new ArgumentException("trainRatio + validationRatio must not exceed 1.0.");

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        var indices = Enumerable.Range(0, _samples.Count)
            .OrderBy(_ => random.Next())
            .ToArray();

        int trainSize = (int)Math.Round(_samples.Count * trainRatio);
        int valSize = (int)Math.Round(_samples.Count * validationRatio);

        var train = new MultimodalDataset<T>();
        var val = new MultimodalDataset<T>();
        var test = new MultimodalDataset<T>();

        for (int i = 0; i < indices.Length; i++)
        {
            var sample = _samples[indices[i]];
            if (i < trainSize)
                train.Add(sample);
            else if (i < trainSize + valSize)
                val.Add(sample);
            else
                test.Add(sample);
        }

        return (train, val, test);
    }

    /// <summary>
    /// Iterates over the dataset in batches.
    /// </summary>
    /// <param name="batchSize">The number of samples per batch.</param>
    /// <returns>An enumerable of (startIndex, batchSize) tuples.</returns>
    public IEnumerable<(int StartIndex, int BatchSize)> GetBatchIndices(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");

        for (int start = 0; start < _samples.Count; start += batchSize)
        {
            int actualSize = Math.Min(batchSize, _samples.Count - start);
            yield return (start, actualSize);
        }
    }

    /// <summary>
    /// Gets all unique modality types present across all samples.
    /// </summary>
    /// <returns>A set of modality types.</returns>
    public HashSet<ModalityType> GetPresentModalities()
    {
        var result = new HashSet<ModalityType>();
        foreach (var sample in _samples)
        {
            foreach (var modality in sample.Modalities)
            {
                result.Add(modality.Modality);
            }
        }

        return result;
    }

    /// <summary>
    /// Gets all unique modality keys present across all samples.
    /// </summary>
    /// <returns>A set of modality keys.</returns>
    public HashSet<string> GetPresentKeys()
    {
        var result = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var sample in _samples)
        {
            foreach (var key in sample.Keys)
            {
                result.Add(key);
            }
        }

        return result;
    }
}
