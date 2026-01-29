using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Data;

/// <summary>
/// Data loader for financial time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinancialDataLoader turns a list of OHLCV points into windowed tensors
/// suitable for training forecasting models. It implements the standard
/// InputOutputDataLoader API so it works with AiDotNet's training pipeline.
/// </para>
/// <para>
/// <b>For Beginners:</b> This loader creates training samples like:
/// - Input: the last N time steps (lookback window)
/// - Output: the next M time steps (prediction horizon)
/// It handles batching, shuffling, and splitting into train/val/test.
/// </para>
/// </remarks>
public sealed class FinancialDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly IReadOnlyList<MarketDataPoint<T>> _series;
    private readonly FinancialPreprocessor<T> _preprocessor;
    private readonly int _sequenceLength;
    private readonly int _predictionHorizon;
    private readonly bool _includeVolume;
    private readonly bool _includeReturns;
    private readonly bool _predictReturns;
    private readonly bool _normalizeMinMax;

    private int _sampleCount;
    private int _featureCount;
    private int _outputDimension;

    /// <inheritdoc/>
    public override string Name => "FinancialDataLoader";

    /// <inheritdoc/>
    public override string Description => "Windowed time series loader for financial forecasting.";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => _outputDimension;

    /// <summary>
    /// Creates a new financial data loader.
    /// </summary>
    /// <param name="series">The OHLCV series to load.</param>
    /// <param name="sequenceLength">The lookback window size.</param>
    /// <param name="predictionHorizon">The number of steps to predict.</param>
    /// <param name="includeVolume">Whether to include volume as a feature.</param>
    /// <param name="includeReturns">Whether to include returns as a feature.</param>
    /// <param name="predictReturns">Whether to predict returns instead of prices.</param>
    /// <param name="normalizeMinMax">Whether to apply min-max normalization to features.</param>
    /// <param name="preprocessor">Optional preprocessor instance.</param>
    /// <param name="batchSize">Batch size for iteration.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Provide a list of price bars, then choose how many
    /// past steps the model sees (sequenceLength) and how many future steps it
    /// should predict (predictionHorizon).
    /// </para>
    /// </remarks>
    public FinancialDataLoader(
        IReadOnlyList<MarketDataPoint<T>> series,
        int sequenceLength,
        int predictionHorizon,
        bool includeVolume = true,
        bool includeReturns = false,
        bool predictReturns = false,
        bool normalizeMinMax = false,
        FinancialPreprocessor<T>? preprocessor = null,
        int batchSize = 32)
        : base(batchSize)
    {
        _series = series ?? throw new ArgumentNullException(nameof(series));
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _includeVolume = includeVolume;
        _includeReturns = includeReturns;
        _predictReturns = predictReturns;
        _normalizeMinMax = normalizeMinMax;
        _preprocessor = preprocessor ?? new FinancialPreprocessor<T>();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This turns the raw price list into training tensors and
    /// prepares the loader for iteration.
    /// </para>
    /// </remarks>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var (features, targets) = _preprocessor.CreateSupervisedLearningTensors(
            _series,
            _sequenceLength,
            _predictionHorizon,
            _includeVolume,
            _includeReturns,
            _predictReturns);

        if (_normalizeMinMax)
        {
            features = _preprocessor.NormalizeMinMax(features, out _);
        }

        LoadedFeatures = features;
        LoadedLabels = targets;

        _sampleCount = features.Shape[0];
        _featureCount = features.Shape.Length > 2 ? features.Shape[2] : features.Shape[1];
        _outputDimension = targets.Shape.Length > 2 ? targets.Shape[2] : targets.Shape[1];

        InitializeIndices(_sampleCount);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This clears cached tensors so memory can be reclaimed.
    /// </para>
    /// </remarks>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
        _featureCount = 0;
        _outputDimension = 0;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a mini-batch by selecting only the requested
    /// sample indices from the full dataset.
    /// </para>
    /// </remarks>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        if (LoadedFeatures == null || LoadedLabels == null)
        {
            throw new InvalidOperationException("Data is not loaded.");
        }

        var features = ExtractTensorSubset(LoadedFeatures, indices);
        var labels = ExtractTensorSubset(LoadedLabels, indices);
        return (features, labels);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This splits the dataset into train/validation/test portions
    /// so you can evaluate models fairly.
    /// </para>
    /// </remarks>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffledIndices = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        var trainIndices = shuffledIndices.Take(trainSize).ToArray();
        var valIndices = shuffledIndices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffledIndices.Skip(trainSize + valSize).ToArray();

        var trainLoader = CreateSubsetLoader(trainIndices);
        var valLoader = CreateSubsetLoader(valIndices);
        var testLoader = CreateSubsetLoader(testIndices);

        return (trainLoader, valLoader, testLoader);
    }

    /// <summary>
    /// Creates a subset loader for a list of indices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helps split the dataset into train/validation/test
    /// without reprocessing the raw series.
    /// </para>
    /// </remarks>
    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSubsetLoader(int[] indices)
    {
        if (LoadedFeatures == null || LoadedLabels == null)
        {
            throw new InvalidOperationException("Data is not loaded.");
        }

        var subsetFeatures = ExtractTensorSubset(LoadedFeatures, indices);
        var subsetLabels = ExtractTensorSubset(LoadedLabels, indices);
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(subsetFeatures, subsetLabels);
    }

    /// <summary>
    /// Extracts a subset of samples from a tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This slices out specific samples so we can build
    /// batches or dataset splits.
    /// </para>
    /// </remarks>
    private static Tensor<T> ExtractTensorSubset(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
