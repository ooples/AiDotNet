using AiDotNet.FederatedLearning.Benchmarks.Leaf;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Data loader that reads LEAF benchmark JSON splits and exposes both aggregated (X, Y) data and per-client partitions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LEAF is a federated learning benchmark suite where each user corresponds to one client.
/// This loader preserves that structure through <see cref="ClientData"/> while also providing
/// aggregated <see cref="InputOutputDataLoaderBase{T, TInput, TOutput}.Features"/>/<see cref="InputOutputDataLoaderBase{T, TInput, TOutput}.Labels"/>
/// for compatibility with the standard training facade.
/// </para>
/// <para><b>For Beginners:</b> Use this loader when you want to run federated learning with realistic per-user splits
/// provided by LEAF datasets.
/// </para>
/// </remarks>
public sealed class LeafFederatedDataLoader<T> :
    InputOutputDataLoaderBase<T, Matrix<T>, Vector<T>>,
    IFederatedClientDataLoader<T, Matrix<T>, Vector<T>>
{
    private readonly string _trainFilePath;
    private readonly string? _testFilePath;
    private readonly LeafFederatedDatasetLoadOptions _options;

    private IReadOnlyDictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>>? _clientData;
    private IReadOnlyDictionary<int, string>? _clientIdToUserId;
    private LeafFederatedSplit<Matrix<T>, Vector<T>>? _trainSplit;
    private LeafFederatedSplit<Matrix<T>, Vector<T>>? _testSplit;

    private int _totalCount;
    private int _featureCount;

    /// <inheritdoc/>
    public override string Name => "LeafFederatedDataLoader";

    /// <inheritdoc/>
    public override string Description => "LEAF federated benchmark data loader (aggregated + per-client)";

    /// <inheritdoc/>
    public override int TotalCount => _totalCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => 1;

    /// <inheritdoc/>
    public IReadOnlyDictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>> ClientData
    {
        get
        {
            EnsureLoaded();
            return _clientData!;
        }
    }

    /// <summary>
    /// Gets the mapping from internal client IDs (0..N-1) to original LEAF user IDs.
    /// </summary>
    public IReadOnlyDictionary<int, string> ClientIdToUserId
    {
        get
        {
            EnsureLoaded();
            return _clientIdToUserId!;
        }
    }

    /// <summary>
    /// Gets the loaded training split (one dataset per LEAF user).
    /// </summary>
    public LeafFederatedSplit<Matrix<T>, Vector<T>> TrainSplit
    {
        get
        {
            EnsureLoaded();
            return _trainSplit!;
        }
    }

    /// <summary>
    /// Gets the loaded optional test split (one dataset per LEAF user).
    /// </summary>
    public LeafFederatedSplit<Matrix<T>, Vector<T>>? TestSplit
    {
        get
        {
            EnsureLoaded();
            return _testSplit;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LeafFederatedDataLoader{T}"/> class from LEAF JSON files.
    /// </summary>
    /// <param name="trainFilePath">Path to the LEAF train JSON file.</param>
    /// <param name="testFilePath">Optional path to the LEAF test JSON file.</param>
    /// <param name="options">Optional load options (subset, validation).</param>
    public LeafFederatedDataLoader(
        string trainFilePath,
        string? testFilePath = null,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(trainFilePath))
        {
            throw new ArgumentException("Train file path cannot be null/empty.", nameof(trainFilePath));
        }

        _trainFilePath = trainFilePath;
        _testFilePath = testFilePath;
        _options = options ?? new LeafFederatedDatasetLoadOptions();
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        var leafLoader = new LeafFederatedDatasetLoader<T>();
        var dataset = leafLoader.LoadDatasetFromFiles(_trainFilePath, _testFilePath, _options);

        _trainSplit = dataset.Train;
        _testSplit = dataset.Test;

        var clientData = _trainSplit.ToClientIdDictionary(out var clientIdToUserId);
        _clientData = clientData;
        _clientIdToUserId = clientIdToUserId;

        var (features, labels, totalCount, featureCount) = ConcatenateClientData(clientData);
        _totalCount = totalCount;
        _featureCount = featureCount;

        LoadedFeatures = features;
        LoadedLabels = labels;
        InitializeIndices(_totalCount);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = null;
        LoadedLabels = null;
        Indices = null;

        _clientData = null;
        _clientIdToUserId = null;
        _trainSplit = null;
        _testSplit = null;

        _totalCount = 0;
        _featureCount = 0;
    }

    /// <inheritdoc/>
    protected override (Matrix<T> Features, Vector<T> Labels) ExtractBatch(int[] indices)
    {
        EnsureLoaded();

        var matrix = LoadedFeatures!;
        var vector = LoadedLabels!;

        var batchX = new Matrix<T>(indices.Length, matrix.Columns);
        var batchY = new Vector<T>(indices.Length);

        for (int i = 0; i < indices.Length; i++)
        {
            int rowIndex = indices[i];
            batchX.SetRow(i, matrix.GetRow(rowIndex));
            batchY[i] = vector[rowIndex];
        }

        return (batchX, batchY);
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Matrix<T>, Vector<T>> Train,
        IInputOutputDataLoader<T, Matrix<T>, Vector<T>> Validation,
        IInputOutputDataLoader<T, Matrix<T>, Vector<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_totalCount, trainRatio, validationRatio);

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.ThreadSafeRandom;

        var shuffledIndices = Enumerable.Range(0, _totalCount).OrderBy(_ => random.Next()).ToArray();

        var trainIndices = shuffledIndices.Take(trainSize).ToArray();
        var valIndices = shuffledIndices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffledIndices.Skip(trainSize + valSize).ToArray();

        var trainLoader = CreateSubsetLoader(trainIndices);
        var valLoader = CreateSubsetLoader(valIndices);
        var testLoader = CreateSubsetLoader(testIndices);

        return (trainLoader, valLoader, testLoader);
    }

    private InMemoryDataLoader<T, Matrix<T>, Vector<T>> CreateSubsetLoader(int[] indices)
    {
        EnsureLoaded();

        var fullX = LoadedFeatures!;
        var fullY = LoadedLabels!;

        var subsetX = new Matrix<T>(indices.Length, fullX.Columns);
        var subsetY = new Vector<T>(indices.Length);

        for (int i = 0; i < indices.Length; i++)
        {
            int rowIndex = indices[i];
            subsetX.SetRow(i, fullX.GetRow(rowIndex));
            subsetY[i] = fullY[rowIndex];
        }

        return new InMemoryDataLoader<T, Matrix<T>, Vector<T>>(subsetX, subsetY);
    }

    private static (Matrix<T> Features, Vector<T> Labels, int TotalCount, int FeatureCount) ConcatenateClientData(
        IReadOnlyDictionary<int, FederatedClientDataset<Matrix<T>, Vector<T>>> clientData)
    {
        if (clientData is null)
        {
            throw new ArgumentNullException(nameof(clientData));
        }

        if (clientData.Count == 0)
        {
            throw new ArgumentException("Client data cannot be empty.", nameof(clientData));
        }

        int totalSamples = 0;
        int featureCount = -1;

        foreach (var dataset in clientData.Values)
        {
            if (dataset is null)
            {
                throw new ArgumentException("Client data cannot contain null datasets.", nameof(clientData));
            }

            totalSamples += dataset.SampleCount;

            if (featureCount < 0)
            {
                featureCount = dataset.Features.Columns;
            }
            else if (dataset.Features.Columns != featureCount)
            {
                throw new InvalidOperationException(
                    $"LEAF client datasets must have consistent feature counts. Expected {featureCount} but found {dataset.Features.Columns}.");
            }
        }

        if (featureCount < 0)
        {
            throw new InvalidOperationException("Client data must contain at least one dataset.");
        }

        var x = new Matrix<T>(totalSamples, featureCount);
        var y = new Vector<T>(totalSamples);

        int row = 0;
        foreach (var dataset in clientData.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value))
        {
            for (int i = 0; i < dataset.SampleCount; i++)
            {
                x.SetRow(row, dataset.Features.GetRow(i));
                y[row] = dataset.Labels[i];
                row++;
            }
        }

        return (x, y, totalSamples, featureCount);
    }
}

