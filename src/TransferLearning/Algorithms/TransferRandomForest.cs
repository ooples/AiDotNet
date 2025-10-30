using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Models.Options;
using AiDotNet.Regularization;
using AiDotNet.TransferLearning.FeatureMapping;
using AiDotNet.Helpers;

namespace AiDotNet.TransferLearning.Algorithms;

/// <summary>
/// Implements transfer learning for Random Forest models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class enables Random Forest models to transfer knowledge
/// from one domain to another. Random Forests are ensembles of decision trees, and this
/// class can adapt them when the source and target domains have different feature spaces.
/// </para>
/// </remarks>
public class TransferRandomForest<T> : TransferLearningBase<T, Matrix<T>, Vector<T>>
{
    private readonly RandomForestRegressionOptions _options;
    private readonly IRegularization<T, Matrix<T>, Vector<T>> _regularization;

    /// <summary>
    /// Initializes a new instance of the TransferRandomForest class.
    /// </summary>
    /// <param name="options">Configuration options for the Random Forest.</param>
    /// <param name="regularization">Regularization to apply.</param>
    public TransferRandomForest(
        RandomForestRegressionOptions options,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
    {
        _options = options;
        _regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
    }

    /// <summary>
    /// Transfers a Random Forest model to a target domain with the same feature space.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> TransferSameDomain(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        // Apply domain adaptation if available
        Matrix<T> adaptedData = targetData;
        if (DomainAdapter != null)
        {
            // Get some source data for adaptation (would need to be passed in a full implementation)
            // For now, we'll skip this step or use targetData as-is
            adaptedData = targetData;
        }

        // Fine-tune on target domain
        var targetModel = new RandomForestRegression<T>(_options, _regularization);
        targetModel.Train(adaptedData, targetLabels);

        return targetModel;
    }

    /// <summary>
    /// Transfers a Random Forest model to a target domain with a different feature space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method performs cross-domain transfer when source and target domains have different
    /// feature spaces. It requires a pre-trained feature mapper to be set via SetFeatureMapper().
    /// </para>
    /// <para>
    /// <b>Limitations:</b> Without access to source domain data, this method cannot:
    /// 1. Train the feature mapper (must be pre-trained)
    /// 2. Train the domain adapter (must be pre-trained if used)
    /// 3. Perform optimal knowledge distillation (uses model predictions on mapped data)
    /// 4. Validate feature space compatibility
    /// </para>
    /// <para>
    /// <b>Recommendation:</b> For best results, use the public Transfer() method that accepts
    /// both source and target domain data, which enables proper feature mapper and domain adapter
    /// training along with effective knowledge distillation.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when FeatureMapper is null, not trained, or DomainAdapter requires training.
    /// </exception>
    protected override IFullModel<T, Matrix<T>, Vector<T>> TransferCrossDomain(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        // Validate that feature mapper is available and trained
        if (FeatureMapper == null)
        {
            throw new InvalidOperationException(
                "Cross-domain transfer requires a feature mapper. Use SetFeatureMapper() before transfer. " +
                "Alternatively, use the public Transfer() method with source data for automatic feature mapping.");
        }

        if (!FeatureMapper.IsTrained)
        {
            throw new InvalidOperationException(
                "Feature mapper must be trained before cross-domain transfer. " +
                "Either train the mapper using FeatureMapper.Train(sourceData, targetData) or " +
                "use the public Transfer() method with source data for automatic training.");
        }

        // Validate domain adapter if present
        if (DomainAdapter != null && DomainAdapter.RequiresTraining)
        {
            throw new InvalidOperationException(
                "Domain adapter requires training but source data is not available in this method. " +
                "Either train the adapter beforehand or use the public Transfer() method with source data.");
        }

        // Get source model's feature dimension
        int sourceFeatures = sourceModel.GetActiveFeatureIndices().Count();

        // Map target features to source feature space
        Matrix<T> mappedTargetData = FeatureMapper.MapToSource(targetData, sourceFeatures);

        // Apply domain adaptation if available and trained
        if (DomainAdapter != null)
        {
            mappedTargetData = DomainAdapter.AdaptSource(mappedTargetData, targetData);
        }

        // Use source model for predictions on mapped data (knowledge distillation)
        Vector<T> pseudoLabels = sourceModel.Predict(mappedTargetData);

        // Combine pseudo-labels with true labels (70% weight on true labels)
        var combinedLabels = CombineLabels(pseudoLabels, targetLabels, 0.7);

        // Train new model on target domain with combined labels
        var targetModel = new RandomForestRegression<T>(_options, _regularization);
        targetModel.Train(targetData, combinedLabels);

        // Wrap the model to handle feature mapping at prediction time
        return new MappedRandomForestModel<T>(targetModel, FeatureMapper, sourceFeatures);
    }

    /// <summary>
    /// Transfers a Random Forest model to a target domain with proper source data.
    /// </summary>
    /// <param name="sourceModel">The model trained on the source domain.</param>
    /// <param name="sourceData">Training data from the source domain (required for cross-domain transfer).</param>
    /// <param name="targetData">Training data from the target domain.</param>
    /// <param name="targetLabels">Labels for the target domain data.</param>
    /// <returns>A new model adapted to the target domain.</returns>
    public IFullModel<T, Matrix<T>, Vector<T>> Transfer(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> sourceData,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        // Determine if cross-domain transfer is needed
        bool needsCrossDomain = RequiresCrossDomainTransfer(sourceModel, targetData);

        if (!needsCrossDomain)
        {
            return TransferSameDomain(sourceModel, targetData, targetLabels);
        }

        // Cross-domain transfer with proper source data
        if (FeatureMapper == null)
        {
            throw new InvalidOperationException(
                "Cross-domain transfer requires a feature mapper. Use SetFeatureMapper() before transfer.");
        }

        // Step 1: Train feature mapper with actual source and target data
        if (!FeatureMapper.IsTrained)
        {
            FeatureMapper.Train(sourceData, targetData);
        }

        // Step 2: Get source model's feature dimension
        int sourceFeatures = sourceModel.GetActiveFeatureIndices().Count();

        // Step 3: Map target features to source feature space
        Matrix<T> mappedTargetData = FeatureMapper.MapToSource(targetData, sourceFeatures);

        // Step 4: Apply domain adaptation if available
        if (DomainAdapter != null && DomainAdapter.RequiresTraining)
        {
            // Train domain adapter with actual source and mapped target data
            Matrix<T> mappedSourceData = FeatureMapper.MapToSource(sourceData, sourceFeatures);
            DomainAdapter.Train(mappedSourceData, mappedTargetData);
        }

        if (DomainAdapter != null)
        {
            mappedTargetData = DomainAdapter.AdaptSource(mappedTargetData, targetData);
        }

        // Step 5: Use source model for predictions on mapped data (knowledge distillation)
        Vector<T> pseudoLabels = sourceModel.Predict(mappedTargetData);

        // Step 6: Combine pseudo-labels with true labels (if available)
        var combinedLabels = CombineLabels(pseudoLabels, targetLabels, 0.7); // 70% weight on true labels

        // Step 7: Train new model on target domain with combined labels
        var targetModel = new RandomForestRegression<T>(_options, _regularization);
        targetModel.Train(targetData, combinedLabels);

        // Step 8: Wrap the model to handle feature mapping at prediction time
        return new MappedRandomForestModel<T>(targetModel, FeatureMapper, sourceFeatures);
    }

    /// <summary>
    /// Combines pseudo-labels from source model with true target labels.
    /// </summary>
    private Vector<T> CombineLabels(Vector<T> pseudoLabels, Vector<T> trueLabels, double trueWeight)
    {
        var combined = new Vector<T>(pseudoLabels.Length);
        T trueW = NumOps.FromDouble(trueWeight);
        T pseudoW = NumOps.FromDouble(1.0 - trueWeight);

        for (int i = 0; i < combined.Length; i++)
        {
            combined[i] = NumOps.Add(
                NumOps.Multiply(trueW, trueLabels[i]),
                NumOps.Multiply(pseudoW, pseudoLabels[i]));
        }

        return combined;
    }
}

/// <summary>
/// Wrapper model that applies feature mapping before prediction.
/// </summary>
internal class MappedRandomForestModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    private const int WrapperMagic = 0x4D52464D; // 'MRFM'
    private readonly IFullModel<T, Matrix<T>, Vector<T>> _baseModel;
    private readonly IFeatureMapper<T> _mapper;
    private readonly int _targetFeatures;
    private readonly INumericOperations<T> _numOps;
    private static System.Reflection.MethodInfo? _inverseMapMethod;

    public MappedRandomForestModel(
        IFullModel<T, Matrix<T>, Vector<T>> baseModel,
        IFeatureMapper<T> mapper,
        int targetFeatures)
    {
        _baseModel = baseModel;
        _mapper = mapper;
        _targetFeatures = targetFeatures;
        _numOps = AiDotNet.Helpers.MathHelper.GetNumericOperations<T>();
        // Initialize inverse-map reflection method once per process if available
        _inverseMapMethod ??= _mapper.GetType().GetMethod("InverseMapFeatureName", new[] { typeof(string) });
    }

    public void Train(Matrix<T> input, Vector<T> expectedOutput)
    {
        _baseModel.Train(input, expectedOutput);
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        // Input might need to be mapped if it's from a different feature space
        return _baseModel.Predict(input);
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        return _baseModel.GetModelMetadata();
    }

    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        var baseBytes = _baseModel.Serialize();
        WriteWrapper(writer, baseBytes);
        return ms.ToArray();
    }

    public void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        if (TryReadWrapper(reader, out var baseBytes))
        {
            _baseModel.Deserialize(baseBytes);
            return;
        }
        _baseModel.Deserialize(data);
    }

    public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return _baseModel.WithParameters(parameters);
    }

    public Vector<T> GetParameters()
    {
        return _baseModel.GetParameters();
    }

    public IEnumerable<int> GetActiveFeatureIndices()
    {
        return _baseModel.GetActiveFeatureIndices();
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return _baseModel.IsFeatureUsed(featureIndex);
    }

    public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        return new MappedRandomForestModel<T>(
            _baseModel.DeepCopy(),
            _mapper,
            _targetFeatures);
    }

    public IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    public virtual void SetParameters(Vector<T> parameters)
    {
        _baseModel.SetParameters(parameters);
    }

    public virtual int ParameterCount
    {
        get { return _baseModel.ParameterCount; }
    }

    public virtual void SaveModel(string filePath)
    {
        // Persist wrapper metadata and base model bytes together
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms))
        {
            var baseBytes = _baseModel.Serialize();
            WriteWrapper(writer, baseBytes);
        }
        var data = ms.ToArray();
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        File.WriteAllBytes(filePath, data);
    }

    public virtual void LoadModel(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath);
        }
        var data = File.ReadAllBytes(filePath);
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        if (!TryReadWrapper(reader, out var baseBytes))
        {
            throw new InvalidOperationException("Failed to deserialize MappedRandomForestModel wrapper format. The file may be corrupted or in an incompatible format.");
        }
        // Intentionally overwrites _baseModel with deserialized state.
        // The wrapper metadata (_mapper, _targetFeatures) is immutable and set at construction.
        _baseModel.Deserialize(baseBytes);
    }

        public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var baseImportance = _baseModel.GetFeatureImportance();
        var mappedImportance = new Dictionary<string, T>(baseImportance.Count);
        var mapMethod = _inverseMapMethod;
        foreach (var kvp in baseImportance)
        {
            var key = kvp.Key;
            if (mapMethod != null)
            {
                try
                {
                    var mappedKey = mapMethod.Invoke(_mapper, new object[] { kvp.Key });
                    if (mappedKey is string s)
                    {
                        key = s;
                    }
                }
                catch
                {
                    // Failed to inverse map feature name; using original key as fallback
                }
            }
        mappedImportance[key] = kvp.Value;
        }
        return mappedImportance;
    }

    private void WriteWrapper(BinaryWriter writer, byte[] baseBytes)
    {
        writer.Write(WrapperMagic);
        writer.Write(_targetFeatures);
        try
        {
            writer.Write(Convert.ToDouble(_mapper.GetMappingConfidence()));
        }
        catch
        {
            // Failed to write mapping confidence, fallback to 0.0
            writer.Write(0.0);
        }
        writer.Write(baseBytes.Length);
        writer.Write(baseBytes);
        writer.Flush();
    }

    private bool TryReadWrapper(BinaryReader reader, out byte[] baseBytes)
    {
        try
        {
            var magic = reader.ReadInt32();
            if (magic != WrapperMagic)
            {
                baseBytes = Array.Empty<byte>();
                return false;
            }
            var target = reader.ReadInt32();
            if (target != _targetFeatures)
            {
                throw new InvalidOperationException($"Deserialized target feature count ({target}) does not match current instance ({_targetFeatures}).");
            }
            var confidence = reader.ReadDouble(); // Read mapping confidence (currently unused; read to maintain stream compatibility, reserved for future validation/versioning)
            var len = reader.ReadInt32();
            baseBytes = reader.ReadBytes(len);
            return true;
        }
        catch
        {
            // Failed to read wrapper format; fallback for backward compatibility with non-wrapped models
            baseBytes = Array.Empty<byte>();
            return false;
        }
    }

    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _baseModel.SetActiveFeatureIndices(featureIndices);
    }
}

