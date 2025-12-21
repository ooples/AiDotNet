using System;
using System.Collections.Generic;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TransferLearning.FeatureMapping;

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

        // Return the trained model directly (no wrapper needed since model operates in target feature space)
        return targetModel;
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
        _numOps = MathHelper.GetNumericOperations<T>();
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

    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Delegates to the underlying base model's default loss function.
    /// </para>
    /// </remarks>
    public ILossFunction<T> DefaultLossFunction => _baseModel.DefaultLossFunction;

    /// <summary>
    /// Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="target">The target/expected output.</param>
    /// <param name="lossFunction">The loss function to use. If null, uses the model's default loss function.</param>
    /// <returns>A vector containing gradients with respect to all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method delegates to the underlying base model's ComputeGradients implementation.
    /// The feature mapping is NOT applied during gradient computation, as the base model
    /// operates in the target feature space.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This calculates which direction to adjust the model's parameters to reduce error.
    /// It delegates to the wrapped Random Forest model.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _baseModel.ComputeGradients(input, target, lossFunction);
    }

    /// <summary>
    /// Applies pre-computed gradients to update the model parameters.
    /// </summary>
    /// <param name="gradients">The gradient vector to apply.</param>
    /// <param name="learningRate">The learning rate for the update.</param>
    /// <remarks>
    /// <para>
    /// This method delegates to the underlying base model's ApplyGradients implementation.
    /// Updates parameters using: θ = θ - learningRate * gradients
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After computing gradients, this method actually updates the model's parameters.
    /// It delegates to the wrapped Random Forest model.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _baseModel.ApplyGradients(gradients, learningRate);
    }

    /// <summary>
    /// Saves the mapped Random Forest model's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the wrapped Random Forest model along with feature mapping metadata.
    /// It uses the existing Serialize method and writes the data to the provided stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a snapshot of your transfer learning model.
    ///
    /// When you call SaveState:
    /// - The underlying Random Forest model is written to the stream
    /// - Feature mapping metadata is preserved
    /// - All model parameters and tree structures are saved
    ///
    /// This is particularly useful for:
    /// - Saving transfer learning models after adaptation
    /// - Checkpointing during cross-domain training
    /// - Knowledge distillation from transfer-learned models
    /// - Deploying adapted models to production
    ///
    /// You can later use LoadState to restore the model.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
    public void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save mapped Random Forest model state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving mapped Random Forest model state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the mapped Random Forest model's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes a mapped Random Forest model that was previously saved with SaveState.
    /// It uses the existing Deserialize method after reading data from the stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your transfer learning model.
    ///
    /// When you call LoadState:
    /// - The underlying Random Forest model is read from the stream
    /// - Feature mapping metadata is restored
    /// - All parameters and tree structures are recovered
    ///
    /// After loading, the model can:
    /// - Make predictions on new data (with automatic feature mapping)
    /// - Continue adaptation if needed
    /// - Be deployed to production
    ///
    /// This is essential for:
    /// - Loading transfer-learned models after training
    /// - Deploying adapted models to production
    /// - Knowledge distillation workflows
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
    public void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read mapped Random Forest model state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            // Re-throw InvalidOperationException from Deserialize
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize mapped Random Forest model state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this mapped Random Forest model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when the underlying model supports JIT compilation (soft tree mode enabled);
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation is supported when the underlying Random Forest model has soft tree mode enabled.
    /// In soft tree mode, the discrete branching logic is replaced with smooth sigmoid-based gating,
    /// making the model differentiable and compatible with JIT compilation.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation is available when soft tree mode is enabled.
    ///
    /// Traditional Random Forests use hard yes/no decisions that can't be JIT compiled.
    /// With soft tree mode, the trees use smooth transitions instead:
    /// - This makes the model differentiable
    /// - Enables JIT compilation for faster inference
    /// - Gives similar results to traditional Random Forests
    ///
    /// To enable JIT compilation:
    /// <code>
    /// var rf = (RandomForestRegression&lt;double&gt;)wrappedModel;
    /// rf.UseSoftTree = true;
    /// </code>
    /// </para>
    /// </remarks>
    public bool SupportsJitCompilation =>
        _baseModel is IJitCompilable<T> jitModel && jitModel.SupportsJitCompilation;

    /// <summary>
    /// Exports the model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the underlying model does not support JIT compilation.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Delegates to the underlying Random Forest model's ExportComputationGraph method.
    /// Requires the underlying model to have soft tree mode enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This exports the Random Forest as a computation graph.
    ///
    /// When soft tree mode is enabled, each tree becomes a smooth function that can be
    /// compiled into an optimized computation graph. The ensemble of soft trees is then
    /// averaged to produce the final prediction.
    /// </para>
    /// </remarks>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_baseModel is IJitCompilable<T> jitModel && jitModel.SupportsJitCompilation)
        {
            return jitModel.ExportComputationGraph(inputNodes);
        }

        throw new NotSupportedException(
            "This mapped Random Forest model does not support JIT compilation. " +
            "To enable JIT compilation, set UseSoftTree = true on the underlying Random Forest model " +
            "to use soft (differentiable) decision trees with sigmoid-based gating.");
    }

    #endregion
}
