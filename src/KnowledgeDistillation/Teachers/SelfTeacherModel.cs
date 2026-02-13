using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Self teacher model that uses the student's own predictions from earlier training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Self-distillation is a technique where a model learns from its own
/// earlier predictions. This teacher can operate in two modes:</para>
/// <list type="bullet">
/// <item><description><b>Cached Mode:</b> Uses pre-computed predictions from earlier epochs (no JIT support)</description></item>
/// <item><description><b>Model Mode:</b> Wraps an IJitCompilable model for dynamic predictions (JIT support available)</description></item>
/// </list>
/// </remarks>
public class SelfTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private Vector<T>[]? _cachedPredictions;
    private readonly int _outputDim;
    private readonly IJitCompilable<T>? _underlyingModel;

    /// <summary>
    /// Gets the output dimension of the teacher model.
    /// </summary>
    public override int OutputDimension => _outputDim;

    /// <summary>
    /// Initializes a new instance of SelfTeacherModel for cached predictions mode.
    /// </summary>
    /// <param name="outputDimension">The output dimension of predictions.</param>
    /// <remarks>
    /// <para>Use this constructor when you want to manually cache predictions via
    /// <see cref="CachePredictions"/> and retrieve them via <see cref="GetCachedPrediction"/>.
    /// JIT compilation is not supported in this mode.</para>
    /// </remarks>
    public SelfTeacherModel(int outputDimension)
    {
        _outputDim = outputDimension;
        _underlyingModel = null;
    }

    /// <summary>
    /// Initializes a new instance of SelfTeacherModel wrapping an IJitCompilable model.
    /// </summary>
    /// <param name="model">The JIT-compilable model to wrap.</param>
    /// <param name="outputDimension">The output dimension of the model.</param>
    /// <remarks>
    /// <para>Use this constructor when you want the teacher to generate predictions dynamically
    /// from the underlying model. JIT compilation is supported when the underlying model supports it.</para>
    /// <para>You can still use <see cref="CachePredictions"/> to cache predictions if needed.</para>
    /// </remarks>
    public SelfTeacherModel(IJitCompilable<T> model, int outputDimension)
    {
        Guard.NotNull(model);
        _underlyingModel = model;
        _outputDim = outputDimension;
    }

    public void CachePredictions(Vector<T>[] predictions)
    {
        if (predictions == null)
            throw new ArgumentNullException(nameof(predictions));

        if (predictions.Length == 0)
            throw new ArgumentException("Predictions array cannot be empty", nameof(predictions));

        // Validate all predictions have correct dimension
        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == null)
                throw new ArgumentException($"Prediction at index {i} is null", nameof(predictions));

            if (predictions[i].Length != _outputDim)
                throw new ArgumentException(
                    $"Prediction at index {i} has dimension {predictions[i].Length}, expected {_outputDim}",
                    nameof(predictions));
        }

        _cachedPredictions = predictions;
    }

    /// <summary>
    /// Gets logits from the underlying model.
    /// </summary>
    /// <param name="input">Input to the model.</param>
    /// <returns>The logits from the underlying model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no underlying model is configured.</exception>
    /// <remarks>
    /// <para>This method is only available when the SelfTeacherModel was constructed with an
    /// IJitCompilable model. For cached prediction mode, use <see cref="GetCachedPrediction"/>.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        if (_underlyingModel != null)
        {
            // IJitCompilable doesn't have execution methods - need to cast to a model interface
            // that has Predict. Typically IJitCompilable models also implement IModel.
            if (_underlyingModel is IModel<Vector<T>, Vector<T>, ModelMetadata<T>> model)
            {
                return model.Predict(input);
            }

            throw new InvalidOperationException(
                "Underlying model must implement IModel<Vector<T>, Vector<T>, ModelMetadata<T>> to execute predictions. " +
                "IJitCompilable only provides computation graph export for JIT compilation.");
        }

        throw new InvalidOperationException(
            "Self teacher in cached mode does not support direct input. Use GetCachedPrediction instead, " +
            "or construct with an IJitCompilable model for dynamic predictions.");
    }

    /// <summary>
    /// Gets a cached prediction by index.
    /// </summary>
    /// <param name="index">Index of the cached prediction.</param>
    /// <returns>The cached logits for the specified index.</returns>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw cached logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public Vector<T> GetCachedPrediction(int index)
    {
        if (index < 0)
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative");
        if (_cachedPredictions == null || index >= _cachedPredictions.Length)
            throw new InvalidOperationException("Predictions not cached or index out of range");
        return _cachedPredictions[index];
    }

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if constructed with an IJitCompilable model that supports JIT;
    /// <c>false</c> if using cached predictions mode.
    /// </value>
    public override bool SupportsJitCompilation => _underlyingModel?.SupportsJitCompilation ?? false;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <exception cref="NotSupportedException">Thrown when using cached predictions mode.</exception>
    /// <remarks>
    /// <para>
    /// When constructed with an IJitCompilable model, this method delegates to the underlying model's
    /// computation graph export. When using cached predictions mode, JIT compilation is not supported
    /// because there is no computation to represent as a graph.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_underlyingModel != null && _underlyingModel.SupportsJitCompilation)
        {
            return _underlyingModel.ExportComputationGraph(inputNodes);
        }

        return ThrowJitNotSupported(
            nameof(SelfTeacherModel<T>),
            "it uses cached predictions rather than a computation graph. Use the constructor with an IJitCompilable model for JIT support");
    }
}
