using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Born Again Neural Networks",
    "https://arxiv.org/abs/1805.04770",
    Year = 2018,
    Authors = "Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, et al.")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class SelfTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private Vector<T>[]? _cachedPredictions;
    private readonly int _outputDim;

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
        throw new InvalidOperationException(
            "Self teacher in cached mode does not support direct input. Use GetCachedPrediction instead.");
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
}
