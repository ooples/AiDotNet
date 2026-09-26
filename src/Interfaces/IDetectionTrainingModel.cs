using AiDotNet.ComputerVision.Detection;

namespace AiDotNet.Interfaces;

/// <summary>A model that implements its detection task's assignment and classification/box loss.</summary>
/// <typeparam name="T">The detector's numeric type.</typeparam>
/// <remarks>
/// This capability is separate from raw-output tensor regression. Implementing it promises a real
/// family-specific detection objective, not a generic MSE fallback or an inferred tensor format.
/// </remarks>
public interface IDetectionTrainingModel<T>
{
    /// <summary>Runs one semantic detection training step.</summary>
    /// <param name="input">Model-ready NCHW image batch, preprocessed like the model's Predict input.</param>
    /// <param name="targets">One immutable foreground target list per image; empty lists are valid.</param>
    /// <remarks>Inputs are borrowed. Models must reject unsupported target cardinality before updating.</remarks>
    void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets);
}
