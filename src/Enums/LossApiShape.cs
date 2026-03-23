namespace AiDotNet.Enums;

/// <summary>
/// Describes the method signature shape a loss function uses for its primary calculation.
/// The test scaffold generator uses this to select the correct test base class.
/// </summary>
public enum LossApiShape
{
    /// <summary>
    /// Standard CalculateLoss(Vector, Vector) interface.
    /// Used by most loss functions (MSE, MAE, Huber, CrossEntropy, etc.).
    /// </summary>
    VectorVector,

    /// <summary>
    /// Triplet-style CalculateLoss(Matrix, Matrix, Matrix) with anchor, positive, negative.
    /// Used by TripletLoss.
    /// </summary>
    TripletMatrix,

    /// <summary>
    /// Contrastive-style Calculate(Vector, Matrix) with target logits and noise logits.
    /// Used by NoiseContrastiveEstimationLoss.
    /// </summary>
    TargetNoiseMatrix,

    /// <summary>
    /// Image-based Calculate(Matrix, Matrix) requiring a feature extractor.
    /// Used by PerceptualLoss. Cannot be auto-constructed due to required function parameter.
    /// </summary>
    ImageMatrix,

    /// <summary>
    /// Self-supervised CreateTask interface, not a standard loss calculation.
    /// Used by RotationPredictionLoss.
    /// </summary>
    SelfSupervised,

    /// <summary>
    /// Standard CalculateLoss(Vector, Vector) but predicted and actual have different lengths.
    /// Predicted = class probabilities (length = num_classes), actual = class indices (length = batch_size).
    /// Used by SparseCategoricalCrossEntropyLoss.
    /// </summary>
    SparseIndex,

    /// <summary>
    /// Standard CalculateLoss(Vector, Vector) but inputs are complex-interleaved pairs [real, imag, real, imag, ...].
    /// Used by QuantumLoss which operates on quantum state fidelity.
    /// </summary>
    ComplexInterleaved
}
