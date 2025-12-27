namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Defines the contract for projection heads used in self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A projection head is a small neural network that transforms
/// encoder outputs into a space optimized for the SSL loss. After pretraining, the projection
/// head is typically discarded and only the encoder is used for downstream tasks.</para>
///
/// <para><b>Why use a projection head?</b></para>
/// <list type="bullet">
/// <item>Prevents the contrastive loss from degrading encoder representations</item>
/// <item>Allows the encoder to keep more general information</item>
/// <item>Empirically shown to significantly improve downstream performance</item>
/// </list>
///
/// <para><b>Common architectures:</b></para>
/// <list type="bullet">
/// <item><b>Linear:</b> Single linear layer (simplest)</item>
/// <item><b>MLP:</b> 2-3 layer MLP with BatchNorm and ReLU (most common)</item>
/// <item><b>Symmetric:</b> Predictor network in BYOL/SimSiam</item>
/// </list>
/// </remarks>
public interface IProjectorHead<T>
{
    /// <summary>
    /// Gets the input dimension expected by this projector.
    /// </summary>
    int InputDimension { get; }

    /// <summary>
    /// Gets the output dimension produced by this projector.
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 128-2048. SimCLR uses 128, MoCo uses 128, BYOL uses 256.</para>
    /// </remarks>
    int OutputDimension { get; }

    /// <summary>
    /// Gets the hidden dimension (for MLP projectors).
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 2048-4096. Usually larger than output dimension.</para>
    /// </remarks>
    int? HiddenDimension { get; }

    /// <summary>
    /// Projects encoder output to the SSL embedding space.
    /// </summary>
    /// <param name="input">The encoder output tensor.</param>
    /// <returns>The projected embedding tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This transforms encoder features into a lower-dimensional
    /// space where the SSL loss is computed. The projection helps separate the pretraining
    /// objective from the learned representations.</para>
    /// </remarks>
    Tensor<T> Project(Tensor<T> input);

    /// <summary>
    /// Performs the backward pass through the projector.
    /// </summary>
    /// <param name="gradients">The gradients from the loss with respect to projector output.</param>
    /// <returns>The gradients with respect to projector input (for encoder backprop).</returns>
    Tensor<T> Backward(Tensor<T> gradients);

    /// <summary>
    /// Gets all trainable parameters of the projector.
    /// </summary>
    /// <returns>A vector containing all parameters.</returns>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets the parameters of the projector.
    /// </summary>
    /// <param name="parameters">The parameter vector to load.</param>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the gradients computed during the last backward pass.
    /// </summary>
    /// <returns>A vector containing gradients for all parameters.</returns>
    Vector<T> GetParameterGradients();

    /// <summary>
    /// Clears accumulated gradients.
    /// </summary>
    void ClearGradients();

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    int ParameterCount { get; }

    /// <summary>
    /// Sets training or evaluation mode.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for evaluation.</param>
    void SetTrainingMode(bool isTraining);

    /// <summary>
    /// Resets the projector state (clears any internal buffers).
    /// </summary>
    void Reset();
}
