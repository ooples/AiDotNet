namespace AiDotNet.Enums;

/// <summary>
/// Update rule used to train the FedGen server generator.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After each training step the generator's weights move against the gradient; this
/// chooses how far and in which way they move.</para>
/// </remarks>
public enum FedGenGeneratorOptimizer
{
    /// <summary>
    /// Adam with L2 weight decay added to the gradient (torch.optim.Adam), as in the official FedGen code.
    /// </summary>
    Adam,

    /// <summary>
    /// Plain stochastic gradient descent with L2 weight decay added to the gradient.
    /// </summary>
    Sgd,
}
