namespace AiDotNet.Enums
{
    /// <summary>
    /// Defines the types of Stochastic Differential Equations (SDEs) used in score-based diffusion models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different SDE formulations offer various trade-offs in terms of training stability, sample quality,
    /// and computational efficiency. The choice of SDE type affects how noise is added during the forward
    /// process and how it's removed during generation.
    /// </para>
    /// <para><b>For Beginners:</b> Think of these as different recipes for adding and removing noise.
    /// 
    /// Each type has different characteristics:
    /// - Some preserve the overall brightness (variance) of images
    /// - Some let the noise grow without bounds
    /// - Some offer a middle ground
    /// 
    /// The choice affects:
    /// - How fast the model trains
    /// - The quality of generated samples
    /// - The stability of the training process
    /// </para>
    /// </remarks>
    public enum SDEType
    {
        /// <summary>
        /// Variance Exploding (VE) SDE - noise variance grows without bound.
        /// </summary>
        /// <remarks>
        /// <para>
        /// In VE-SDE, the noise level increases exponentially over time, allowing the variance
        /// of the data distribution to explode. This formulation is similar to NCSN models.
        /// </para>
        /// <para><b>For Beginners:</b> Like turning up the static on a TV until the image is completely lost.
        /// 
        /// Characteristics:
        /// - Noise keeps growing over time
        /// - Good for high-resolution generation
        /// - Can handle diverse data distributions
        /// - Used in models like NCSN and NCSN++
        /// </para>
        /// </remarks>
        VE,

        /// <summary>
        /// Variance Preserving (VP) SDE - maintains bounded variance throughout the process.
        /// </summary>
        /// <remarks>
        /// <para>
        /// VP-SDE maintains a bounded variance by scaling down the data while adding noise.
        /// This formulation is equivalent to DDPM (Denoising Diffusion Probabilistic Models).
        /// </para>
        /// <para><b>For Beginners:</b> Like gradually replacing a clear image with static while keeping brightness constant.
        /// 
        /// Characteristics:
        /// - Keeps overall image brightness stable
        /// - More stable training
        /// - Good general-purpose choice
        /// - Used in models like DDPM and DDIM
        /// </para>
        /// </remarks>
        VP,

        /// <summary>
        /// Sub-Variance Preserving (Sub-VP) SDE - a variant with improved numerical stability.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Sub-VP SDE is a modification of VP-SDE that offers better numerical stability
        /// and can lead to improved sample quality in certain scenarios.
        /// </para>
        /// <para><b>For Beginners:</b> A refined version of VP that's more numerically stable.
        /// 
        /// Characteristics:
        /// - Better numerical properties
        /// - Can improve sample quality
        /// - More complex to implement
        /// - Good for when VP has stability issues
        /// </para>
        /// </remarks>
        SubVP
    }
}