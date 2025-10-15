using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for generative models that can create new data samples
    /// </summary>
    /// <remarks>
    /// <para>
    /// Generative models are AI models that can create new data samples similar to their training data.
    /// These models learn the underlying patterns and distributions in the training data and can generate
    /// new, synthetic examples that resemble the original data.
    /// </para>
    /// <para><b>For Beginners:</b> A generative model is like an artist that learns to create new artwork.
    /// 
    /// After studying many paintings:
    /// - It learns the patterns, styles, and techniques
    /// - It can then create entirely new paintings in a similar style
    /// - Each new creation is unique but follows the learned patterns
    /// 
    /// Common types of generative models include:
    /// - Diffusion Models: Learn to gradually remove noise from random data
    /// - GANs: Use two networks competing against each other
    /// - VAEs: Learn a compressed representation of data
    /// 
    /// These models are used for:
    /// - Image generation (creating new photos or artwork)
    /// - Text generation (writing new text)
    /// - Music generation (composing new melodies)
    /// - Data augmentation (creating more training examples)
    /// </para>
    /// </remarks>
    public interface IGenerativeModel
    {
        /// <summary>
        /// Generates new data samples
        /// </summary>
        /// <param name="shape">The shape of the data to generate (e.g., [batch_size, height, width, channels] for images)</param>
        /// <param name="seed">Optional random seed for reproducible generation</param>
        /// <returns>A tensor containing the generated data samples</returns>
        /// <remarks>
        /// <para>
        /// This method creates new data samples based on what the model has learned during training.
        /// The shape parameter determines the dimensions of the output, while the seed allows for
        /// reproducible results.
        /// </para>
        /// <para><b>For Beginners:</b> This method creates new data from scratch.
        /// 
        /// The shape parameter tells the model what size of data to create:
        /// - For a single 28x28 grayscale image: [1, 28, 28, 1]
        /// - For 10 color images of size 64x64: [10, 64, 64, 3]
        /// - For text generation: [batch_size, sequence_length, vocabulary_size]
        /// 
        /// The seed parameter is like a recipe number:
        /// - Using the same seed always produces the same output
        /// - Different seeds produce different outputs
        /// - If no seed is provided, outputs will be random each time
        /// 
        /// Example usage:
        /// ```csharp
        /// // Generate 5 new images
        /// var newImages = model.Generate(new[] { 5, 28, 28, 1 });
        /// 
        /// // Generate the same image every time
        /// var sameImage = model.Generate(new[] { 1, 28, 28, 1 }, seed: 42);
        /// ```
        /// </para>
        /// </remarks>
        Tensor<double> Generate(int[] shape, int? seed = null);
    }
}