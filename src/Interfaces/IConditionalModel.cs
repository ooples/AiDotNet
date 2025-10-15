using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines the interface for models that can make predictions based on additional conditioning information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A conditional model is a type of machine learning model that makes predictions not just based on input data,
    /// but also considering additional conditioning information. This allows the model to generate different outputs
    /// for the same input depending on the provided conditions. Conditional models are fundamental to many advanced
    /// AI applications including text-to-image generation, style transfer, and controllable generation.
    /// </para>
    /// <para><b>For Beginners:</b> Think of a conditional model as an artist who can draw in different styles.
    /// 
    /// Imagine you ask an artist to draw a cat:
    /// - Without conditions: They draw a cat in their default style
    /// - With condition "cartoon style": They draw a cartoon cat
    /// - With condition "realistic style": They draw a photorealistic cat
    /// - With condition "Van Gogh style": They draw a cat like Van Gogh would
    /// 
    /// The same concept applies to AI models:
    /// - Input: The base data (like noise for image generation)
    /// - Conditioning: Additional information (like text description, style, or class label)
    /// - Output: Results that follow both the input patterns and the conditioning
    /// 
    /// Common applications:
    /// - Text-to-image: Condition on text to generate matching images
    /// - Image editing: Condition on editing instructions
    /// - Voice synthesis: Condition on speaker identity
    /// - Music generation: Condition on genre or mood
    /// </para>
    /// </remarks>
    public interface IConditionalModel
    {
        /// <summary>
        /// Makes a prediction based on input data, timestep, and conditioning information.
        /// </summary>
        /// <param name="input">The primary input data to process.</param>
        /// <param name="timestep">The timestep or progression indicator for the model.</param>
        /// <param name="conditioning">Additional conditioning information to guide the prediction.</param>
        /// <returns>The model's prediction influenced by the conditioning.</returns>
        /// <remarks>
        /// <para>
        /// This method extends standard prediction by incorporating conditioning information that guides
        /// or modifies the model's output. The timestep parameter is commonly used in diffusion models
        /// to indicate the current step in the denoising process, while conditioning provides semantic
        /// guidance such as text descriptions, class labels, or other control signals.
        /// </para>
        /// <para><b>For Beginners:</b> This method is like giving instructions along with your request.
        /// 
        /// Breaking down the parameters:
        /// - Input: The main data (like a noisy image being cleaned up)
        /// - Timestep: How far along we are in the process (like "step 50 out of 100")
        /// - Conditioning: The instructions (like "make it look like a sunset")
        /// 
        /// For example, in text-to-image generation:
        /// - Input: Random noise that will become an image
        /// - Timestep: Which step of the denoising process we're at
        /// - Conditioning: Text like "a red car on a mountain road"
        /// - Output: Noise predictions that will eventually create the described image
        /// 
        /// The conditioning ensures the model doesn't just create any image, but specifically
        /// one that matches your description. This is how AI can create images from text prompts.
        /// </para>
        /// </remarks>
        Tensor<double> PredictConditional(Tensor<double> input, Tensor<double> timestep, Tensor<double> conditioning);
    }
}