using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for autoencoder models used in latent diffusion
    /// </summary>
    public interface IAutoencoder
    {
        /// <summary>
        /// Encode input data to latent representation
        /// </summary>
        Tensor<double> Encode(Tensor<double> input);
        
        /// <summary>
        /// Decode latent representation back to data space
        /// </summary>
        Tensor<double> Decode(Tensor<double> latent);
    }
    
    /// <summary>
    /// Interface for text encoding models (e.g., CLIP)
    /// </summary>
    public interface ITextEncoder
    {
        /// <summary>
        /// Encode text to embedding representation
        /// </summary>
        Tensor<double> Encode(string text);
    }
    
    /// <summary>
    /// Interface for conditional models
    /// </summary>
    public interface IConditionalModel
    {
        /// <summary>
        /// Make predictions with conditioning information
        /// </summary>
        Tensor<double> PredictConditional(Tensor<double> input, Tensor<double> timestep, Tensor<double> conditioning);
    }
}