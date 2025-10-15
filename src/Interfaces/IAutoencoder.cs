using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines the interface for autoencoder models used in various machine learning tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An autoencoder is a type of neural network used to learn efficient data encodings in an unsupervised manner.
    /// The aim is to learn a representation (encoding) for a set of data, typically for dimensionality reduction
    /// or feature learning. An autoencoder consists of two main parts: an encoder that compresses the input into
    /// a latent-space representation, and a decoder that reconstructs the input from the latent representation.
    /// </para>
    /// <para><b>For Beginners:</b> Think of an autoencoder as a data compression and decompression system.
    /// 
    /// Imagine you have a large image file that you want to send over email:
    /// - The encoder is like a ZIP compression tool that makes the file smaller
    /// - The compressed file (latent representation) takes up less space
    /// - The decoder is like unzipping the file to get back the original image
    /// 
    /// In machine learning, autoencoders:
    /// - Learn to compress data into a smaller representation (encoding)
    /// - Can reconstruct the original data from this compressed form (decoding)
    /// - Are useful for removing noise, generating new data, or finding important features
    /// 
    /// Common uses include:
    /// - Image compression and denoising
    /// - Feature extraction for other machine learning tasks
    /// - Generating new data similar to the training data
    /// - Anomaly detection (unusual data won't compress/decompress well)
    /// </para>
    /// </remarks>
    public interface IAutoencoder
    {
        /// <summary>
        /// Encodes input data into a latent representation.
        /// </summary>
        /// <param name="input">The input data to encode.</param>
        /// <returns>The encoded latent representation.</returns>
        /// <remarks>
        /// <para>
        /// The encode method takes high-dimensional input data and compresses it into a lower-dimensional
        /// latent representation. This latent representation captures the most important features of the input
        /// while discarding less important information.
        /// </para>
        /// <para><b>For Beginners:</b> Encoding is like creating a summary of your data.
        /// 
        /// For example, if you have a 256x256 pixel image (65,536 numbers), the encoder might
        /// compress this into just 100 numbers that capture the essential information about the image.
        /// This is useful because:
        /// - It reduces storage requirements
        /// - It can remove noise from the data
        /// - It finds the most important patterns in your data
        /// - The compressed form can be used by other machine learning models
        /// </para>
        /// </remarks>
        Tensor<double> Encode(Tensor<double> input);

        /// <summary>
        /// Decodes a latent representation back into the original data space.
        /// </summary>
        /// <param name="latent">The latent representation to decode.</param>
        /// <returns>The reconstructed data.</returns>
        /// <remarks>
        /// <para>
        /// The decode method takes a latent representation and reconstructs data in the original input space.
        /// The quality of reconstruction depends on how well the autoencoder has learned to compress and
        /// decompress the data during training.
        /// </para>
        /// <para><b>For Beginners:</b> Decoding is like expanding a summary back into full detail.
        /// 
        /// Continuing the image example, the decoder takes those 100 numbers and tries to recreate
        /// the original 256x256 pixel image. The reconstructed image might not be perfect, but it
        /// should capture the important features of the original.
        /// 
        /// This is useful for:
        /// - Generating new data by modifying the latent representation
        /// - Denoising data (noise gets removed during compression)
        /// - Checking if data is normal (abnormal data won't reconstruct well)
        /// </para>
        /// </remarks>
        Tensor<double> Decode(Tensor<double> latent);
    }
}