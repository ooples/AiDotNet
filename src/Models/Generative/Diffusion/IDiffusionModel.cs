using AiDotNet.Interfaces;

namespace AiDotNet.Models.Generative.Diffusion;

/// <summary>
/// Interface for diffusion-based generative models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Diffusion models are a type of generative AI that learns to create
/// data (like images) by learning to reverse a gradual noising process.
///
/// How diffusion works:
/// 1. Forward process: Gradually add noise to data until it becomes pure noise
/// 2. Reverse process: Learn to gradually remove noise, starting from pure noise
/// 3. Generation: Start with random noise, apply the learned reverse process to create new data
///
/// This interface extends IFullModel to provide a consistent API for diffusion models
/// while inheriting all the standard model capabilities (training, saving, loading, etc.).
/// </para>
/// </remarks>
public interface IDiffusionModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
}

