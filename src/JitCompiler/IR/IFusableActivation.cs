using AiDotNet.Tensors.Engines;

namespace AiDotNet.JitCompiler.IR;

/// <summary>
/// Implemented by activation IR operations that can be fused into other operations
/// (Conv2D+Bias+Activation, GroupNorm+Activation, etc.).
/// Follows the Open/Closed Principle: new activations just implement this interface
/// instead of modifying every fusion pattern.
/// </summary>
public interface IFusableActivation
{
    /// <summary>
    /// The FusedActivationType that IEngine operations use for this activation.
    /// </summary>
    FusedActivationType FusedType { get; }
}
