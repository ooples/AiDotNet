using AiDotNet.Models.Options;

namespace AiDotNet.Interfaces;

/// <summary>
/// Implemented by models with SPECIALIZED internal update paths (e.g.
/// <see cref="NeuralRadianceFields.Models.GaussianSplatting{T}"/>'s per-attribute LR schedule
/// for position / scale / opacity / spherical harmonics, or <see cref="Diffusion.DDPMModel{T}"/>'s
/// noise-prediction schedule) that need to consume the caller's <c>ConfigureOptimizer</c>
/// settings but can't be driven through the standard
/// <see cref="IGradientComputable{T, TInput, TOutput}"/> + Adam.Step path.
/// </summary>
/// <remarks>
/// <para>
/// When a model implements this interface, <c>AiModelBuilder</c> invokes
/// <see cref="ApplyOptimizerHyperparameters"/> immediately before the optimizer's first
/// <c>Optimize</c> call, passing the finalized <see cref="OptimizationAlgorithmOptions{T,TInput,TOutput}"/>
/// the caller configured. The model reads whatever knobs it needs
/// (<c>InitialLearningRate</c>, <c>MaxGradientNorm</c>, and — via downcast to
/// <see cref="AdamOptimizerOptions{T,TInput,TOutput}"/> — the Adam-family betas / epsilon /
/// weight decay) and stores them in its internal training state.
/// </para>
/// <para>
/// Without this hook, models that store parameters outside
/// <see cref="NeuralNetworks.NeuralNetworkBase{T}.Layers"/> (GaussianSplatting's
/// <c>_gaussians</c> list, DDPM's UNet checkpoint) silently ignore every
/// <c>AdamOptimizerOptions</c> knob except <c>MaxIterations</c> — the facade's Adam step
/// runs but finds no chunks to update because it walks the standard <c>Layers</c> path
/// that specialized models bypass. See #1833 for the full diagnostic.
/// </para>
/// <para>
/// <b>Beyond industry standard.</b> Reference GS implementations require users to
/// manually construct parameter groups: <c>torch.optim.Adam([{params: positions, lr: 1e-4},
/// {params: scales, lr: 5e-3}, ...])</c>. Users must know each attribute's magic LR from
/// the paper. AiDotNet with this hook: caller passes ONE base LR via
/// <c>AdamOptimizerOptions.InitialLearningRate</c>; the specialized model's
/// <c>ApplyOptimizerHyperparameters</c> derives the per-attribute schedule automatically
/// from that base. Industry defaults live inside the model (users don't need to know GS's
/// paper constants), but ARE fully customizable through the same options object.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public interface IHyperparameterAware<T, TInput, TOutput>
{
    /// <summary>
    /// Called once by <c>AiModelBuilder.BuildAsync</c> before the first optimizer step,
    /// with the finalized options the caller passed to <c>ConfigureOptimizer</c>. The
    /// model reads whatever hyperparameters it needs and stores them for use during its
    /// internal training path.
    /// </summary>
    /// <param name="options">
    /// The optimizer options the facade user configured. Concrete type
    /// (<see cref="AdamOptimizerOptions{T,TInput,TOutput}"/> and friends) can be
    /// recovered via <c>is</c> / <c>as</c> for Adam-family-specific knobs.
    /// </param>
    void ApplyOptimizerHyperparameters(OptimizationAlgorithmOptions<T, TInput, TOutput> options);
}
