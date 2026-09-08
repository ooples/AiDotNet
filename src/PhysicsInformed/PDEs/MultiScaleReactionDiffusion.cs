using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed.PDEs;

/// <summary>
/// A two-scale reaction-diffusion equation: the same physics at a coarse and a fine length scale, with
/// a coupling term that penalises the two disagreeing where they overlap.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the reference <see cref="IMultiScalePDE{T}"/>, and until it existed there was none:
/// <c>MultiScalePINN</c> takes one in its constructor and the library implemented the interface zero
/// times, so the type could not be constructed at all (#2105).
/// </para>
/// <para>
/// <b>For Beginners:</b> Some systems do the same thing at two very different sizes at once — heat
/// spreading through a whole engine block while also spreading within each grain of the metal. Modelling
/// only the coarse scale misses the fine detail; modelling only the fine one needs impossibly many
/// points. A multi-scale model runs both and asks them to agree.
/// </para>
/// <para>
/// The residual at each scale is the heat/diffusion form <c>∂u/∂t − k ∂²u/∂x²</c>, with a different
/// diffusivity per scale — a large one for the coarse field and a small one for the fine.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var pde = new MultiScaleReactionDiffusion&lt;double&gt;(
///     coarseDiffusivity: 1.0,
///     fineDiffusivity: 0.01);
///
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 1);
///
/// var pinn = new MultiScalePINN&lt;double&gt;(
///     architecture, pde, new IBoundaryCondition&lt;double&gt;[] { new DirichletBoundaryCondition&lt;double&gt;() });
/// </code>
/// </example>
public class MultiScaleReactionDiffusion<T> : PDESpecificationBase<T>, IMultiScalePDE<T>
{
    private readonly T _coarseDiffusivity;
    private readonly T _fineDiffusivity;
    private readonly T[] _characteristicLengths;
    private readonly T _fineScaleWeight;

    /// <summary>
    /// Initializes a two-scale reaction-diffusion equation.
    /// </summary>
    /// <param name="coarseDiffusivity">Diffusivity of the coarse field. Must be positive.</param>
    /// <param name="fineDiffusivity">Diffusivity of the fine field. Must be positive, and is normally much smaller.</param>
    /// <param name="coarseLength">Characteristic length of the coarse scale.</param>
    /// <param name="fineLength">Characteristic length of the fine scale.</param>
    /// <param name="fineScaleWeight">
    /// How much the fine scale's residual counts for. Above 1 by default: the fine field contributes
    /// less to the total loss simply by being smaller, and would otherwise be ignored by training.
    /// </param>
    public MultiScaleReactionDiffusion(
        double coarseDiffusivity = 1.0,
        double fineDiffusivity = 0.01,
        double coarseLength = 1.0,
        double fineLength = 0.1,
        double fineScaleWeight = 10.0)
    {
        _coarseDiffusivity = NumOps.FromDouble(coarseDiffusivity);
        _fineDiffusivity = NumOps.FromDouble(fineDiffusivity);
        ValidatePositive(_coarseDiffusivity, nameof(coarseDiffusivity));
        ValidatePositive(_fineDiffusivity, nameof(fineDiffusivity));

        var coarse = NumOps.FromDouble(coarseLength);
        var fine = NumOps.FromDouble(fineLength);
        ValidatePositive(coarse, nameof(coarseLength));
        ValidatePositive(fine, nameof(fineLength));

        _characteristicLengths = [coarse, fine];
        _fineScaleWeight = NumOps.FromDouble(fineScaleWeight);
    }

    /// <inheritdoc/>
    public override string Name => "Two-scale reaction-diffusion";

    /// <inheritdoc/>
    /// <remarks>Position and time.</remarks>
    public override int InputDimension => 2;

    /// <inheritdoc/>
    /// <remarks>One concentration.</remarks>
    public override int OutputDimension => 1;

    /// <inheritdoc/>
    public int NumberOfScales => 2;

    /// <inheritdoc/>
    public T[] ScaleCharacteristicLengths => (T[])_characteristicLengths.Clone();

    /// <inheritdoc/>
    public int GetScaleOutputDimension(int scaleIndex) => 1;

    /// <inheritdoc/>
    /// <remarks>
    /// The fine scale is weighted up because its residual is numerically smaller, not less important —
    /// left equal, training optimises the coarse field and lets the fine one drift.
    /// </remarks>
    public T GetScaleLossWeight(int scaleIndex) =>
        scaleIndex == 0 ? NumOps.One : _fineScaleWeight;

    /// <inheritdoc/>
    public T ComputeScaleResidual(
        int scaleIndex, T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
    {
        if (derivatives.FirstDerivatives is null || derivatives.SecondDerivatives is null)
        {
            return NumOps.Zero;
        }

        T diffusivity = scaleIndex == 0 ? _coarseDiffusivity : _fineDiffusivity;

        // ∂u/∂t − k ∂²u/∂x²
        T dudt = derivatives.FirstDerivatives[0, 1];
        T d2udx2 = derivatives.SecondDerivatives[0, 0, 0];

        return NumOps.Subtract(dudt, NumOps.Multiply(diffusivity, d2udx2));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// How far apart the two scales are where they describe the same point. Driving this to zero is
    /// what makes the pair one model rather than two independent ones.
    /// </remarks>
    public T ComputeScaleCoupling(
        int coarseIndex,
        int fineIndex,
        T[] inputs,
        T[] coarseOutputs,
        T[] fineOutputs,
        PDEDerivatives<T> coarseDerivatives,
        PDEDerivatives<T> fineDerivatives)
    {
        return NumOps.Abs(NumOps.Subtract(coarseOutputs[0], fineOutputs[0]));
    }

    /// <inheritdoc/>
    /// <remarks>Both scales together, which is what the single-scale contract asks for.</remarks>
    public override T ComputeResidual(
        Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives)
    {
        var x = inputs.ToArray();
        var u = outputs.ToArray();

        return NumOps.Add(
            ComputeScaleResidual(0, x, u, derivatives),
            ComputeScaleResidual(1, x, u, derivatives));
    }
}
