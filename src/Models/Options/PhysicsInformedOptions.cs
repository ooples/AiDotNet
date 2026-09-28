namespace AiDotNet.Models.Options;

/// <summary>
/// Base configuration options for physics-informed neural network models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Physics-informed models learn from data AND from the physical laws the
/// data must obey. This is the common ancestor of their options classes.
/// </para>
/// <para>
/// It declares no properties of its own. Twelve options classes derive from it — PINNs, neural
/// operators, Hamiltonian and Lagrangian networks, the Deep Ritz method — and they share no
/// hyperparameter universally. The PDE and boundary loss weights, for instance, belong to the
/// PINN family and would be meaningless on FourierNeuralOperatorOptions. Hoisting them here would
/// put a property on nine classes that have no use for it, and requiring one would make those
/// classes throw at their own defaults.
/// </para>
/// <para>
/// What it does contribute is its parent: deriving from
/// <see cref="ModelHyperparameterOptions"/> gives every physics-informed options class
/// <c>MaxGradNorm</c> and the <c>Require</c> validation helpers.
/// </para>
/// </remarks>
public class PhysicsInformedOptions : ModelHyperparameterOptions
{
}
