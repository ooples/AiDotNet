using AiDotNet.Attributes;
namespace AiDotNet.FederatedLearning.Infrastructure;

/// <summary>
/// Base class for federated learning components that need numeric operations for a generic numeric type.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
public abstract class FederatedLearningComponentBase<T>
{
    protected static readonly AiDotNet.Tensors.Interfaces.INumericOperations<T> NumOps =
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
}

