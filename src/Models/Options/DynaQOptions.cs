using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Validation;

namespace AiDotNet.Models.Options;

public class DynaQOptions<T> : ReinforcementLearningOptions<T>
{
    public DynaQOptions()
    {
    }

    public DynaQOptions(DynaQOptions<T> other)
    {
        Guard.NotNull(other);

        StateSize = other.StateSize;
        ActionSize = other.ActionSize;
        PlanningSteps = other.PlanningSteps;
    }

    public int StateSize { get; init; } = 16;
    public int ActionSize { get; init; } = 4;
    public int PlanningSteps { get; init; } = 50;
}
