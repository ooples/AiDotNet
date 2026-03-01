using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class DynaQOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; } = 16;
    public int ActionSize { get; init; } = 4;
    public int PlanningSteps { get; init; } = 50;  // Number of planning updates per real step
}
