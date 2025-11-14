using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class ThompsonSamplingOptions<T> : ReinforcementLearningOptions<T>
{
    public int NumArms { get; init; }
}
