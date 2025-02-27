namespace AiDotNet.Interfaces;

public interface IOptimizableModel<T>
{
    T Evaluate(Vector<T> input);
    IOptimizableModel<T> Mutate(double mutationRate);
    IOptimizableModel<T> Crossover(IOptimizableModel<T> other, double crossoverRate);
    IOptimizableModel<T> Clone();
}