namespace AiDotNet.Interfaces;

public interface ISelectionMethod<T>
{
    public void ApplySelection(List<IChromosome<T>> chromosomes, int size);
}