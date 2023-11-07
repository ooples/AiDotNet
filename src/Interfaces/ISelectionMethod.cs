namespace AiDotNet.Interfaces;

public interface ISelectionMethod<T, CType>
{
    public void ApplySelection(List<IChromosome<T, CType>> chromosomes, int size);
}