namespace AiDotNet.Genetics.SelectionMethods;

public class EliteSelection<T> : ISelectionMethod<T>
{
    public void ApplySelection(List<IChromosome<T>> chromosomes, int size)
    {
        // sort chromosomes
        chromosomes.Sort();

        // remove bad chromosomes
        chromosomes.RemoveRange(size, chromosomes.Count - size);
    }
}