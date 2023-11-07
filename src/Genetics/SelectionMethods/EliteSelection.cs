namespace AiDotNet.Genetics.SelectionMethods;

public class EliteSelection<T, CType> : ISelectionMethod<T, CType>
{
    public void ApplySelection(List<IChromosome<T, CType>> chromosomes, int size)
    {
        // sort chromosomes
        chromosomes.Sort();

        // remove bad chromosomes
        chromosomes.RemoveRange(size, chromosomes.Count - size);
    }
}