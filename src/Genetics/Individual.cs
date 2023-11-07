namespace AiDotNet.Genetics;

internal class Individual
{
    internal double[] Chromosome { get; set; }
    internal double Fitness { get; set; }
    internal Individual(double[] chromosome)
    {
        Chromosome = chromosome;
    }
}