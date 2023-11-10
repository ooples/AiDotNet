using AiDotNet.Genetics.Chromosomes;

namespace AiDotNet.Genetics;

public class GeneticsAi<T>
{
    public GeneticsAi(IChromosome<T> chromosome, GeneticAiOptions<T> geneticAiOptions)
    {
        var test = new GeneticsFacade<T>(chromosome, geneticAiOptions);

        var test2 = new StringChromosome("I want this string");
        test2.
    }
}