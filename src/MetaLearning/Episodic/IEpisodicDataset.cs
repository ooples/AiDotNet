namespace AiDotNet.MetaLearning.Episodic;

using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

public interface IEpisodicDataset<T>
{
    // Returns an enumerable of episodes; implementations may randomize based on seed.
    IEnumerable<Episode<T>> GetEpisodes(int ways, int shots, int queriesPerClass, int episodeCount, int? seed = null);
}

