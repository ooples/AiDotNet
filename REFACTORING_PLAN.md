# PredictionModelBuilder & DataLoader Refactoring Plan

## Confirmed Decisions (User Approved)

| Question | Decision |
|----------|----------|
| IRLDataLoader design | Follow same pattern as other domain interfaces, include everything RL-specific |
| Graph loaders | **Yes, fix now** - refactor to extend GraphDataLoaderBase |
| BuildAsync transition | **Delete immediately** - remove old overloads after consolidation |
| Meta-learning integration | Keep ConfigureMetaLearning, but wire data loader internally in BuildAsync |

---

## Current State

### BuildAsync Methods (3 overloads - TO BE CONSOLIDATED)
1. `BuildAsync()` - handles data loader + meta-learning paths
2. `BuildAsync(TInput x, TOutput y)` - explicit features/labels → **DELETE**
3. `BuildAsync(int episodes, bool verbose)` - RL training → **DELETE**

### Data Loader Hierarchy (Current with Issues)
```
DataLoaderBase<T>
├── InputOutputDataLoaderBase<T, TInput, TOutput>
│   └── InMemoryDataLoader<T, TInput, TOutput>
├── GraphDataLoaderBase<T>
│   └── (NO concrete implementations - they wrongly extend DataLoaderBase)
├── EpisodicDataLoaderBase<T, TInput, TOutput>
│   ├── UniformEpisodicDataLoader ✓
│   ├── StratifiedEpisodicDataLoader ✓
│   ├── BalancedEpisodicDataLoader ✓
│   └── CurriculumEpisodicDataLoader ✓
└── (RLDataLoaderBase - MISSING)

ISSUES:
- OGBDatasetLoader extends DataLoaderBase (should extend GraphDataLoaderBase)
- MolecularDatasetLoader extends DataLoaderBase (should extend GraphDataLoaderBase)
- CitationNetworkLoader extends DataLoaderBase (should extend GraphDataLoaderBase)
- No IRLDataLoader or RLDataLoaderBase exists
```

---

## Target Architecture

### Unified BuildAsync Flow
```csharp
public async Task<PredictionModelResult<T, TInput, TOutput>> BuildAsync()
{
    // Single entry point - routes based on data loader type

    if (_dataLoader is IRLDataLoader<T> rlLoader)
        return await BuildRLInternalAsync(rlLoader);

    if (_dataLoader is IEpisodicDataLoader<T, TInput, TOutput> episodicLoader)
    {
        // Wire episodic loader to meta-learner internally (facade pattern)
        return await BuildMetaLearningInternalAsync(episodicLoader);
    }

    if (_dataLoader is IGraphDataLoader<T> graphLoader)
        return await BuildGraphInternalAsync(graphLoader);

    if (_dataLoader is IInputOutputDataLoader<T, TInput, TOutput> ioLoader)
        return await BuildSupervisedInternalAsync(ioLoader);

    throw new InvalidOperationException("ConfigureDataLoader must be called first.");
}
```

### Target Data Loader Hierarchy
```
DataLoaderBase<T>
├── InputOutputDataLoaderBase<T, TInput, TOutput>
│   └── InMemoryDataLoader<T, TInput, TOutput>
│   └── CsvDataLoader<T, TInput, TOutput> (future)
├── GraphDataLoaderBase<T>
│   ├── OGBDatasetLoader<T>          ← REFACTOR
│   ├── MolecularDatasetLoader<T>    ← REFACTOR
│   └── CitationNetworkLoader<T>     ← REFACTOR
├── EpisodicDataLoaderBase<T, TInput, TOutput>
│   ├── UniformEpisodicDataLoader
│   ├── StratifiedEpisodicDataLoader
│   ├── BalancedEpisodicDataLoader
│   └── CurriculumEpisodicDataLoader
└── RLDataLoaderBase<T>              ← CREATE NEW
    └── EnvironmentDataLoader<T>     ← CREATE NEW
```

### User-Facing API (Clean Facade)

**Supervised Learning:**
```csharp
var result = await new PredictionModelBuilder<T, TInput, TOutput>()
    .ConfigureDataLoader(new InMemoryDataLoader<T>(features, labels))
    .ConfigureModel(model)
    .BuildAsync();
```

**Meta-Learning:**
```csharp
var result = await new PredictionModelBuilder<T, TInput, TOutput>()
    .ConfigureDataLoader(new UniformEpisodicDataLoader<T>(...))
    .ConfigureMetaLearning(metaLearnerConfig)  // Config only, NO data loader param
    .ConfigureModel(model)
    .BuildAsync();  // Internally wires data loader to meta-learner
```

**Reinforcement Learning:**
```csharp
var result = await new PredictionModelBuilder<T, TInput, TOutput>()
    .ConfigureDataLoader(new EnvironmentDataLoader<T>(environment, episodes, verbose))
    .ConfigureModel(rlAgent)
    .BuildAsync();
```

**Graph Neural Networks:**
```csharp
var result = await new PredictionModelBuilder<T, TInput, TOutput>()
    .ConfigureDataLoader(new CitationNetworkLoader<T>("cora"))
    .ConfigureModel(gnnModel)
    .BuildAsync();
```

---

## IRLDataLoader Interface Design

Following same pattern as IGraphDataLoader and IEpisodicDataLoader:

```csharp
public interface IRLDataLoader<T> : IDataLoader<T>, IBatchIterable<RLExperience<T>>
{
    // Environment
    IEnvironment<T> Environment { get; }

    // Episode configuration
    int Episodes { get; }
    int MaxStepsPerEpisode { get; }
    bool Verbose { get; }

    // Replay buffer settings (if using experience replay)
    int ReplayBufferCapacity { get; }
    int MinBufferSizeBeforeTraining { get; }

    // Sampling
    RLExperience<T> SampleExperience();
    IReadOnlyList<RLExperience<T>> SampleBatch(int batchSize);

    // Episode management
    void ResetEpisode();
    EpisodeResult<T> RunEpisode();
}
```

---

## Implementation Order

### Phase 1: RL Data Loader Infrastructure
1. [ ] Create `IRLDataLoader<T>` interface in `src/Interfaces/`
2. [ ] Create `RLDataLoaderBase<T>` class in `src/DataLoading/RL/`
3. [ ] Create `EnvironmentDataLoader<T>` concrete class in `src/DataLoading/RL/`
4. [ ] Create supporting structures (`RLExperience<T>`, `EpisodeResult<T>`) if needed

### Phase 2: Fix Graph Loader Hierarchy
5. [ ] Refactor `OGBDatasetLoader<T>` to extend `GraphDataLoaderBase<T>`
6. [ ] Refactor `MolecularDatasetLoader<T>` to extend `GraphDataLoaderBase<T>`
7. [ ] Refactor `CitationNetworkLoader<T>` to extend `GraphDataLoaderBase<T>`

### Phase 3: Consolidate BuildAsync
8. [ ] Extract supervised training logic to `BuildSupervisedInternalAsync()`
9. [ ] Extract meta-learning logic to `BuildMetaLearningInternalAsync()`
10. [ ] Extract RL training logic to `BuildRLInternalAsync()`
11. [ ] Extract graph training logic to `BuildGraphInternalAsync()`
12. [ ] Update main `BuildAsync()` to route based on data loader type
13. [ ] Delete `BuildAsync(TInput x, TOutput y)` overload
14. [ ] Delete `BuildAsync(int episodes, bool verbose)` overload
15. [ ] Update `ConfigureMetaLearning` to not require data loader (if needed)

### Phase 4: Verification
16. [ ] Build for net8.0 and net471
17. [ ] Run all tests
18. [ ] Update any tests that used deleted overloads
19. [ ] Commit changes

---

## Notes

- **Facade Pattern**: Hide all complexity from users. PredictionModelBuilder is the single entry point.
- **IP Protection**: Keep internal training logic private, expose only clean Configure*/Build API.
- **Backwards Compatibility**: Delete old overloads immediately (user decision), update all call sites.
- **Domain Base Classes**: Each domain (IO, Graph, Episodic, RL) has its own base class with shared code.
- **This file must be kept updated** to avoid losing context between sessions.

---

## Open Items (None - All Decisions Confirmed)

All major decisions have been confirmed by the user.
