# DataLoader and Batching Utilities Design Document

## Issue Reference
GitHub Issue: #443

## Design Decisions

### Architecture Approach
**Decision**: Extend existing DataLoader infrastructure
- Modify existing `DataLoaderBase<T>` and derived classes
- Add methods to existing `IBatchIterable<TBatch>` interface
- Create extension methods for fluent API

### Scope
**Decision**: All phases in a single comprehensive PR

### Backward Compatibility
**Decision**: Focus on production-readiness and exceeding industry standards
- Backward compatibility is NOT a priority
- Code must be mathematically accurate and fully production-ready

### Industry Standards
**Decision**: Support BOTH PyTorch and TensorFlow paradigms
- PyTorch patterns: num_workers, pin_memory, prefetch_factor, collate_fn, Sampler interface
- TensorFlow patterns: map, filter, cache, prefetch, interleave, padded_batch

### Threading/Parallelism
**Decision**: Support BOTH async prefetching AND parallel workers

### Memory Model
**Decision**: Support BOTH in-memory AND streaming datasets

### Sampling Strategies
**Decision**: Comprehensive sampling support
- Random sampling
- Stratified sampling
- Weighted sampling
- Curriculum learning
- Importance sampling
- Active learning
- Custom sampler interface

## Implementation Progress

### Phase 1: Core Batching Infrastructure
| Task | Status | Commit |
|------|--------|--------|
| 1A: Add GetBatches() with yield return | COMPLETE | 45870c8 |
| 1B: Add CreateBatches fluent API | COMPLETE | dc8b256 |
| 1C: dropLast, shuffle, seed parameters | COMPLETE | (included in 1A) |

### Phase 2: Sampling Strategies
| Task | Status | Commit |
|------|--------|--------|
| 2A: IDataSampler interface + Random/Stratified/Weighted | COMPLETE | 9ff9833 |
| 2B: Curriculum/Importance/Active learning | COMPLETE | f5bea56 |

### Phase 3: Async and Parallel Support
| Task | Status | Commit |
|------|--------|--------|
| 3A: Async prefetching with Channel | COMPLETE | (included in 1A) |
| 3B: Parallel worker support | COMPLETE | 908b09d |

### Phase 4: Advanced Features
| Task | Status | Commit |
|------|--------|--------|
| 4A: StreamingDataLoader | COMPLETE | 32879d3 |
| 4B: TensorFlow-style operators | COMPLETE | 4cd9a11 |

### Phase 5: Optimizer Integration
| Task | Status | Commit |
|------|--------|--------|
| 5: Refactor optimizers | PENDING | - |

### Phase 6: Testing
| Task | Status | Commit |
|------|--------|--------|
| 6: Unit tests (90%+ coverage) | PENDING | - |

### Phase 7: Benchmarks
| Task | Status | Commit |
|------|--------|--------|
| 7: Performance benchmarks | PENDING | - |

## Files Modified/Created

### Interface Changes
- `src/Interfaces/IBatchIterable.cs` - Added GetBatches() and GetBatchesAsync()

### Base Class Implementations
- `src/Data/Loaders/InputOutputDataLoaderBase.cs` - Full batch iteration support
- `src/Data/Loaders/EpisodicDataLoaderBase.cs` - Meta-learning batch support
- `src/Data/Loaders/GraphDataLoaderBase.cs` - Graph batch support
- `src/Data/Loaders/RLDataLoaderBase.cs` - RL experience batch support

### Extension Methods
- `src/Extensions/DataLoaderExtensions.cs` - Fluent API builders

### Sampling Infrastructure
- `src/Interfaces/IDataSampler.cs` - Sampler interfaces
- `src/Data/Sampling/RandomSampler.cs` - Random, Sequential, Subset samplers
- `src/Data/Sampling/StratifiedSampler.cs` - Stratified and StratifiedBatch samplers
- `src/Data/Sampling/WeightedSampler.cs` - Weighted sampling with class balancing
- `src/Data/Sampling/CurriculumSampler.cs` - Curriculum and Self-paced learning
- `src/Data/Sampling/ImportanceSampler.cs` - Importance and Active learning samplers

### Parallel Loading
- `src/Data/Loaders/ParallelBatchLoader.cs` - Multi-worker batch loading with prefetch

### Streaming Data Loaders
- `src/Data/Loaders/StreamingDataLoader.cs` - Streaming, File, CSV, and MemoryMapped loaders

### Pipeline Operators
- `src/Data/Pipeline/DataPipeline.cs` - TensorFlow-style pipeline operators

## API Design

### PyTorch-Style Batch Iteration
```csharp
// Direct method call
foreach (var (x, y) in dataLoader.GetBatches(batchSize: 32, shuffle: true))
{
    model.TrainOnBatch(x, y);
}

// Fluent API
foreach (var batch in dataLoader.CreateBatches(32).Shuffled().DropLast())
{
    model.TrainOnBatch(batch);
}

// Async with prefetching
await foreach (var batch in dataLoader.GetBatchesAsync(prefetchCount: 2))
{
    await model.TrainOnBatchAsync(batch);
}
```

### TensorFlow-Style Pipeline (Phase 4)
```csharp
// Planned API
var pipeline = dataLoader
    .Map(Augment)
    .Filter(IsValid)
    .Cache()
    .Shuffle(bufferSize: 1000)
    .Batch(32)
    .Prefetch(2);
```

## Success Criteria
- [ ] All phases complete
- [ ] Build succeeds on net8.0 and net471
- [ ] 90%+ test coverage
- [ ] Performance matches or exceeds manual batching
- [ ] All optimizers refactored to use new API
