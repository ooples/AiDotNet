# Replay Buffers for Reinforcement Learning

This folder contains all the replay buffer implementations for reinforcement learning in the AiDotNet library.

## Types of Replay Buffers

1. **Standard Replay Buffers**
   - `ReplayBufferBase`: Base class for replay buffers
   - `StandardReplayBuffer`: Generic action implementation of basic experience replay

2. **Prioritized Replay Buffers**
   - `PrioritizedReplayBuffer`: Implementation of prioritized experience replay
   - `PrioritizedNStepReplayBuffer`: Combines n-step returns with prioritized experience replay

3. **N-Step Replay Buffers**
   - `NStepReplayBuffer`: Implementation of n-step experience replay

4. **Sequential Replay Buffers**
   - `SequentialReplayBuffer`: For sequential models like Decision Transformers

## Helper Classes

- `ReplayBatch`: Container for batches of experiences
- `PrioritizedReplayBatch`: Container for prioritized experience batches
- `TrajectoryBatch`: Container for sequential experience batches
- `SumTree`: Data structure for efficient sampling in prioritized experience replay

## Interface Compatibility

- Legacy interfaces use `IReplayBuffer<TState, T>` with int-based actions
- Generic action interfaces use `IReplayBuffer<TState, TAction, T>` for any action type

## Usage Example

```csharp
// Create a new replay buffer with generic action type
var buffer = new StandardReplayBuffer<Tensor<float>, int, float>(capacity: 10000);

// Add experiences
buffer.Add(state, action, reward, nextState, done);

// Sample a batch of experiences
var batch = buffer.SampleBatch(batchSize: 32);
```

## Migration Notes

Previous versions of AiDotNet had replay buffer implementations scattered across multiple folders:
- Memory/
- Utilities/ 
- ReplayBuffers/

This folder now consolidates all replay buffer implementations for easier maintenance and consistency.