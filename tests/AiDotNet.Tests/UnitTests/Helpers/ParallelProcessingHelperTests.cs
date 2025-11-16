using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class ParallelProcessingHelperTests
    {
        [Fact]
        public async Task ProcessTasksInParallel_WithEmptyList_ReturnsEmptyList()
        {
            // Arrange
            var tasks = new List<Func<int>>();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithSingleTask_ReturnsCorrectResult()
        {
            // Arrange
            var tasks = new List<Func<int>>
            {
                () => 42
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Single(result);
            Assert.Equal(42, result[0]);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithMultipleTasks_ReturnsAllResults()
        {
            // Arrange
            var tasks = new List<Func<int>>
            {
                () => 1,
                () => 2,
                () => 3,
                () => 4,
                () => 5
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(5, result.Count);
            Assert.Contains(1, result);
            Assert.Contains(2, result);
            Assert.Contains(3, result);
            Assert.Contains(4, result);
            Assert.Contains(5, result);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithCustomMaxDegree_RespectsLimit()
        {
            // Arrange
            int maxConcurrent = 0;
            int currentConcurrent = 0;
            var lockObj = new object();

            var tasks = Enumerable.Range(0, 10).Select<int, Func<int>>(_ => () =>
            {
                lock (lockObj)
                {
                    currentConcurrent++;
                    if (currentConcurrent > maxConcurrent)
                        maxConcurrent = currentConcurrent;
                }

                Thread.Sleep(50);

                lock (lockObj)
                {
                    currentConcurrent--;
                }

                return 1;
            }).ToList();

            // Act
            await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 2);

            // Assert
            Assert.True(maxConcurrent <= 2, $"Max concurrent was {maxConcurrent}, expected <= 2");
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithNullMaxDegree_UsesProcessorCount()
        {
            // Arrange
            var tasks = new List<Func<int>>
            {
                () => 1,
                () => 2,
                () => 3
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, null);

            // Assert
            Assert.Equal(3, result.Count);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithDifferentReturnTypes_WorksCorrectly()
        {
            // Arrange
            var tasks = new List<Func<string>>
            {
                () => "hello",
                () => "world",
                () => "test"
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Contains("hello", result);
            Assert.Contains("world", result);
            Assert.Contains("test", result);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithLongRunningTasks_CompletesAll()
        {
            // Arrange
            var tasks = new List<Func<int>>
            {
                () => { Thread.Sleep(100); return 1; },
                () => { Thread.Sleep(100); return 2; },
                () => { Thread.Sleep(100); return 3; }
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Contains(1, result);
            Assert.Contains(2, result);
            Assert.Contains(3, result);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithEmptyList_ReturnsEmptyList()
        {
            // Arrange
            var tasks = new List<Task<int>>();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithSingleTask_ReturnsCorrectResult()
        {
            // Arrange
            var tasks = new List<Task<int>>
            {
                Task.FromResult(42)
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Single(result);
            Assert.Equal(42, result[0]);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithMultipleTasks_ReturnsAllResults()
        {
            // Arrange
            var tasks = new List<Task<int>>
            {
                Task.FromResult(1),
                Task.FromResult(2),
                Task.FromResult(3),
                Task.FromResult(4),
                Task.FromResult(5)
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(5, result.Count);
            Assert.Equal(1, result[0]);
            Assert.Equal(2, result[1]);
            Assert.Equal(3, result[2]);
            Assert.Equal(4, result[3]);
            Assert.Equal(5, result[4]);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithCustomMaxDegree_ProcessesInBatches()
        {
            // Arrange
            var tasks = Enumerable.Range(0, 10)
                .Select(i => Task.Run(() => { Thread.Sleep(10); return i; }))
                .ToList();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 2);

            // Assert
            Assert.Equal(10, result.Count);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithNullMaxDegree_UsesProcessorCount()
        {
            // Arrange
            var tasks = new List<Task<int>>
            {
                Task.FromResult(1),
                Task.FromResult(2),
                Task.FromResult(3)
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, null);

            // Assert
            Assert.Equal(3, result.Count);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithDifferentReturnTypes_WorksCorrectly()
        {
            // Arrange
            var tasks = new List<Task<string>>
            {
                Task.FromResult("alpha"),
                Task.FromResult("beta"),
                Task.FromResult("gamma")
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("alpha", result[0]);
            Assert.Equal("beta", result[1]);
            Assert.Equal("gamma", result[2]);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithLargeNumberOfTasks_HandlesCorrectly()
        {
            // Arrange
            var tasks = Enumerable.Range(0, 100).Select<int, Func<int>>(i => () => i).ToList();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 4);

            // Assert
            Assert.Equal(100, result.Count);
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithTasksThatReturnSameValue_HandlesCorrectly()
        {
            // Arrange
            var tasks = Enumerable.Range(0, 10).Select<int, Func<int>>(_ => () => 42).ToList();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(10, result.Count);
            Assert.All(result, r => Assert.Equal(42, r));
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithCompletedTasks_ReturnsImmediately()
        {
            // Arrange
            var tasks = new List<Task<int>>
            {
                Task.FromResult(1),
                Task.FromResult(2),
                Task.FromResult(3),
                Task.FromResult(4),
                Task.FromResult(5)
            };

            // Act
            var startTime = DateTime.UtcNow;
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);
            var duration = DateTime.UtcNow - startTime;

            // Assert
            Assert.Equal(5, result.Count);
            Assert.True(duration.TotalMilliseconds < 1000, "Should complete quickly with already completed tasks");
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithMixedTaskDurations_CompletesAllCorrectly()
        {
            // Arrange
            var tasks = new List<Func<int>>
            {
                () => { Thread.Sleep(10); return 1; },
                () => { Thread.Sleep(50); return 2; },
                () => { Thread.Sleep(20); return 3; },
                () => { Thread.Sleep(5); return 4; },
                () => { Thread.Sleep(30); return 5; }
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(5, result.Count);
            var sum = result.Sum();
            Assert.Equal(15, sum); // 1+2+3+4+5 = 15
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithMaxDegreeOne_ExecutesSequentially()
        {
            // Arrange
            var executionOrder = new List<int>();
            var lockObj = new object();
            var tasks = Enumerable.Range(0, 5).Select(i => (Func<int>)(() =>
            {
                lock (lockObj)
                {
                    executionOrder.Add(i);
                }
                Thread.Sleep(10);
                return i;
            })).ToList();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 1);

            // Assert
            Assert.Equal(5, result.Count);
            Assert.Equal(5, executionOrder.Count);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_WithLargeNumberOfTasks_HandlesCorrectly()
        {
            // Arrange
            var tasks = Enumerable.Range(0, 100)
                .Select(i => Task.FromResult(i))
                .ToList();

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 10);

            // Assert
            Assert.Equal(100, result.Count);
            Assert.Equal(4950, result.Sum()); // Sum of 0 to 99
        }

        [Fact]
        public async Task ProcessTasksInParallel_WithComplexObjects_WorksCorrectly()
        {
            // Arrange
            var tasks = new List<Func<List<int>>>
            {
                () => new List<int> { 1, 2, 3 },
                () => new List<int> { 4, 5, 6 },
                () => new List<int> { 7, 8, 9 }
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal(3, result[0].Count);
            Assert.Equal(1, result[0][0]);
            Assert.Equal(6, result[1][2]);
        }

        [Fact]
        public async Task ProcessTasksInParallel_PreCreatedTasks_MaintainsOrder()
        {
            // Arrange
            var tasks = new List<Task<int>>
            {
                Task.Run(() => { Thread.Sleep(100); return 1; }),
                Task.Run(() => { Thread.Sleep(50); return 2; }),
                Task.Run(() => { Thread.Sleep(10); return 3; })
            };

            // Act
            var result = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, maxDegreeOfParallelism: 1);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal(1, result[0]);
            Assert.Equal(2, result[1]);
            Assert.Equal(3, result[2]);
        }
    }
}
