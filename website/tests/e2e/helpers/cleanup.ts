export async function withCleanup<T>(work: () => Promise<T>, cleanup: () => Promise<void>): Promise<T> {
  let result: T | undefined;
  const failures: unknown[] = [];
  try { result = await work(); } catch (error) { failures.push(error); }
  try { await cleanup(); } catch (error) { failures.push(error); }
  if (failures.length === 1) throw failures[0];
  if (failures.length) throw new AggregateError(failures, 'Verification and cleanup both failed');
  return result as T;
}
