export function hasExecutedCheck(report) {
  const walk = suite => (suite.specs ?? []).some(spec => !/\.setup\.ts$/.test(spec.file ?? '')
    && (spec.tests ?? []).some(test => !test.projectName?.endsWith('.setup') && (test.results ?? []).some(result => result.status === 'passed')))
    || (suite.suites ?? []).some(walk);
  return (report.suites ?? []).some(walk);
}
