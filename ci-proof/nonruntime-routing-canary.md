# Non-runtime CI routing canary

This documentation-only file exercises the permanent `ci-proof/**` workflow trigger. Its pull
request must skip build, test, parameter, and model validation. After merge, the exact-tree
certificate must prevent that validation from being repeated on `master` while required quality
checks retain their independently typed execution decision.
