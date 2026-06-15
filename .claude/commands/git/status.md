---
description: Show git status with helpful context
allowed-tools: Bash(git:*)
model: haiku
---

Show the current git status with helpful context.

## Steps

1. Run `git status`
2. Summarize the state in a friendly format

## Output Format

```
## Git Status

Branch: [branch name]
State: [clean / has changes / has conflicts]

### Staged Changes (ready to commit)
- [files]

### Unstaged Changes (not staged)
- [files]

### Untracked Files
- [files]

### Suggested Next Steps
- [what you might want to do next]
```

If the repo is clean, just say:
```
✓ Working tree clean on branch [name]
```
