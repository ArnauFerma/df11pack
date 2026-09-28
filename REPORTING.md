# Reporting a problem

These repositories are tested on specific models and GPUs, listed in each README
("Tested on"). If something does not work, or a number does not match what you
measure, please tell us: a report of a failure is as useful as a new result.

Open an issue on the repository's GitHub page. Reports of any size are welcome; the
more of the items below a report has, the faster we can act on it.

## What helps

1. **What you ran**: the exact command or the smallest code that shows the problem.
2. **What you expected, and what happened**: the full error message, or the numbers
   you got and the ones you expected (with where the expected ones come from: a
   README table, a paper, another tool).
3. **Where**: GPU model and memory, driver version, OS, and the versions of Python,
   torch, transformers and the package (`pip freeze` is enough). For a model: its
   Hugging Face name and revision, and how it was compressed (tool, version, options).
4. **Whether it repeats**: every run, sometimes, or once.
5. **The files, if you can share them**: logs, outputs, or a small input that shows it.
   Do not share anything private.

## What happens next

- We try to reproduce it first. If we can, we say so on the issue; if we cannot, we
  ask for what is missing.
- A confirmed problem gets a fix and a test that fails without the fix. It is recorded
  in the repository (CHANGELOG or a dated Corrections section), including when a
  published number turns out wrong, with the old and the new value.
- If it is not a problem in the code (a setup outside what is supported, for example),
  we say why, and add it to the README's scope if others may hit it.

## Security

For a problem that could harm users if made public (for example, a way to make a
loader execute code from a file), do not open a public issue: contact the author
through the address on their GitHub profile.
