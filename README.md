# A toy Scheme interpreter

This is the core of a Scheme implementation, currently in its very
infancy.

## Features already present

- Proper tail calls
- Simple transformation from raw S-expressions to an AST; the
  interpreter works on the latter.
- Garbage collection employing the `gc` crate

## Next steps

- Implement more syntax in `lexpr`:
  - Comments
  - Quote syntactic shorthand (i.e. `'`).
- Implement `begin`
- The above all are needed (or useful) to extend the Scheme tests,
  which should give the codebase pretty good test coverage.
