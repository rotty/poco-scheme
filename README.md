# A toy Scheme interpreter

This is an implementation of a (currently miniscule) subset of
R7RS-small. See [the examples directory](./scheme-examples) for a
glimpse of what already works.

## Features already present

- Proper tail calls.
- Simple transformation from raw S-expressions to an AST; the
  interpreter works on the latter.
- Garbage collection employing the `gc` crate.
- `define`, `lambda`, `if`.
- Fixnums, and some procedures working on numbers.
- A very crude version of `display` and `newline`, both without port arguments.

## Next steps

- Basic cons cell support.
- Implement more syntax in `lexpr`:
  - Quote syntactic shorthand (i.e. `'`).
- Add a transformation of the AST to some kind of bytecode.
