# An implementation of a tiny Scheme subset

This is an implementation of a (currently miniscule) subset of
R7RS-small. See [the examples directory](./scheme-examples) and the
[smoke tests](./tests/scheme/smoke.scm)for to get an idea of what
already works.

The architecture is currently a fairly basic interpreter, with syntax
analysis phase up-front, which will transform the raw S-expressions
into an AST, which is then evaluated. This code sprang from code
intended as an example for using the [`lexpr`] crate to implement a
simple S-expression calculator, and got a bit out of hand 😀.

The current architecture is insufficient to support several major
language features of Scheme, such as macros or continuations. The plan
is to gradually improve the implementation to make such features
possible, before investing in additional features based on the current
architecture.

The evolution of poco-scheme will probably be largely guided by the
[chibi-scheme] codebase. chibi-scheme is a full implementation of
R7RS-small, written in C, employing a bytecode VM, and using syntactic
closures as the basic macro mechanism.

## Features already present

- Proper tail calls.
- Garbage collection employing the `gc` crate.
- `define`, `lambda`, `if`.
- Fixnums, and some procedures working on numbers.
- A very crude version of `display` and `newline`, both without port arguments.

## Next steps

- Basic cons cell support.
- Implement more syntax in `lexpr`:
  - Quote syntactic shorthand (i.e. `'`).
- Add a transformation of the AST to some kind of bytecode.

[chibi-scheme]: http://synthcode.com/wiki/chibi-scheme
[`lexpr`]: https://crates.io/crates/lexpr
