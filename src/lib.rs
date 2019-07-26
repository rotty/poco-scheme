#[macro_export]
macro_rules! make_error {
    ($fmt:literal) => { Value::String(Box::new($fmt.into())) };
    ($fmt:literal, $($args:expr),*) => { $crate::Value::String(format!($fmt, $($args),*).into()) }
}

/// Operations produce either a success or an error value.
type OpResult = Result<Value, Value>;

mod ast;
mod prim;
mod value;
mod vm;

pub use value::{PrimOp, Value};
pub use vm::{EvalError, Vm};
