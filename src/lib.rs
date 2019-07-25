#[macro_export]
macro_rules! make_error {
    ($fmt:literal) => { Value::String(Box::new($fmt.into())) };
    ($fmt:literal, $($args:expr),*) => { $crate::Value::String(format!($fmt, $($args),*).into()) }
}

/// Operations produce either a success or an error value.
type OpResult = Result<Value, Value>;

mod ast;
mod evaluator;
mod prim;
mod value;
mod vm;

// TODO: Fix the somehwat confusing eval names
use evaluator::eval as eval_ast;

pub use evaluator::EvalError;
pub use value::{PrimOp, Value};
pub use vm::Vm;
