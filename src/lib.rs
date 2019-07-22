use std::rc::Rc;

use gc::{Gc, GcCell};

#[macro_export]
macro_rules! make_error {
    ($fmt:literal) => { Value::String($fmt.into()) };
    ($fmt:literal, $($args:expr),*) => { $crate::Value::String(format!($fmt, $($args),*).into()) }
}

/// Operations produce either a success or an error value.
type OpResult = Result<Value, Value>;

mod ast;
mod evaluator;
mod prim;
mod value;

use ast::{Ast, TailPosition};
// TODO: Fix the somehwat confusing eval names
use evaluator::{eval as eval_ast, Env};

pub use evaluator::EvalError;
pub use value::Value;

use TailPosition::*;

macro_rules! prim_op {
    ($name:tt, $func:expr) => {
        ($name, Value::prim_op($name, $func))
    };
}

fn initial_env() -> Vec<(&'static str, Value)> {
    vec![
        prim_op!("+", prim::plus),
        prim_op!("-", prim::minus),
        prim_op!("*", prim::times),
        prim_op!("<", prim::lt),
        prim_op!("<=", prim::le),
        prim_op!(">", prim::gt),
        prim_op!(">=", prim::ge),
        prim_op!("=", prim::eq),
        prim_op!("display", prim::display),
        prim_op!("newline", prim::newline),
    ]
}

pub fn eval(expr: &lexpr::Value) -> Result<Value, EvalError> {
    let (env, mut stack) = Env::new_root(&initial_env());
    let env = Gc::new(GcCell::new(env));

    if let Some(ast) = Ast::definition(&expr, &mut stack, NonTail)? {
        Ok(eval_ast(Rc::new(ast), env.clone())?)
    } else {
        Ok(Value::Null) // TODO: better value
    }
}

pub fn eval_toplevel<I, F>(source: I, mut sink: F) -> Result<(), EvalError>
where
    I: Iterator<Item = Result<lexpr::Value, EvalError>>,
    F: FnMut(Result<Value, EvalError>) -> Result<(), EvalError>,
{
    let (env, mut stack) = Env::new_root(&initial_env());
    let env = Gc::new(GcCell::new(env));

    for expr in source {
        let res = expr.and_then(|expr| Ok(Ast::definition(&expr, &mut stack, NonTail)?));
        if let Some(ast) = res.transpose() {
            let res = stack
                .resolve_rec(env.clone())
                .map_err(Into::into)
                .and_then(|_| ast.and_then(|ast| Ok(eval_ast(Rc::new(ast), env.clone())?)));
            sink(res)?;
        }
    }
    Ok(())
}
