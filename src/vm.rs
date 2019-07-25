use std::rc::Rc;

use gc::{Gc, GcCell};

use crate::{
    ast::{Ast, EnvStack, TailPosition::*},
    evaluator::{eval as eval_ast, Env, EvalError},
    prim,
    value::{PrimOp, Value},
};

macro_rules! prim_op {
    ($name:tt, $func:expr) => {{
        static OP: PrimOp = PrimOp {
            name: $name,
            func: $func,
        };
        ($name, Value::PrimOp(&OP))
    }};
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

#[derive(Debug)]
pub struct Vm {
    env: Gc<GcCell<Env>>,
    stack: EnvStack,
}

impl Default for Vm {
    fn default() -> Self {
        Vm::new()
    }
}

impl Vm {
    pub fn new() -> Self {
        let (idents, values): (Vec<_>, _) = initial_env().iter().cloned().unzip();
        Vm {
            env: Gc::new(GcCell::new(Env::initial(values))),
            stack: EnvStack::initial(idents),
        }
    }

    pub fn eval(&mut self, expr: &lexpr::Value) -> Result<Value, EvalError> {
        if let Some(ast) = Ast::definition(&expr, &mut self.stack, NonTail)? {
            Ok(eval_ast(Rc::new(ast), self.env.clone())?)
        } else {
            Ok(Value::Unspecified)
        }
    }

    pub fn process<S, E>(&mut self, source: S) -> Processor<S::IntoIter>
    where
        S: IntoIterator<Item = Result<lexpr::Value, E>>,
    {
        Processor {
            vm: self,
            source: source.into_iter(),
        }
    }
}

pub struct Processor<'a, S> {
    vm: &'a mut Vm,
    source: S,
}

impl<'a, S, E> Iterator for Processor<'a, S>
where
    S: Iterator<Item = Result<lexpr::Value, E>>,
    E: Into<EvalError>,
{
    type Item = Result<Value, EvalError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(expr) = self.source.next() {
            let res = expr
                .map_err(Into::into)
                .and_then(|expr| Ok(Ast::definition(&expr, &mut self.vm.stack, NonTail)?));
            if let Some(ast) = res.transpose() {
                return Some(
                    self.vm
                        .stack
                        .resolve_rec(self.vm.env.clone())
                        .map_err(Into::into)
                        .and_then(|_| {
                            ast.and_then(|ast| Ok(eval_ast(Rc::new(ast), self.vm.env.clone())?))
                        }),
                );
            }
        }
        None
    }
}
