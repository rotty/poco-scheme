use std::{fmt, io, rc::Rc};

use gc::{Finalize, Gc, GcCell};
use log::debug;

use crate::{
    ast::{Ast, EnvIndex},
    value::{Closure, Value},
    OpResult,
};

#[derive(Default, Clone, Debug)]
pub struct Env {
    parent: Option<Gc<GcCell<Env>>>,
    values: Vec<Value>,
}

impl Env {
    pub fn initial(values: Vec<Value>) -> Self {
        Env {
            parent: None,
            values,
        }
    }

    pub fn new(parent: Gc<GcCell<Env>>, values: Vec<Value>) -> Self {
        Env {
            parent: Some(parent),
            values,
        }
    }

    pub fn init_rec(&mut self, n: usize) -> usize {
        let pos = self.values.len();
        for _ in 0..n {
            self.values.push(Value::Unspecified);
        }
        pos
    }

    pub fn resolve_rec(&mut self, offset: usize, value: Value) {
        self.values[offset] = value;
    }

    pub fn lookup(&self, idx: &EnvIndex) -> Value {
        self.lookup_internal(idx.level(), idx.slot())
    }

    fn lookup_internal(&self, level: usize, slot: usize) -> Value {
        // Use recursion to get arround the borrow checker here. Should be
        // turned into an iterative solution, but should not matter too much for
        // now.
        if level == 0 {
            self.values[slot].clone()
        } else {
            self.parent
                .as_ref()
                .expect("invalid environment reference")
                .borrow()
                .lookup_internal(level - 1, slot)
        }
    }
}

impl gc::Finalize for Env {
    fn finalize(&self) {}
}

unsafe impl gc::Trace for Env {
    unsafe fn trace(&self) {
        if let Some(parent) = &self.parent {
            parent.trace();
        }
        for value in &self.values {
            value.trace()
        }
    }
    unsafe fn root(&self) {
        if let Some(parent) = &self.parent {
            parent.root();
        }
        for value in &self.values {
            value.root()
        }
    }
    unsafe fn unroot(&self) {
        if let Some(parent) = &self.parent {
            parent.unroot();
        }
        for value in &self.values {
            value.unroot()
        }
    }
    fn finalize_glue(&self) {
        self.finalize();
        if let Some(parent) = &self.parent {
            parent.finalize();
        }
        for value in &self.values {
            value.finalize()
        }
    }
}

pub fn eval(ast: Rc<Ast>, env: Gc<GcCell<Env>>) -> OpResult {
    let mut env = env;
    let mut ast = ast;
    loop {
        match eval_step(ast, env)? {
            Thunk::Resolved(v) => break Ok(v),
            Thunk::Eval(thunk_ast, thunk_env) => {
                ast = thunk_ast;
                env = thunk_env;
            }
        }
    }
}

fn eval_step(ast: Rc<Ast>, env: Gc<GcCell<Env>>) -> Result<Thunk, Value> {
    debug!("eval-step: {:?} in {:?}", &ast, &env);
    match &*ast {
        Ast::EnvRef(idx) => Ok(Thunk::Resolved(env.borrow_mut().lookup(idx))),
        Ast::Datum(value) => Ok(Thunk::Resolved(value.into())),
        Ast::Lambda(lambda) => {
            let closure = Value::Closure(Box::new(Closure {
                lambda: Rc::clone(lambda),
                env: env.clone(),
            }));
            Ok(Thunk::Resolved(closure))
        }
        Ast::Seq(exprs) => {
            for (i, expr) in exprs.iter().enumerate() {
                if i + 1 == exprs.len() {
                    return eval_step(Rc::clone(expr), env.clone());
                }
                eval(Rc::clone(expr), env.clone())?;
            }
            unreachable!()
        }
        Ast::Bind(body) => {
            let pos = env.borrow_mut().init_rec(body.bound_exprs.len());
            for (i, expr) in body.bound_exprs.iter().enumerate() {
                let value = eval(Rc::clone(expr), env.clone())?;
                env.borrow_mut().resolve_rec(pos + i, value);
            }
            eval_step(Rc::clone(&body.expr), env)
        }
        Ast::If {
            cond,
            consequent,
            alternative,
        } => {
            let cond = eval(Rc::clone(cond), env.clone())?;
            if cond.is_true() {
                Ok(Thunk::Eval(Rc::clone(consequent), env))
            } else {
                Ok(Thunk::Eval(Rc::clone(alternative), env))
            }
        }
        Ast::Apply { op, operands } => {
            let op = eval(Rc::clone(op), env.clone())?;
            let operands = operands
                .iter()
                .map(|operand| eval(Rc::clone(operand), env.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            apply(op, operands)
        }
        Ast::TailCall { op, operands } => {
            let op = eval(Rc::clone(op), env.clone())?;
            let operands = operands
                .iter()
                .map(|operand| eval(Rc::clone(operand), env.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            // TODO: this should be implemented more efficently
            apply(op, operands)
        }
    }
}

#[derive(Debug)]
pub enum Thunk {
    Resolved(Value),
    Eval(Rc<Ast>, Gc<GcCell<Env>>),
}

pub fn apply(op: Value, args: Vec<Value>) -> Result<Thunk, Value> {
    match op {
        Value::PrimOp(op) => Ok(Thunk::Resolved((op.func)(&args)?)),
        Value::Closure(boxed) => {
            let Closure { lambda, env } = boxed.as_ref();
            // TODO: This code is duplicated in `resolve_rec`
            let env = lambda.params.bind(args, env.clone())?;
            let pos = env.borrow_mut().init_rec(lambda.body.bound_exprs.len());
            for (i, expr) in lambda.body.bound_exprs.iter().enumerate() {
                let value = eval(Rc::clone(expr), env.clone())?;
                env.borrow_mut().resolve_rec(pos + i, value);
            }
            eval_step(Rc::clone(&lambda.body.expr), env.clone())
        }
        _ => Err(make_error!(
            "non-applicable object in operator position: {}",
            op
        )),
    }
}

#[derive(Debug)]
pub enum EvalError {
    Io(io::Error),
    Parse(lexpr::parse::Error),
    Runtime(Value),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use EvalError::*;
        match self {
            Io(e) => write!(f, "I/O error: {}", e),
            Parse(e) => write!(f, "parse error: {}", e),
            Runtime(e) => write!(f, "runtime error: {}", e),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use EvalError::*;
        match self {
            Io(e) => Some(e),
            Parse(e) => Some(e),
            Runtime(_) => None,
        }
    }
}

impl From<lexpr::parse::Error> for EvalError {
    fn from(e: lexpr::parse::Error) -> Self {
        EvalError::Parse(e)
    }
}

impl From<io::Error> for EvalError {
    fn from(e: io::Error) -> Self {
        EvalError::Io(e)
    }
}

impl From<Value> for EvalError {
    fn from(e: Value) -> Self {
        EvalError::Runtime(e)
    }
}
