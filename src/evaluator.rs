use std::{fmt, io, rc::Rc};

use gc::{Finalize, Gc, GcCell};

use crate::{
    ast::{Ast, EnvIndex, EnvStack},
    value::{Closure, Value},
    OpResult,
};

#[derive(Default, Clone, Debug)]
pub struct Env {
    parent: Option<Gc<GcCell<Env>>>,
    values: Vec<Value>,
}

impl Env {
    pub fn new(parent: Gc<GcCell<Env>>, values: Vec<Value>) -> Self {
        Env {
            parent: Some(parent),
            values,
        }
    }

    pub fn init_rec(&mut self, n: usize) -> usize {
        let pos = self.values.len();
        for _ in 0..n {
            // TODO: use better value here
            self.values.push(Value::Null);
        }
        pos
    }

    pub fn resolve_rec(&mut self, offset: usize, value: Value) {
        self.values[offset] = value;
    }

    pub fn new_root(bindings: &[(&str, Value)]) -> (Self, EnvStack) {
        let (idents, values): (Vec<_>, _) = bindings.iter().cloned().unzip();
        let env = Env {
            parent: None,
            values,
        };
        let stack = EnvStack::initial(idents);
        (env, stack)
    }

    pub fn lookup(&self, idx: &EnvIndex) -> Value {
        self.lookup_internal(idx.level(), idx.slot())
    }

    pub fn reset(&mut self, parent: Gc<GcCell<Env>>, values: Vec<Value>) {
        self.parent = Some(parent);
        // Not sure if cloning here is a good idea vs. re-assigning, but let's
        // see.
        self.values = values;
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
                .unwrap()
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
            tail_call(op, operands, env)
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
        Value::PrimOp(_, ref op) => Ok(Thunk::Resolved(op(&args)?)),
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

pub fn tail_call(
    op: Value,
    operands: &[Rc<Ast>],
    parent_env: Gc<GcCell<Env>>,
) -> Result<Thunk, Value> {
    match op {
        Value::PrimOp(_, ref op) => {
            let args = operands
                .iter()
                .map(|operand| eval(Rc::clone(operand), parent_env.clone()))
                .collect::<Result<Vec<Value>, _>>()?;
            Ok(Thunk::Resolved(op(&args)?))
        }
        Value::Closure(boxed) => {
            let Closure { lambda, env } = boxed.as_ref();
            let args = operands
                .iter()
                .map(|operand| eval(Rc::clone(operand), parent_env.clone()))
                .collect::<Result<Vec<Value>, _>>()?;
            let values = lambda.params.values(args)?;
            parent_env.borrow_mut().reset(Gc::clone(env), values);
            Ok(Thunk::Eval(Rc::clone(&lambda.body.expr), parent_env))
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
