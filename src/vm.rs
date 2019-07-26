use std::{fmt, io, rc::Rc};

use gc::{Finalize, Gc, GcCell};
use log::debug;

use crate::{
    ast::{Ast, EnvIndex, EnvStack, Lambda, SyntaxError, TailPosition::*},
    prim,
    value::{Closure, PrimOp, Value},
    OpResult,
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
        prim_op!("modulo", prim::modulo),
        prim_op!("sqrt", prim::sqrt),
        prim_op!("display", prim::display),
        prim_op!("newline", prim::newline),
    ]
}

#[derive(Debug)]
pub struct Vm {
    env: VmEnv,
    stack: EnvStack,
}

impl Default for Vm {
    fn default() -> Self {
        Vm::new()
    }
}

#[derive(Debug)]
enum VmEnv {
    Reified(Gc<GcCell<Env>>),
    Open {
        parent: Gc<GcCell<Env>>,
        locals: Vec<Value>,
    },
}

impl VmEnv {
    fn new(lambda: &Lambda, args: Vec<Value>, parent: Gc<GcCell<Env>>) -> Result<Self, Value> {
        let locals = lambda.alloc_locals(args)?;
        Ok(VmEnv::Open { parent, locals })
    }
}

impl Vm {
    pub fn new() -> Self {
        let (idents, values): (Vec<_>, _) = initial_env().iter().cloned().unzip();
        Vm {
            env: VmEnv::Reified(Gc::new(GcCell::new(Env::initial(values)))),
            stack: EnvStack::initial(idents),
        }
    }

    pub fn eval(&mut self, expr: &lexpr::Value) -> Result<Value, EvalError> {
        if let Some(ast) = Ast::definition(&expr, &mut self.stack, NonTail)? {
            Ok(self.eval_ast(Rc::new(ast))?)
        } else {
            let bodies = self.stack.reap_rec_bodies()?;
            let pos = self.local_alloc(bodies.len());
            self.resolve_bodies(&bodies, pos)?;
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

    fn resolve_bodies(&mut self, bodies: &[Rc<Ast>], offset: usize) -> Result<(), Value> {
        for (i, body) in bodies.iter().enumerate() {
            let value = self.eval_ast(Rc::clone(body))?;
            debug!("resolved body [{}] {:?} -> {}", i, body, value);
            self.local_set(offset + i, value);
        }
        Ok(())
    }

    fn local_alloc(&mut self, n: usize) -> usize {
        match &mut self.env {
            VmEnv::Open { locals, .. } => {
                let pos = locals.len();
                locals.resize(pos + n, Value::Unspecified);
                pos
            }
            VmEnv::Reified(env) => env.borrow_mut().local_alloc(n),
        }
    }

    fn local_set(&mut self, slot: usize, value: Value) {
        match &mut self.env {
            VmEnv::Open { locals, .. } => locals[slot] = value,
            VmEnv::Reified(env) => env.borrow_mut().local_set(slot, value),
        }
    }

    fn eval_ast(&mut self, ast: Rc<Ast>) -> OpResult {
        let mut ast = ast;
        loop {
            match self.eval_step(ast)? {
                Thunk::Resolved(v) => break Ok(v),
                Thunk::Eval(thunk_ast) => {
                    ast = thunk_ast;
                }
            }
        }
    }

    fn eval_step(&mut self, ast: Rc<Ast>) -> Result<Thunk, Value> {
        debug!("eval-step: {:?} in {:?}", &ast, &self.env);
        match &*ast {
            Ast::EnvRef(idx) => Ok(Thunk::Resolved(self.env_lookup(idx))),
            Ast::Datum(value) => Ok(Thunk::Resolved(value.into())),
            Ast::Lambda(lambda) => {
                let closure = Value::Closure(Box::new(Closure {
                    lambda: Rc::clone(lambda),
                    env: self.reify_env(),
                }));
                Ok(Thunk::Resolved(closure))
            }
            Ast::Seq(exprs) => {
                for (i, expr) in exprs.iter().enumerate() {
                    if i + 1 == exprs.len() {
                        return self.eval_step(Rc::clone(expr));
                    }
                    self.eval_ast(Rc::clone(expr))?;
                }
                unreachable!()
            }
            Ast::Bind(body) => {
                let pos = self.local_alloc(body.bound_exprs.len());
                self.resolve_bodies(&body.bound_exprs, pos)?;
                self.eval_step(Rc::clone(&body.expr))
            }
            Ast::If {
                cond,
                consequent,
                alternative,
            } => {
                let cond = self.eval_ast(Rc::clone(cond))?;
                if cond.is_true() {
                    Ok(Thunk::Eval(Rc::clone(consequent)))
                } else {
                    Ok(Thunk::Eval(Rc::clone(alternative)))
                }
            }
            Ast::Apply { op, operands } => {
                let op = self.eval_ast(Rc::clone(op))?;
                let operands = operands
                    .iter()
                    .map(|operand| self.eval_ast(Rc::clone(operand)))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Thunk::Resolved(self.apply(op, operands)?))
            }
            Ast::TailCall { op, operands } => {
                let op = self.eval_ast(Rc::clone(op))?;
                let operands = operands
                    .iter()
                    .map(|operand| self.eval_ast(Rc::clone(operand)))
                    .collect::<Result<Vec<_>, _>>()?;
                self.tail_call(op, operands)
            }
        }
    }

    fn with_env<F, R>(&mut self, env: VmEnv, f: F) -> R
    where
        F: FnOnce(&mut Vm) -> R,
    {
        let mut tmp = env;
        std::mem::swap(&mut self.env, &mut tmp);
        let res = f(self);
        std::mem::swap(&mut self.env, &mut tmp);
        res
    }

    fn tail_call(&mut self, op: Value, args: Vec<Value>) -> Result<Thunk, Value> {
        match op {
            Value::PrimOp(op) => Ok(Thunk::Resolved((op.func)(&args)?)),
            Value::Closure(boxed) => {
                let Closure { lambda, env } = boxed.as_ref();
                self.env = VmEnv::new(lambda, args, env.clone())?;
                self.resolve_bodies(&lambda.body.bound_exprs, lambda.params.env_slots())?;
                self.eval_step(Rc::clone(&lambda.body.expr))
            }
            _ => Err(make_error!(
                "non-applicable object in operator position: {}",
                op
            )),
        }
    }

    fn apply(&mut self, op: Value, args: Vec<Value>) -> Result<Value, Value> {
        match op {
            Value::PrimOp(op) => Ok((op.func)(&args)?),
            Value::Closure(boxed) => {
                let Closure { lambda, env } = boxed.as_ref();
                let new_env = VmEnv::new(lambda, args, env.clone())?;
                self.with_env(new_env, move |vm| {
                    vm.resolve_bodies(&lambda.body.bound_exprs, lambda.params.env_slots())?;
                    vm.eval_ast(Rc::clone(&lambda.body.expr))
                })
            }
            _ => Err(make_error!(
                "non-applicable object in operator position: {}",
                op
            )),
        }
    }

    fn reify_env(&mut self) -> Gc<GcCell<Env>> {
        match &mut self.env {
            VmEnv::Reified(env) => env.clone(),
            VmEnv::Open { parent, locals } => {
                let mut new_locals = Vec::new();
                std::mem::swap(locals, &mut new_locals);
                let env = Gc::new(GcCell::new(Env::new(parent.clone(), new_locals)));
                self.env = VmEnv::Reified(env.clone());
                env
            }
        }
    }

    fn env_lookup(&self, idx: &EnvIndex) -> Value {
        match &self.env {
            VmEnv::Open { locals, parent } => {
                if idx.level() == 0 {
                    locals[idx.slot()].clone()
                } else {
                    parent.borrow().lookup(idx.level() - 1, idx.slot())
                }
            }
            VmEnv::Reified(env) => env.borrow().lookup(idx.level(), idx.slot()),
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
                let bodies = match self.vm.stack.reap_rec_bodies() {
                    Err(e) => return Some(Err(e.into())),
                    Ok(bodies) => bodies,
                };
                let pos = self.vm.local_alloc(bodies.len());
                return Some(
                    self.vm
                        .resolve_bodies(&bodies, pos)
                        .map_err(Into::into)
                        .and_then(|_| ast.and_then(|ast| Ok(self.vm.eval_ast(Rc::new(ast))?))),
                );
            }
        }
        let res = self
            .vm
            .stack
            .reap_rec_bodies()
            .map_err(Into::into)
            .and_then(|bodies| {
                let pos = self.vm.local_alloc(bodies.len());
                Ok(self.vm.resolve_bodies(&bodies, pos)?)
            });
        // Force syntax check of accumulated bodies
        match res {
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

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

    fn local_alloc(&mut self, n: usize) -> usize {
        let pos = self.values.len();
        self.values.resize(pos + n, Value::Unspecified);
        pos
    }

    fn local_set(&mut self, offset: usize, value: Value) {
        self.values[offset] = value;
    }

    fn lookup(&self, level: usize, slot: usize) -> Value {
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
                .lookup(level - 1, slot)
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

#[derive(Debug)]
pub enum Thunk {
    Resolved(Value),
    Eval(Rc<Ast>),
}

#[derive(Debug)]
pub enum EvalError {
    Io(io::Error),
    Parse(lexpr::parse::Error),
    Syntax(SyntaxError),
    Runtime(Value),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use EvalError::*;
        match self {
            Io(e) => write!(f, "I/O error: {}", e),
            Parse(e) => write!(f, "parse error: {}", e),
            Runtime(e) => write!(f, "runtime error: {}", e),
            Syntax(e) => write!(f, "syntax error: {}", e),
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
            Syntax(e) => Some(e),
        }
    }
}

impl From<lexpr::parse::Error> for EvalError {
    fn from(e: lexpr::parse::Error) -> Self {
        EvalError::Parse(e)
    }
}

impl From<SyntaxError> for EvalError {
    fn from(e: SyntaxError) -> Self {
        EvalError::Syntax(e)
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
