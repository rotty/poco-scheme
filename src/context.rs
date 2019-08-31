#![allow(dead_code)]

use std::{fmt, io, rc::Rc};

use gc::{Finalize, Gc, GcCell};
use log::debug;

use crate::{
    ast::{self, Application, Ast, Conditional, EnvRef, Lambda, Params, SyntaxError},
    prim,
    value::{Core, Exception, Procedure, Value},
};

macro_rules! syntax_error {
    ($fmt:literal) => { SyntaxError::Message($fmt.into()) };
    ($fmt:literal, $($args:expr),*) => { SyntaxError::Message(format!($fmt, $($args),*)) }
}

#[derive(Debug, Default)]
pub struct Context(GcCtx);

pub type GcCtx = Gc<GcCell<Ctx>>;

#[derive(Debug, Default)]
pub struct Ctx {
    env: GcEnv,
    stack: Vec<Value>,
    specific: Option<ContextSpecific>,
}

impl Ctx {
    fn make_child(&self, lambda: Gc<GcCell<Lambda>>, env: GcEnv) -> Self {
        Ctx {
            env: env.clone(),
            stack: Default::default(),
            specific: Some(ContextSpecific { lambda }),
        }
    }

    fn get_lambda(&self) -> Option<Gc<GcCell<Lambda>>> {
        self.specific.as_ref().map(|s| s.lambda.clone())
    }

    fn has_lambda_env(&self) -> bool {
        self.env.borrow().has_lambda()
    }
}

impl gc::Finalize for Ctx {
    fn finalize(&self) {}
}

macro_rules! impl_context_trace_body {
    ($this:ident, $method:ident) => {{
        $this.env.$method();
        for value in &$this.stack {
            value.$method();
        }
        if let Some(specific) = &$this.specific {
            specific.lambda.$method();
        }
    }};
}

unsafe impl gc::Trace for Ctx {
    unsafe fn trace(&self) {
        impl_context_trace_body!(self, trace)
    }
    unsafe fn root(&self) {
        impl_context_trace_body!(self, root)
    }
    unsafe fn unroot(&self) {
        impl_context_trace_body!(self, unroot)
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_context_trace_body!(self, finalize)
    }
}

#[derive(Debug)]
struct ContextSpecific {
    lambda: Gc<GcCell<Lambda>>,
}

const MAX_ANALYZE_DEPTH: usize = 128;

fn eval_step(ctx: &GcCtx, ast: &Ast) -> Thunk {
    debug!("eval-step: {:?} in {:?}", &ast, &ctx.borrow().env);
    use std::ops::{Deref, DerefMut};
    match &*ast {
        Ast::Datum(value) => Thunk::Resolved(value.into()),
        Ast::Ref(env_ref) => {
            let res = match env_ref.cell.value.borrow().deref() {
                Value::Undefined => make_error!("undefined value `{}`", env_ref.ident),
                Value::Lambda(lambda) => {
                    // FIXME: this is ugly
                    {
                        let lambda = lambda.borrow();
                        debug!("resolving {:?} in params of {:?}", env_ref.ident, lambda);
                        if let Some(idx) = lambda.param_idx(&env_ref.ident) {
                            //dbg!((&lambda, &env_ref.ident, idx, &self.stack));
                            return Thunk::Resolved(param_ref(ctx, &lambda.params, idx));
                        }
                    }
                    Value::Lambda(lambda.clone())
                }
                value => value.clone(),
            };
            Thunk::Resolved(res)
        }
        Ast::Set(env_ref, expr) => {
            let value = match eval_ast(ctx, expr).into_result() {
                Err(e) => return Thunk::Resolved(Value::Exception(e)),
                Ok(v) => v,
            };
            match env_ref.cell.value.borrow_mut().deref_mut() {
                Value::Lambda(lambda) => {
                    let lambda = lambda.borrow();
                    let idx = lambda.param_idx(&env_ref.ident).unwrap();
                    param_set(ctx, lambda.params.stack_slots(), idx, value);
                }
                loc => {
                    *loc = value;
                }
            };
            Thunk::Resolved(Value::Unspecified)
        }
        Ast::Seq(exprs, last) => {
            for expr in exprs {
                if let v @ Value::Exception(_) = eval_ast(ctx, expr) {
                    return Thunk::Resolved(v);
                }
            }
            eval_step(ctx, last)
        }
        Ast::Apply(app) => {
            let (op, operands) = match eval_app(ctx, &app) {
                Ok(v) => v,
                Err(e) => return e.into(),
            };
            apply(ctx, op, operands)
        }
        Ast::Lambda(lambda) => {
            let lambda = lambda.borrow();
            let proc = Procedure {
                name: lambda.name.clone(),
                params: lambda.params.clone(),
                body: lambda.body.as_ref().unwrap().clone(),
            };
            Thunk::Resolved(Value::Procedure(Gc::new(proc)))
        }
        Ast::If(c) => {
            let cond = try_value!(eval_ast(ctx, &c.cond));
            if cond.is_true() {
                Thunk::Eval(c.consequent.clone(), ctx.clone())
            } else {
                Thunk::Eval(c.alternative.clone(), ctx.clone())
            }
        }
    }
}

fn eval_ast(ctx: &GcCtx, ast: &Ast) -> Value {
    match eval_step(ctx, ast) {
        Thunk::Resolved(v) => v,
        Thunk::Eval(ast, ctx) => {
            let mut ctx = ctx;
            let mut ast = ast;
            loop {
                match eval_step(&ctx, &ast) {
                    Thunk::Resolved(v) => break v,
                    Thunk::Eval(thunk_ast, thunk_ctx) => {
                        ast = thunk_ast;
                        ctx = thunk_ctx;
                    }
                }
            }
        }
    }
}

fn param_ref(ctx: &GcCtx, params: &Params, idx: usize) -> Value {
    let stack = &ctx.borrow().stack;
    debug!("param_ref: {} {} in {:?}", params, idx, stack);
    stack[stack.len() - params.stack_slots() + idx].clone()
}

fn param_set(ctx: &GcCtx, param_slots: usize, idx: usize, value: Value) {
    let stack = &mut ctx.borrow_mut().stack;
    let idx = stack.len() - param_slots + idx;
    stack[idx] = value;
}

fn eval_app(ctx: &GcCtx, app: &Application) -> Result<(Value, Vec<Value>), Box<Exception>> {
    let op = eval_ast(ctx, &app.op).into_result()?;
    let operands = app
        .operands
        .iter()
        .map(|operand| eval_ast(ctx, operand).into_result())
        .collect::<Result<Vec<_>, _>>()?;
    Ok((op, operands))
}

fn apply(ctx: &GcCtx, op: Value, args: Vec<Value>) -> Thunk {
    match op {
        Value::PrimOp(op) => Thunk::Resolved((op.func)(&args)),
        Value::Procedure(proc) => {
            // FIXME: double copying/allocation wrt. `values`
            let values = try_result!(proc.params.values(args));
            let mut child = make_eval_child(ctx);
            for value in values {
                child.stack.push(value);
            }
            let child = Gc::new(GcCell::new(child));
            eval_step(&child, &proc.body)
        }
        _ => Thunk::Resolved(make_error!(
            "non-applicable object in operator position";
            op
        )),
    }
}

fn make_eval_child(ctx: &GcCtx) -> Ctx {
    Ctx {
        env: ctx.borrow().env.clone(),
        stack: Default::default(),
        specific: None,
    }
}

pub fn eval(ctx: &GcCtx, expr: &lexpr::Value) -> Result<Value, EvalError> {
    let child = Gc::new(GcCell::new(make_eval_child(ctx)));
    let ast = analyze(&child, expr, 0, true)?;
    eval_ast(&child, &ast)
        .into_result()
        .map_err(EvalError::Exception)
}

fn create_env_cell(ctx: &GcCtx, env: GcEnv, ident: Rc<str>) -> GcEnvCell {
    if let Some(cell) = get_env_cell(ctx, env.clone(), &ident) {
        return cell;
    }
    env.borrow_mut().define(ident, Value::Undefined)
}

fn get_env_cell(_ctx: &GcCtx, env: GcEnv, ident: &str) -> Option<GcEnvCell> {
    // TODO: select environment based on `fv` field, resolve syntactic
    // closures; see `sexp_env_cell_loc`. This also means that we need an
    // addition "context" argument in quite a few places.
    env.borrow_mut().get_cell(ident)
}

fn make_lambda(ctx: &GcCtx, params: ast::Params) -> GcCtx {
    let idents: Vec<_> = params.idents().cloned().collect();
    let lambda = Gc::new(GcCell::new(Lambda::new(params)));
    let env = Env::extend(
        ctx.borrow().env.clone(),
        idents,
        Value::Lambda(lambda.clone()),
    );
    env.borrow_mut().lambda = Some(lambda.clone());
    Gc::new(GcCell::new(ctx.borrow().make_child(lambda, env)))
}

fn analyze(ctx: &GcCtx, expr: &lexpr::Value, depth: usize, defs: bool) -> Result<Ast, SyntaxError> {
    let depth = depth + 1;
    if depth > MAX_ANALYZE_DEPTH {
        return Err(SyntaxError::MaxAnalyzeDepthExceeded);
    }
    match expr {
        lexpr::Value::Null => Err(SyntaxError::EmptyApplication),
        lexpr::Value::Nil
        | lexpr::Value::Char(_)
        | lexpr::Value::Keyword(_)
        | lexpr::Value::Bytes(_)
        | lexpr::Value::Vector(_) => Err(SyntaxError::UnsupportedDatum(expr.clone())),
        lexpr::Value::Bool(_) | lexpr::Value::Number(_) | lexpr::Value::String(_) => {
            Ok(Ast::Datum(expr.clone()))
        }
        lexpr::Value::Symbol(ident) => analyze_var_ref(ctx, ident.clone().into()),
        lexpr::Value::Cons(pair) => {
            let maybe_cell = pair
                .car()
                .as_symbol()
                .and_then(|ident| get_env_cell(ctx, ctx.borrow().env.clone(), ident));
            if let Some(cell) = maybe_cell {
                let value = cell.value.borrow();
                match &*value {
                    Value::Core(op) => match op {
                        Core::Lambda => analyze_lambda(ctx, pair, depth),
                        Core::Begin => analyze_begin(ctx, pair, depth, defs),
                        Core::Define => analyze_define(ctx, pair, depth),
                        Core::If => analyze_if(ctx, pair, depth),
                        _ => unimplemented!(),
                    },
                    _ => Ok(analyze_app(ctx, pair, depth)?),
                }
            } else {
                Ok(analyze_app(ctx, pair, depth)?)
            }
        }
    }
}

fn analyze_app(ctx: &GcCtx, pair: &lexpr::Cons, depth: usize) -> Result<Ast, SyntaxError> {
    let op = analyze(ctx, pair.car(), depth, false)?;
    let operands = proper_list(pair.cdr())?;
    let app = Application {
        op,
        operands: operands
            .iter()
            .map(|expr| analyze(ctx, expr, depth, false))
            .collect::<Result<_, _>>()?,
    };
    Ok(Ast::Apply(Box::new(app)))
}

fn analyze_lambda(ctx: &GcCtx, pair: &lexpr::Cons, depth: usize) -> Result<Ast, SyntaxError> {
    let exprs = proper_list(pair.cdr())?;
    if exprs.len() < 2 {
        return Err(syntax_error!("`lambda` expects at least two forms"));
    }
    analyze_lambda_list(ctx, None, exprs[0], &exprs[1..], depth)
}

fn analyze_lambda_list(
    ctx: &GcCtx,
    name: Option<Rc<str>>,
    args: &lexpr::Value,
    body: &[&lexpr::Value],
    depth: usize,
) -> Result<Ast, SyntaxError> {
    let child = make_lambda(ctx, Params::new(args)?);
    let body = analyze_seq(&child, body, depth, true)?;
    let lambda = child.borrow().get_lambda().unwrap();
    let mut lambda = lambda.borrow_mut();
    let mut defs = Vec::new();
    for (exprs, def_ctx) in &lambda.defs {
        // TODO: this should return a non-option `name`
        let (name, value) = match &exprs[0] {
            lexpr::Value::Cons(cell) => {
                let name = cell.car().as_symbol().map(|s| s.to_owned().into());
                (
                    name.clone(),
                    analyze_lambda_list(
                        def_ctx,
                        name,
                        args,
                        &exprs[1..].iter().collect::<Vec<_>>(),
                        depth,
                    )?,
                )
            }
            name => (
                name.as_symbol().map(|s| s.to_owned().into()),
                analyze(def_ctx, &exprs[1], depth, false)?,
            ),
        };
        let name = name.unwrap();
        // TODO: transfer lambda name
        let env_ref = analyze_env_ref(def_ctx, name)?;
        let set_var = Ast::Set(env_ref, Gc::new(value));
        defs.push(set_var);
    }
    let body = if defs.is_empty() {
        body
    } else {
        match body {
            Ast::Seq(elts, last) => Ast::Seq(elts, last),
            _ => Ast::Seq(defs, Gc::new(body)),
        }
    };
    lambda.body = Some(Gc::new(body));
    lambda.name = name;
    let gc_lambda = child.borrow().get_lambda();
    Ok(Ast::Lambda(gc_lambda.unwrap().clone()))
}

fn analyze_begin(
    ctx: &GcCtx,
    pair: &lexpr::Cons,
    depth: usize,
    defs: bool,
) -> Result<Ast, SyntaxError> {
    let seq = proper_list(pair.cdr())?;
    analyze_seq(ctx, &seq, depth, defs)
}

fn analyze_define(ctx: &GcCtx, pair: &lexpr::Cons, depth: usize) -> Result<Ast, SyntaxError> {
    let exprs = proper_list(pair.cdr())?;
    if exprs.len() < 2 {
        return Err(syntax_error!("bad define syntax"));
    }
    let env = ctx.borrow().env.clone();
    let ident = match exprs[0] {
        lexpr::Value::Symbol(s) => s.clone().into(),
        lexpr::Value::Cons(pair) => pair
            .car()
            .as_symbol()
            .ok_or_else(|| syntax_error!("bad name in define: {}", pair.car()))?
            .clone()
            .into(),
        _ => return Err(syntax_error!("bad name in define: {}", exprs[0])),
    };
    if env.borrow().lambda.is_some() {
        let mut env = env.borrow_mut();
        env.push(
            Rc::clone(&ident),
            Value::Lambda(ctx.borrow().get_lambda().unwrap()),
        );
        let mut lambda = env.lambda.as_ref().unwrap().borrow_mut();
        lambda.locals.push(ident);
        // TODO: move arg list parsing here, improve `defs` element type.
        lambda
            .defs
            .push((exprs.into_iter().cloned().collect(), ctx.clone()));
        Ok(Ast::Datum(lexpr::Value::Nil))
    } else {
        create_env_cell(ctx, env.clone(), Rc::clone(&ident));
        let value = match exprs[0] {
            lexpr::Value::Cons(pair) => {
                analyze_lambda_list(ctx, Some(ident.clone()), pair.cdr(), &exprs[1..], depth)?
            }
            _ => {
                if exprs.len() != 2 {
                    return Err(syntax_error!("bad define syntax"));
                }
                analyze(ctx, exprs[1], depth, false)?
            }
        };
        // TODO: immutable bindings
        Ok(Ast::Set(
            EnvRef {
                ident: Rc::clone(&ident),
                cell: get_env_cell(ctx, env, &ident).unwrap(),
            },
            Gc::new(value),
        ))
    }
}

fn analyze_seq(
    ctx: &GcCtx,
    seq: &[&lexpr::Value],
    depth: usize,
    defs: bool,
) -> Result<Ast, SyntaxError> {
    if let Some((last, exprs)) = seq.split_last() {
        if exprs.is_empty() {
            analyze(ctx, last, depth, defs)
        } else {
            let mut ast_seq = Vec::new();
            let mut defs = defs;
            for expr in exprs {
                let ast = analyze(ctx, expr, depth, defs)?;
                defs = if ctx.borrow().has_lambda_env() && !ast.is_definition() {
                    false
                } else {
                    defs
                };
                ast_seq.push(ast);
            }
            Ok(Ast::Seq(ast_seq, Gc::new(analyze(ctx, last, depth, defs)?)))
        }
    } else {
        Ok(Ast::Datum(lexpr::Value::Nil))
    }
}

fn analyze_var_ref(ctx: &GcCtx, ident: Rc<str>) -> Result<Ast, SyntaxError> {
    analyze_env_ref(ctx, ident).map(Ast::Ref)
}

fn analyze_env_ref(ctx: &GcCtx, ident: Rc<str>) -> Result<EnvRef, SyntaxError> {
    let cell = create_env_cell(ctx, ctx.borrow().env.clone(), ident.clone());
    let value = cell.value.borrow();
    if let Value::Core(_) = &*value {
        Err(SyntaxError::UseOfSyntaxAsValue(ident.into()))
    } else {
        Ok(EnvRef {
            ident: cell.name.clone(),
            cell: cell.clone(),
        })
    }
}

fn analyze_if(ctx: &GcCtx, pair: &lexpr::Cons, depth: usize) -> Result<Ast, SyntaxError> {
    let args = proper_list(pair.cdr())?;
    if args.len() < 2 {
        return Err(syntax_error!("`if` expects at least two forms"));
    }
    let cond = analyze(ctx, &args[0], depth, false)?;
    let consequent = analyze(ctx, &args[1], depth, false)?.into();
    let alternative = if args.len() == 3 {
        analyze(ctx, &args[2], depth, false)?.into()
    } else if args.len() == 2 {
        Ast::Datum(lexpr::Value::Nil).into()
    } else {
        return Err(syntax_error!(
            "`if` expects at least no more than three forms"
        ));
    };
    Ok(Ast::If(Box::new(Conditional {
        cond,
        consequent,
        alternative,
    })))
}

impl Context {
    fn new(ctx: Ctx) -> Self {
        Context(Gc::new(GcCell::new(ctx)))
    }

    fn env(&self) -> GcEnv {
        self.0.borrow().env.clone()
    }

    pub fn make_eval() -> Self {
        Context::new(Ctx {
            env: Gc::new(GcCell::new(Env::make_primitive())),
            stack: Default::default(),
            specific: None,
        })
    }

    fn analyze(&self, expr: &lexpr::Value) -> Result<Ast, SyntaxError> {
        analyze(&self.0, expr, 0, false)
    }

    pub fn eval(&self, expr: &lexpr::Value) -> Result<Value, EvalError> {
        eval(&self.0, expr)
    }

    pub fn env_ref(&self, ident: &str) -> Option<Value> {
        get_env_cell(&self.0, self.0.borrow().env.clone(), ident)
            .map(|cell| cell.value.borrow().clone())
    }

    pub fn eval_iter<S, E>(&self, source: S) -> EvalIter<S::IntoIter>
    where
        S: IntoIterator<Item = Result<lexpr::Value, E>>,
    {
        EvalIter {
            ctx: self,
            source: source.into_iter(),
        }
    }
}

pub struct EvalIter<'a, S> {
    ctx: &'a Context,
    source: S,
}

impl<'a, S, E> Iterator for EvalIter<'a, S>
where
    S: Iterator<Item = Result<lexpr::Value, E>>,
    E: Into<EvalError>,
{
    type Item = Result<Value, EvalError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next().map(|item| {
            item.map_err(Into::into)
                .and_then(|expr| self.ctx.eval(&expr))
        })
    }
}

fn check_undef(name: &str, value: Value) -> Value {
    match value {
        Value::Undefined => make_error!("undefined value `{}`", name),
        _ => value,
    }
}

type GcEnv = Gc<GcCell<Env>>;

pub type GcEnvCell = Gc<EnvCell>;

#[derive(Debug)]
pub struct EnvCell {
    name: Rc<str>,
    value: GcCell<Value>,
}

impl EnvCell {
    fn new(name: Rc<str>, value: Value) -> Self {
        EnvCell {
            name,
            value: GcCell::new(value),
        }
    }
}

impl gc::Finalize for EnvCell {
    fn finalize(&self) {}
}

macro_rules! impl_env_cell_trace_body {
    ($this:ident, $method:ident) => {{
        ($this.value).$method();
    }};
}

unsafe impl gc::Trace for EnvCell {
    unsafe fn trace(&self) {
        impl_env_cell_trace_body!(self, trace)
    }
    unsafe fn root(&self) {
        impl_env_cell_trace_body!(self, root)
    }
    unsafe fn unroot(&self) {
        impl_env_cell_trace_body!(self, unroot)
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_env_cell_trace_body!(self, finalize)
    }
}

#[derive(Debug, Default)]
pub struct Env {
    bindings: Vec<GcEnvCell>,
    lambda: Option<Gc<GcCell<Lambda>>>,
    parent: Option<GcEnv>,
}

impl gc::Finalize for Env {
    fn finalize(&self) {}
}

macro_rules! impl_env_trace_body {
    ($this:ident, $method:ident) => {{
        if let Some(parent) = &$this.parent {
            parent.$method();
        }
        if let Some(lambda) = &$this.lambda {
            lambda.$method();
        }
        for value in &$this.bindings {
            value.$method()
        }
    }};
}

unsafe impl gc::Trace for Env {
    unsafe fn trace(&self) {
        impl_env_trace_body!(self, trace)
    }
    unsafe fn root(&self) {
        impl_env_trace_body!(self, root)
    }
    unsafe fn unroot(&self) {
        impl_env_trace_body!(self, unroot)
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_env_trace_body!(self, finalize)
    }
}

impl Env {
    pub fn make_null() -> Self {
        Env {
            bindings: Core::items()
                .iter()
                .map(|item| Gc::new(EnvCell::new(item.name().into(), Value::Core(*item))))
                .collect(),
            lambda: None,
            parent: None,
        }
    }

    pub fn make_primitive() -> Self {
        let mut env = Env::make_null();
        for (name, op) in prim::make_ops() {
            env.define(name.into(), op);
        }
        env
    }

    fn extend<'a>(env: GcEnv, idents: impl IntoIterator<Item = Rc<str>>, value: Value) -> GcEnv {
        Gc::new(GcCell::new(Env {
            bindings: idents
                .into_iter()
                .map(|ident| Gc::new(EnvCell::new(ident.clone(), value.clone())))
                .collect(),
            lambda: None,
            parent: Some(env),
        }))
    }

    fn get_cell(&self, name: &str) -> Option<GcEnvCell> {
        if let Some(cell) = self.get_local_cell(name) {
            return Some(cell);
        }
        if let Some(parent) = &self.parent {
            parent.borrow().get_cell(name)
        } else {
            None
        }
    }

    fn get_local_cell(&self, name: &str) -> Option<GcEnvCell> {
        self.bindings.iter().find_map(|cell| {
            if name == cell.name.as_ref() {
                Some(cell.clone())
            } else {
                None
            }
        })
    }

    fn push(&mut self, name: Rc<str>, value: Value) -> GcEnvCell {
        let cell = Gc::new(EnvCell {
            name: name,
            value: GcCell::new(value),
        });
        self.bindings.push(cell.clone());
        cell
    }

    fn has_lambda(&self) -> bool {
        if self.lambda.is_some() {
            true
        } else if let Some(parent) = &self.parent {
            parent.borrow().has_lambda()
        } else {
            false
        }
    }

    // Will panic if there is no top-level environment
    fn with_var_env<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Env) -> R,
    {
        if self.lambda.is_some() {
            let parent = self.parent.as_mut().unwrap();
            // Use recursion to get this code to borrow-check
            parent.borrow_mut().with_var_env(f)
        } else {
            f(self)
        }
    }

    pub fn define(&mut self, name: Rc<str>, value: Value) -> GcEnvCell {
        self.with_var_env(|env| {
            if let Some(cell) = env.get_local_cell(&name) {
                *cell.value.borrow_mut() = value;
                cell
            } else {
                env.push(name, value)
            }
        })
    }
}

fn proper_list(expr: &lexpr::Value) -> Result<Vec<&lexpr::Value>, SyntaxError> {
    match expr {
        lexpr::Value::Cons(cell) => match cell.to_ref_vec() {
            (args, tail) => {
                if tail != &lexpr::Value::Null {
                    Err(SyntaxError::ImproperList(
                        args.into_iter().cloned().collect(),
                        tail.clone(),
                    ))
                } else {
                    Ok(args)
                }
            }
        },
        lexpr::Value::Null => Ok(Vec::new()),
        value => Err(SyntaxError::NonList(value.clone())),
    }
}

#[derive(Debug)]
pub enum EvalError {
    Io(io::Error),
    Parse(lexpr::parse::Error),
    Syntax(SyntaxError),
    Exception(Box<Exception>),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use EvalError::*;
        match self {
            Io(e) => write!(f, "I/O error: {}", e),
            Parse(e) => write!(f, "parse error: {}", e),
            Exception(e) => write!(f, "runtime exception: {}", e),
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
            Exception(_) => None,
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

impl From<Exception> for EvalError {
    fn from(e: Exception) -> Self {
        EvalError::Exception(Box::new(e))
    }
}

impl From<Box<Exception>> for EvalError {
    fn from(e: Box<Exception>) -> Self {
        EvalError::Exception(e)
    }
}

#[derive(Debug)]
pub enum Thunk {
    Resolved(Value),
    Eval(Gc<Ast>, GcCtx),
}

impl From<Value> for Thunk {
    fn from(v: Value) -> Self {
        Thunk::Resolved(v)
    }
}

impl From<Box<Exception>> for Thunk {
    fn from(e: Box<Exception>) -> Self {
        Thunk::Resolved(Value::Exception(e))
    }
}

#[cfg(test)]
mod tests;
