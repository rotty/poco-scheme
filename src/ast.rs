use std::{fmt, iter, rc::Rc};

use lexpr::sexp;
use log::debug;

use crate::{util::ShowSlice, Value};

macro_rules! syntax_error {
    ($fmt:literal) => { SyntaxError::Message($fmt.into()) };
    ($fmt:literal, $($args:expr),*) => { SyntaxError::Message(format!($fmt, $($args),*)) }
}

#[derive(Debug)]
pub enum Params {
    Any(Box<str>),
    Exact(Vec<Box<str>>),
    AtLeast(Vec<Box<str>>, Box<str>),
}

#[derive(Debug)]
struct EnvFrame {
    idents: Vec<Box<str>>,
    rec_bodies: Vec<lexpr::Value>,
}

impl EnvFrame {
    fn new(idents: Vec<Box<str>>) -> Self {
        EnvFrame {
            idents,
            rec_bodies: Vec::new(),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct EnvIndex(usize, usize);

impl EnvIndex {
    pub fn level(&self) -> usize {
        self.0
    }

    pub fn slot(&self) -> usize {
        self.1
    }
}

#[derive(Debug)]
pub struct EnvStack {
    frames: Vec<EnvFrame>,
}

impl EnvStack {
    pub fn initial<T>(idents: impl IntoIterator<Item = T>) -> Self
    where
        T: Into<Box<str>>,
    {
        EnvStack {
            frames: vec![EnvFrame {
                idents: idents.into_iter().map(Into::into).collect(),
                rec_bodies: Vec::new(),
            }],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn with_pushed<T, F>(&mut self, params: &Params, f: F) -> Result<(Vec<Ast>, T), SyntaxError>
    where
        F: FnOnce(&mut Self) -> Result<T, SyntaxError>,
    {
        match params {
            Params::Any(ident) => self.frames.push(EnvFrame::new(vec![ident.clone()])),
            Params::Exact(idents) => self.frames.push(EnvFrame::new(idents.to_vec())),
            Params::AtLeast(idents, rest) => {
                let mut idents = idents.to_vec();
                idents.push(rest.clone());
                self.frames.push(EnvFrame::new(idents));
            }
        }
        debug!("extended env-stack {:?}", self);
        match f(self) {
            Ok(v) => {
                let bodies = self.reap_rec_bodies()?;
                self.pop();
                debug!("done with extended env-stack {:?} -> {:?}", self, bodies);
                Ok((bodies, v))
            }
            Err(e) => {
                self.pop();
                Err(e)
            }
        }
    }

    fn pop(&mut self) {
        assert!(self.frames.len() > 1);
        self.frames.pop().unwrap();
    }

    pub fn lookup(&self, name: &str) -> Option<EnvIndex> {
        for (level, frame) in self.frames.iter().rev().enumerate() {
            if let Some(i) = frame.idents.iter().position(|ident| ident.as_ref() == name) {
                return Some(EnvIndex(level, i));
            }
        }
        None
    }

    pub fn bind_rec(&mut self, name: &str, body: lexpr::Value) {
        let last = self.frames.len() - 1;
        self.frames[last].idents.push(name.into());
        self.frames[last].rec_bodies.push(body);
        debug!("bound {} recursively -> {:?}", name, self);
    }

    fn last_frame_mut(&mut self) -> &mut EnvFrame {
        let last = self.frames.len() - 1;
        &mut self.frames[last]
    }

    pub fn reap_rec_bodies(&mut self) -> Result<Vec<Ast>, SyntaxError> {
        let bodies = self.last_frame_mut().rec_bodies.split_off(0);
        bodies
            .into_iter()
            .map(|b| Ok(Ast::expr(&b, self, NonTail)?))
            .collect::<Result<_, _>>()
    }
}

impl Params {
    pub fn new(v: &lexpr::Value) -> Result<Self, SyntaxError> {
        use lexpr::Value::*;
        match v {
            Null => Ok(Params::Exact(vec![])),
            Cons(cell) => match cell.to_ref_vec() {
                (params, Null) => Ok(Params::Exact(param_list(&params)?)),
                (params, rest) => Ok(Params::AtLeast(param_list(&params)?, param_rest(rest)?)),
            },
            _ => Ok(Params::Any(param_rest(v)?)),
        }
    }

    pub fn env_slots(&self) -> usize {
        match self {
            Params::Any(_) => 1,
            Params::Exact(names) => names.len(),
            Params::AtLeast(names, _) => names.len() + 1,
        }
    }

    /// Form the values for argument vector
    pub fn values(&self, args: Vec<Value>) -> Result<Vec<Value>, Value> {
        match self {
            Params::Any(_) => Ok(vec![Value::list(args)]),
            Params::Exact(names) => {
                if names.len() != args.len() {
                    Err(make_error!(
                        "parameter length mismatch; got ({}), expected ({})",
                        ShowSlice(&args),
                        ShowSlice(names)
                    ))
                } else {
                    Ok(args)
                }
            }
            Params::AtLeast(names, _) => {
                if names.len() > args.len() {
                    Err(make_error!(
                        "too few parameters; got ({}), expected ({})",
                        ShowSlice(&args),
                        ShowSlice(names)
                    ))
                } else {
                    let (named, rest) = args.split_at(names.len());
                    let values = named
                        .iter()
                        .cloned()
                        .chain(iter::once(Value::list(rest.iter().cloned())))
                        .collect();
                    Ok(values)
                }
            }
        }
    }
}

fn param_list(params: &[&lexpr::Value]) -> Result<Vec<Box<str>>, SyntaxError> {
    params
        .iter()
        .map(|p| {
            p.as_symbol()
                .ok_or(SyntaxError::ExpectedSymbol)
                .map(Into::into)
        })
        .collect()
}

fn param_rest(rest: &lexpr::Value) -> Result<Box<str>, SyntaxError> {
    rest.as_symbol()
        .ok_or(SyntaxError::ExpectedSymbol)
        .map(Into::into)
}

#[derive(Debug)]
pub enum Ast {
    Datum(lexpr::Value),
    Lambda(Rc<Lambda>),
    If(Box<Conditional>),
    Apply(Box<Application>),
    TailCall(Box<Application>),
    EnvRef(EnvIndex),
    Seq(Vec<Ast>, Rc<Ast>),
    Bind(Body),
}

#[derive(Debug)]
pub struct Application {
    pub op: Ast,
    pub operands: Vec<Ast>,
}

#[derive(Debug)]
pub struct Conditional {
    pub cond: Ast,
    pub consequent: Rc<Ast>,
    pub alternative: Rc<Ast>,
}

#[derive(Debug)]
pub struct Lambda {
    pub params: Params,
    pub body: Body,
}

#[derive(Debug)]
pub struct Body {
    pub bound_exprs: Vec<Ast>,
    pub expr: Rc<Ast>,
}

impl Body {
    fn new(bound_exprs: Vec<Ast>, expr: Ast) -> Self {
        Body {
            bound_exprs,
            expr: Rc::new(expr),
        }
    }
}

impl Lambda {
    fn new(
        params: Params,
        exprs: &[&lexpr::Value],
        stack: &mut EnvStack,
    ) -> Result<Self, SyntaxError> {
        let (bound_exprs, body_exprs) =
            stack.with_pushed(&params, |stack| -> Result<_, SyntaxError> {
                let mut body_exprs = Vec::with_capacity(exprs.len());
                let mut definitions = true;
                for (i, expr) in exprs.iter().enumerate() {
                    let tail = if i + 1 == exprs.len() { Tail } else { NonTail };
                    if definitions {
                        if let Some(ast) = Ast::definition(expr, stack, tail)? {
                            body_exprs.push(ast);
                            definitions = false;
                        }
                    } else {
                        body_exprs.push(Ast::expr(expr, stack, tail)?);
                    }
                }
                Ok(body_exprs)
            })?;
        Ok(Lambda {
            params,
            body: Body::new(bound_exprs, Ast::seq(body_exprs)),
        })
    }

    pub fn env_slots(&self) -> usize {
        self.params.env_slots() + self.body.bound_exprs.len()
    }

    pub fn alloc_locals(&self, args: Vec<Value>) -> Result<Vec<Value>, Value> {
        let mut locals = self.params.values(args)?;
        locals.resize(self.env_slots(), Value::Unspecified);
        Ok(locals)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum TailPosition {
    Tail,
    NonTail,
}

use TailPosition::*;

impl Ast {
    pub fn expr(
        expr: &lexpr::Value,
        stack: &mut EnvStack,
        tail: TailPosition,
    ) -> Result<Ast, SyntaxError> {
        debug!("forming AST for {} in {:?}", expr, stack);
        match expr {
            lexpr::Value::Null => Err(SyntaxError::EmptyApplication),
            lexpr::Value::Nil
            | lexpr::Value::Char(_)
            | lexpr::Value::Keyword(_)
            | lexpr::Value::Bytes(_)
            | lexpr::Value::Vector(_) => Err(SyntaxError::UnsupportedDatum(expr.clone())),
            lexpr::Value::Bool(_) => Ok(Ast::Datum(expr.clone())),
            lexpr::Value::Number(_) => Ok(Ast::Datum(expr.clone())),
            lexpr::Value::String(_) => Ok(Ast::Datum(expr.clone())),
            lexpr::Value::Symbol(ident) => stack
                .lookup(ident)
                .map(Ast::EnvRef)
                .ok_or_else(|| SyntaxError::UnboundIdentifier(ident.to_owned())),
            lexpr::Value::Cons(cell) => {
                let (first, rest) = cell.as_pair();
                match first.as_symbol() {
                    Some("quote") => {
                        let args = proper_list(rest)?;
                        if args.len() != 1 {
                            return Err(syntax_error!("`quote' expects a single form"));
                        }
                        Ok(Ast::Datum(args[0].clone()))
                    }
                    Some("lambda") => {
                        let args = proper_list(rest)?;
                        if args.len() < 2 {
                            return Err(syntax_error!("`lambda` expects at least two forms"));
                        }
                        Ast::lambda(args[0], &args[1..], stack)
                    }
                    Some("begin") => {
                        let mut exprs = proper_list(rest)?;
                        if let Some(last_expr) = exprs.pop() {
                            let tail_ast = Ast::expr(last_expr, stack, tail)?;
                            if exprs.is_empty() {
                                Ok(tail_ast)
                            } else {
                                let seq = exprs
                                    .iter()
                                    .map(|expr| Ok(Ast::expr(expr, stack, NonTail)?))
                                    .collect::<Result<_, _>>()?;
                                Ok(Ast::Seq(seq, Rc::new(tail_ast)))
                            }
                        } else {
                            Ok(Ast::Datum(lexpr::Value::Nil))
                        }
                    }
                    Some("define") => {
                        Err(syntax_error!("`define` not allowed in expression context"))
                    }
                    Some("if") => {
                        let args = proper_list(rest)?;
                        if args.len() < 2 {
                            return Err(syntax_error!("`if` expects at least two forms"));
                        }
                        let cond = Ast::expr(&args[0], stack, NonTail)?;
                        let consequent = Ast::expr(&args[1], stack, tail)?;
                        let alternative = if args.len() == 3 {
                            Ast::expr(&args[2], stack, tail)?
                        } else if args.len() == 2 {
                            Ast::Datum(lexpr::Value::Nil)
                        } else {
                            return Err(syntax_error!(
                                "`if` expects at least no more than three forms"
                            ));
                        };
                        Ok(Ast::conditional(cond, consequent, alternative))
                    }
                    _ => {
                        let arg_exprs = proper_list(rest)?;
                        let op = Ast::expr(first, stack, NonTail)?;
                        let operands = arg_exprs
                            .into_iter()
                            .map(|arg| Ok(Ast::expr(arg, stack, NonTail)?))
                            .collect::<Result<Vec<Ast>, _>>()?;
                        if tail == Tail {
                            Ok(Ast::tail_call(op, operands))
                        } else {
                            Ok(Ast::apply(op, operands))
                        }
                    }
                }
            }
        }
    }

    pub fn definition(
        expr: &lexpr::Value,
        stack: &mut EnvStack,
        tail: TailPosition,
    ) -> Result<Option<Ast>, SyntaxError> {
        // Check for definition, return `Ok(None)` if found
        if let lexpr::Value::Cons(cell) = expr {
            let (first, rest) = cell.as_pair();
            match first.as_symbol() {
                Some("define") => {
                    let args = proper_list(rest)?;
                    if args.len() < 2 {
                        return Err(syntax_error!("`define` expects at least two forms"));
                    }
                    match args[0] {
                        lexpr::Value::Symbol(ident) => {
                            if args.len() != 2 {
                                return Err(syntax_error!(
                                    "`define` for variable expects one value form"
                                ));
                            }
                            stack.bind_rec(ident, args[1].clone()); // TODO: clone
                            Ok(None)
                        }
                        lexpr::Value::Cons(cell) => {
                            let ident = cell.car().as_symbol().ok_or_else(|| {
                                syntax_error!("invalid use of `define': non-identifier")
                            })?;
                            let body = lexpr::Value::list(args[1..].iter().map(|e| (*e).clone()));
                            let lambda = lexpr::Value::cons(
                                sexp!(lambda),
                                lexpr::Value::cons(cell.cdr().clone(), body),
                            );
                            stack.bind_rec(ident, lambda);
                            Ok(None)
                        }
                        _ => Err(syntax_error!("invalid `define' form")),
                    }
                }
                Some("begin") => {
                    // TODO: if this contains definitions /and/ expressions,
                    // the definitions are not resolved before evaluating
                    // the expressions.
                    let exprs = proper_list(rest)?;
                    // TODO: This is similar to `Lambda::new`
                    let mut body_exprs = Vec::with_capacity(exprs.len());
                    let mut definitions = true;
                    for (i, expr) in exprs.iter().enumerate() {
                        let tail = if i + 1 == exprs.len() { tail } else { NonTail };
                        if definitions {
                            if let Some(ast) = Ast::definition(expr, stack, tail)? {
                                body_exprs.push(ast);
                                definitions = false;
                            }
                        } else {
                            body_exprs.push(Ast::expr(expr, stack, tail)?)
                        }
                    }
                    if body_exprs.is_empty() {
                        return Ok(None);
                    }
                    let bodies = stack.reap_rec_bodies()?;
                    if bodies.is_empty() {
                        Ok(Some(Ast::seq(body_exprs)))
                    } else {
                        Ok(Some(Ast::Bind(Body {
                            bound_exprs: bodies,
                            expr: Rc::new(Ast::seq(body_exprs)),
                        })))
                    }
                }
                _ => Ok(Some(Ast::expr(expr, stack, tail)?)),
            }
        } else {
            // Otherwise, it must be an expression
            Ok(Some(Ast::expr(expr, stack, tail)?))
        }
    }

    fn lambda(
        params: &lexpr::Value,
        body: &[&lexpr::Value],
        stack: &mut EnvStack,
    ) -> Result<Self, SyntaxError> {
        let params = Params::new(params)?;
        Ok(Ast::Lambda(Rc::new(Lambda::new(params, body, stack)?)))
    }

    fn tail_call(op: Ast, operands: Vec<Ast>) -> Self {
        Ast::TailCall(Box::new(Application { op, operands }))
    }

    fn apply(op: Ast, operands: Vec<Ast>) -> Self {
        Ast::Apply(Box::new(Application { op, operands }))
    }

    fn conditional(cond: Ast, consequent: Ast, alternative: Ast) -> Self {
        Ast::If(Box::new(Conditional {
            cond,
            consequent: consequent.into(),
            alternative: alternative.into(),
        }))
    }

    fn seq(mut seq: Vec<Ast>) -> Self {
        if let Some(last) = seq.pop() {
            if seq.is_empty() {
                last
            } else {
                Ast::Seq(seq, Rc::new(last))
            }
        } else {
            Ast::Datum(lexpr::Value::Nil)
        }
    }
}

#[derive(Debug)]
pub enum SyntaxError {
    EmptyApplication,
    UnsupportedDatum(lexpr::Value),
    ExpectedSymbol,
    ImproperList(Vec<lexpr::Value>, lexpr::Value),
    NonList(lexpr::Value),
    Message(String),
    UnboundIdentifier(Box<str>),
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use SyntaxError::*;
        match self {
            EmptyApplication => write!(f, "empty application"),
            UnsupportedDatum(datum) => write!(f, "unsupported datum `{}`", datum),
            ExpectedSymbol => write!(f, "expected symbol"),
            ImproperList(elts, tail) => {
                write!(f, "improper list `({} . {})'", ShowSlice(&elts), tail)
            }
            NonList(value) => write!(f, "non-list `{}'", value),
            Message(msg) => write!(f, "{}", msg),
            UnboundIdentifier(ident) => write!(f, "unbound identifier `{}`", ident),
        }
    }
}

impl std::error::Error for SyntaxError {}

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
