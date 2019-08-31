use std::{fmt, iter, rc::Rc};

use gc::{Finalize, Gc, GcCell, Trace};

use crate::{
    context::{GcCtx, GcEnvCell},
    util::ShowSlice,
    value::Value,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Params {
    Any(Rc<str>),
    Exact(Vec<Rc<str>>),
    AtLeast(Vec<Rc<str>>, Rc<str>),
}

impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Params::Any(name) => write!(f, "{}", name),
            Params::Exact(names) => write!(f, "({})", names.join(" ")),
            Params::AtLeast(names, rest) => write!(f, "({} . {})", names.join(" "), rest),
        }
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

    pub fn idents(&self) -> ParamIdents {
        match self {
            Params::Any(rest) => ParamIdents(&[], Some(rest)),
            Params::Exact(idents) => ParamIdents(idents, None),
            Params::AtLeast(idents, rest) => ParamIdents(idents, Some(rest)),
        }
    }

    pub fn stack_slots(&self) -> usize {
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

pub struct ParamIdents<'a>(&'a [Rc<str>], Option<&'a Rc<str>>);

impl<'a> Iterator for ParamIdents<'a> {
    type Item = &'a Rc<str>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((first, rest)) = self.0.split_first() {
            self.0 = rest;
            Some(first)
        } else {
            self.1.take()
        }
    }
}

fn param_list(params: &[&lexpr::Value]) -> Result<Vec<Rc<str>>, SyntaxError> {
    params
        .iter()
        .map(|p| {
            p.as_symbol()
                .ok_or(SyntaxError::ExpectedSymbol)
                .map(Into::into)
        })
        .collect()
}

fn param_rest(rest: &lexpr::Value) -> Result<Rc<str>, SyntaxError> {
    rest.as_symbol()
        .ok_or(SyntaxError::ExpectedSymbol)
        .map(Into::into)
}

#[derive(Debug)]
pub enum Ast {
    Datum(lexpr::Value),
    Lambda(Gc<GcCell<Lambda>>),
    If(Box<Conditional>),
    Apply(Box<Application>),
    //TailCall(Box<Application>),
    Ref(EnvRef),
    Set(EnvRef, Gc<Ast>),
    Seq(Vec<Ast>, Gc<Ast>),
}

impl Finalize for Ast {}

macro_rules! impl_ast_trace_body {
    ($this:ident, $method:ident) => {
        match $this {
            Ast::Datum(_) => {}
            Ast::Lambda(lambda) => lambda.$method(),
            Ast::If(c) => {
                c.cond.$method();
                c.consequent.$method();
                c.alternative.$method();
            }
            Ast::Apply(app) => {
                app.op.$method();
                for rand in &app.operands {
                    rand.$method();
                }
            }
            Ast::Ref(env_ref) => env_ref.cell.$method(),
            Ast::Set(env_ref, ast) => {
                env_ref.cell.$method();
                ast.$method();
            }
            Ast::Seq(elts, last) => {
                for elt in elts {
                    elt.$method();
                }
                last.$method();
            }
        }
    };
}

unsafe impl Trace for Ast {
    unsafe fn trace(&self) {
        impl_ast_trace_body!(self, trace);
    }
    unsafe fn root(&self) {
        impl_ast_trace_body!(self, root);
    }
    unsafe fn unroot(&self) {
        impl_ast_trace_body!(self, unroot);
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_ast_trace_body!(self, finalize_glue);
    }
}

pub struct EnvRef {
    pub ident: Rc<str>,
    pub cell: GcEnvCell,
}

impl fmt::Debug for EnvRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("EnvRef").field(&self.ident).finish()
    }
}

#[derive(Debug)]
pub struct Application {
    pub op: Ast,
    pub operands: Vec<Ast>,
}

#[derive(Debug)]
pub struct Conditional {
    pub cond: Ast,
    pub consequent: Gc<Ast>,
    pub alternative: Gc<Ast>,
}

#[derive(Clone)]
pub struct Lambda {
    pub name: Option<Rc<str>>,
    pub params: Params,
    pub locals: Vec<Rc<str>>,
    pub defs: Vec<(Vec<lexpr::Value>, GcCtx)>,
    pub body: Option<Gc<Ast>>,
}

impl Lambda {
    pub fn new(params: Params) -> Self {
        Lambda {
            name: None,
            params,
            locals: Default::default(),
            defs: Default::default(),
            body: None,
        }
    }
    pub fn param_idx(&self, name: &str) -> Option<usize> {
        self.params.idents().enumerate().find_map(|(i, ident)| {
            if ident.as_ref() == name {
                Some(i)
            } else {
                None
            }
        })
    }
}

impl PartialEq<Lambda> for Lambda {
    fn eq(&self, _: &Lambda) -> bool {
        false
    }
}

impl fmt::Debug for Lambda {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Lambda")
            .field("name", &self.name)
            .field("params", &self.params)
            .field("locals", &self.locals)
            .field(
                "defs",
                &self.defs.iter().map(|(def, _ctx)| def).collect::<Vec<_>>(),
            )
            .field("body", &self.body)
            .finish()
    }
}

impl Finalize for Lambda {}

macro_rules! impl_lambda_trace_body {
    ($this:ident, $method:ident) => {
        for (_, ctx) in &$this.defs {
            ctx.$method();
        }
        if let Some(body) = $this.body.as_ref() {
            body.$method();
        }
    };
}

unsafe impl Trace for Lambda {
    unsafe fn trace(&self) {
        impl_lambda_trace_body!(self, trace);
    }
    unsafe fn root(&self) {
        impl_lambda_trace_body!(self, root);
    }
    unsafe fn unroot(&self) {
        impl_lambda_trace_body!(self, unroot);
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_lambda_trace_body!(self, finalize_glue);
    }
}

impl Ast {
    pub fn is_definition(&self) -> bool {
        match self {
            Ast::Apply(_) | Ast::If(_) => false,
            Ast::Seq(elts, last) => {
                elts.iter().all(|elt| elt.is_definition()) && last.is_definition()
            }
            _ => true,
        }
    }

    pub fn as_datum(&self) -> Option<&lexpr::Value> {
        match self {
            Ast::Datum(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_lambda(&self) -> Option<&Gc<GcCell<Lambda>>> {
        match self {
            Ast::Lambda(lambda) => Some(lambda),
            _ => None,
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
    UnboundIdentifier(Rc<str>),
    MaxAnalyzeDepthExceeded,
    UseOfSyntaxAsValue(Rc<str>),
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
            MaxAnalyzeDepthExceeded => write!(f, "maximim analyzer depth exceeded"),
            UseOfSyntaxAsValue(ident) => write!(f, "use of syntax `{}` as value", ident),
        }
    }
}

impl std::error::Error for SyntaxError {}
