use std::{
    fmt::{self, Display, Write},
    rc::Rc,
};

use gc::{Finalize, Gc, GcCell, Trace};

use crate::{
    ast::{Ast, Lambda, Params},
    util::ShowSlice,
};

// Note that currently, this type is designed to be two machine words on a
// 64-bit architecture. The size should be one machine word on both 32-bit and
// 64-bit architectures, but achieving that would require going full-on unsafe,
// so for now, we settle for 2 machine words.
#[derive(Clone, PartialEq)]
pub enum Value {
    Fixnum(isize),
    String(Box<String>),
    Bool(bool),
    Null,
    Unspecified,
    Undefined,
    Cons(Gc<[Value; 2]>),
    Symbol(Rc<String>), // TODO: interning
    PrimOp(&'static PrimOp),
    Lambda(Gc<GcCell<Lambda>>),
    Procedure(Gc<Procedure>),
    Exception(Box<Exception>),
    Core(Core),
}

#[derive(Debug)]
pub struct Procedure {
    pub name: Option<Rc<str>>,
    pub params: Params,
    pub body: Gc<Ast>,
}

impl Finalize for Procedure {}

macro_rules! impl_procedure_trace_body {
    ($this:ident, $method:ident) => {
        $this.body.$method();
    };
}

unsafe impl Trace for Procedure {
    unsafe fn trace(&self) {
        impl_procedure_trace_body!(self, trace);
    }
    unsafe fn root(&self) {
        impl_procedure_trace_body!(self, root);
    }
    unsafe fn unroot(&self) {
        impl_procedure_trace_body!(self, unroot);
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_procedure_trace_body!(self, finalize_glue);
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Core {
    Define,
    Lambda,
    If,
    Begin,
    Quote,
}

impl Core {
    pub fn name(&self) -> &'static str {
        use Core::*;
        match self {
            Define => "define",
            Lambda => "lambda",
            If => "if",
            Begin => "begin",
            Quote => "quote",
        }
    }
    pub fn items() -> &'static [Core] {
        &[
            Core::Define,
            Core::Lambda,
            Core::If,
            Core::Begin,
            Core::Quote,
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Exception {
    pub message: String,
    pub irritants: Vec<Value>,
}

impl Display for Exception {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.message)?;
        if !self.irritants.is_empty() {
            f.write_char(' ')?;
            for (i, irritant) in self.irritants.iter().enumerate() {
                if i + 1 == self.irritants.len() {
                    write!(f, "{}", irritant)?;
                } else {
                    write!(f, "{} ", irritant)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct PrimOp {
    pub name: &'static str,
    pub func: fn(&[Value]) -> Value,
}

impl PartialEq<PrimOp> for PrimOp {
    fn eq(&self, other: &PrimOp) -> bool {
        self as *const _ == other as *const _
    }
}

impl PartialEq<Procedure> for Procedure {
    fn eq(&self, _: &Procedure) -> bool {
        false
    }
}

impl Value {
    pub fn into_result(self) -> Result<Value, Box<Exception>> {
        match self {
            Value::Exception(e) => Err(e),
            _ => Ok(self),
        }
    }

    pub fn list<I>(_elts: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Value>,
    {
        unimplemented!()
    }

    pub fn number<T>(n: T) -> Self
    where
        T: Into<isize>,
    {
        Value::Fixnum(n.into())
    }

    pub fn as_fixnum(&self) -> Option<isize> {
        match self {
            Value::Fixnum(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_true(&self) -> bool {
        if let Value::Bool(v) = self {
            *v
        } else {
            true
        }
    }

    pub fn to_datum(&self) -> Option<lexpr::Value> {
        use Value::*;
        match self {
            Null => Some(lexpr::Value::Null),
            Unspecified => Some(lexpr::Value::Nil),
            Undefined => None,
            Bool(b) => Some((*b).into()),
            Fixnum(n) => Some((*n as i64).into()),
            String(s) => Some(s.as_str().into()),
            Symbol(s) => Some(lexpr::Value::symbol(s.as_str())),
            Cons(cell) => {
                let cell = &*cell;
                match (cell[0].to_datum(), cell[1].to_datum()) {
                    (Some(car), Some(cdr)) => Some((car, cdr).into()),
                    _ => None,
                }
            }
            PrimOp(_) | Exception(_) | Core(_) | Lambda(_) | Procedure(_) => None,
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl<'a> From<&'a str> for Value {
    fn from(s: &'a str) -> Self {
        Value::String(Box::new(s.into()))
    }
}

impl From<Box<Exception>> for Value {
    fn from(e: Box<Exception>) -> Self {
        Value::Exception(e)
    }
}

impl From<&lexpr::Value> for Value {
    fn from(v: &lexpr::Value) -> Self {
        use lexpr::Value::*;
        match v {
            Bool(b) => Value::Bool(*b),
            Number(n) => {
                if let Some(n) = n.as_i64() {
                    if n <= isize::max_value() as i64 {
                        Value::Fixnum(n as isize)
                    } else {
                        unimplemented!()
                    }
                } else {
                    unimplemented!()
                }
            }
            String(s) => s.as_ref().into(),
            Symbol(s) => Value::Symbol(s.as_ref().to_owned().into()),
            Cons(cell) => {
                let (car, cdr) = cell.as_pair();
                Value::Cons(Gc::new([car.into(), cdr.into()]))
            }
            Null => Value::Null,
            Nil => Value::Unspecified,
            _ => unimplemented!(),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Should probably use a more "Rusty" representation
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Fixnum(n) => write!(f, "{}", n),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Bool(b) => f.write_str(if *b { "#t" } else { "#f" }),
            Value::PrimOp(op) => write!(f, "#<prim-op {}>", op.name),
            Value::Null => write!(f, "()"),
            Value::Unspecified => write!(f, "#<unspecified>"),
            Value::Undefined => write!(f, "#<undefined>"),
            Value::Cons(cell) => write_cons(f, cell),
            Value::String(s) => lexpr::Value::string(s.as_str()).fmt(f),
            Value::Exception(e) => write!(
                f,
                "#<exception {} ({})>",
                e.message,
                ShowSlice(&e.irritants)
            ),
            Value::Core(c) => write!(f, "#<core {}>", c.name()),
            Value::Procedure(proc) => {
                if let Some(name) = &proc.name {
                    write!(f, "#<procedure {} {}>", name, proc.params)
                } else {
                    write!(f, "#<procedure {}>", proc.params)
                }
            }
            Value::Lambda(lambda) => {
                let lambda = lambda.borrow();
                if let Some(name) = &lambda.name {
                    write!(f, "#<lambda {} {}>", name, lambda.params)
                } else {
                    write!(f, "#<lambda {}>", lambda.params)
                }
            }
        }
    }
}

fn write_cons(f: &mut fmt::Formatter, cell: &[Value; 2]) -> fmt::Result {
    f.write_char('(')?;
    cell[0].fmt(f)?;
    let mut next = &cell[1];
    loop {
        match next {
            Value::Null => break,
            Value::Cons(cell) => {
                f.write_char(' ')?;
                cell[0].fmt(f)?;
                next = &cell[1];
            }
            value => {
                f.write_str(" . ")?;
                value.fmt(f)?;
                break;
            }
        }
    }
    f.write_char(')')?;
    Ok(())
}

impl gc::Finalize for Value {
    fn finalize(&self) {}
}

macro_rules! impl_value_trace_body {
    ($this:ident, $method:ident) => {
        match $this {
            Value::Cons(cell) => {
                cell[0].$method();
                cell[1].$method();
            }
            Value::Lambda(lambda) => lambda.$method(),
            Value::Procedure(proc) => proc.$method(),
            _ => {}
        }
    };
}

unsafe impl gc::Trace for Value {
    unsafe fn trace(&self) {
        impl_value_trace_body!(self, trace);
    }
    unsafe fn root(&self) {
        impl_value_trace_body!(self, root);
    }
    unsafe fn unroot(&self) {
        impl_value_trace_body!(self, unroot);
    }
    fn finalize_glue(&self) {
        self.finalize();
        impl_value_trace_body!(self, finalize_glue);
    }
}

#[cfg(test)]
mod tests {
    use super::Value;
    use std::mem;

    #[test]
    fn test_value_size() {
        assert!(mem::size_of::<Value>() <= 2 * mem::size_of::<usize>());
    }
}
