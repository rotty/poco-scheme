use std::{fmt, fs, path::Path};

use lexpr::sexp;

use r3_scheme::{eval, EvalError, Value};

// Poor man's `failure` crate emulation
#[derive(Debug)]
struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! format_err {
    ($($fmtargs:expr),*) => {
        Error(format!($($fmtargs),*))
    }
}

impl<T> From<T> for Error
where
    T: std::error::Error,
{
    fn from(e: T) -> Self {
        Error(e.to_string())
    }
}

type TestResult = Result<(), TestError>;

#[derive(Debug)]
struct TestError {
    description: lexpr::Value,
    kind: TestErrorKind,
}

#[derive(Debug)]
enum TestErrorKind {
    EvalFail(EvalError),
    Unexpected {
        result: Value,
        expected: lexpr::Value,
    },
}

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TestErrorKind::*;
        match &self.kind {
            EvalFail(e) => write!(f, "{}: evaluation failed: {}", self.description, e),
            Unexpected { result, expected } => write!(
                f,
                "{} failed: got {}, expected {}",
                self.description, result, expected
            ),
        }
    }
}

#[derive(Debug)]
struct Test {
    description: lexpr::Value,
    expr: lexpr::Value,
    expected: lexpr::Value,
}

struct ErrorList<'a>(&'a [TestError]);

impl<'a> fmt::Display for ErrorList<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for error in self.0 {
            write!(f, "- {}\n", error)?;
        }
        Ok(())
    }
}

impl Test {
    fn new(spec: lexpr::Value) -> Result<Self, Error> {
        // TODO (lexpr): this should be less tediuous
        let parts = spec
            .as_cons()
            .map(lexpr::Cons::to_ref_vec)
            .and_then(|(elts, rest)| if rest.is_null() { Some(elts) } else { None })
            .ok_or_else(|| format_err!("expected list, got {}", spec))?;
        // TODO (lexpr): This would benefit from some pattern matches on S-expression level
        if parts.len() != 5 || parts[0] != &sexp!(check) || parts[3] != &sexp!(#"=>") {
            return Err(format_err!(
                "test case needs to be of form `(check <description> <test> => <result>)`"
            ));
        }
        Ok(Test {
            description: parts[1].clone(),
            expr: parts[2].clone(),
            expected: parts[4].clone(),
        })
    }
    fn run(&self) -> TestResult {
        let result = eval(&self.expr).map_err(|e| TestError {
            description: self.description.clone(),
            kind: TestErrorKind::EvalFail(e),
        })?;
        if result.to_datum().as_ref() != Some(&self.expected) {
            return Err(TestError {
                description: self.description.clone(),
                kind: TestErrorKind::Unexpected {
                    result,
                    expected: self.expected.clone(),
                }
            });
        }
        Ok(())
    }
}

fn run_scheme_test_file(path: &Path) -> Result<Vec<TestResult>, Error> {
    let file = fs::File::open(path)?;
    let parser = lexpr::Parser::from_reader(file);
    let results = parser
        .map(|item| {
            let datum = item
                .map_err(|e| format_err!("parsing test file {} failed: {}", path.display(), e))?;
            let test = Test::new(datum)?;
            Ok(test.run())
        })
        .collect::<Result<_, Error>>()?;
    Ok(results)
}

#[test]
fn run_scheme_tests() {
    for entry in fs::read_dir("tests/scheme").expect("test dir not found") {
        let path = entry.expect("reading test dir failed").path();
        if let Some("scm") = path.extension().and_then(|e| e.to_str()) {
            let errors: Vec<_> = run_scheme_test_file(&path)
                .unwrap_or_else(|e| {
                    panic!("error running tests in {}: {}", path.display(), e);
                })
                .into_iter()
                .filter_map(Result::err)
                .collect();
            if !errors.is_empty() {
                panic!(
                    "failed tests in {}:\n{}",
                    path.display(),
                    ErrorList(&errors)
                );
            }
        }
    }
}
