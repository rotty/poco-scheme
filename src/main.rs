use std::{
    env, fs,
    io::{self, BufRead},
    path::Path,
};

use poco_scheme::{eval_toplevel, EvalError};

fn load(path: impl AsRef<Path>) -> Result<(), EvalError> {
    let file = fs::File::open(path)?;
    let parser = lexpr::Parser::from_reader(file);
    eval_toplevel(parser.map(|e| e.map_err(Into::into)), |res| {
        let _ = res?;
        Ok(())
    })?;
    Ok(())
}

fn main() -> Result<(), EvalError> {
    env_logger::init();

    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.is_empty() {
        let input = io::BufReader::new(io::stdin());
        eval_toplevel(
            input.lines().map(|line| {
                let line = line?;
                Ok(line.parse()?)
            }),
            |res| {
                match res {
                    Ok(value) => println!("{}", value),
                    Err(e) => println!("; error: {}", e),
                }
                Ok(())
            },
        )?;
    } else {
        for filename in &args {
            load(filename)?;
        }
    }
    Ok(())
}
