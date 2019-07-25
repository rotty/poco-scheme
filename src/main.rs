use std::{
    env, fs,
    io::{self, BufRead},
    path::Path,
};

use poco_scheme::{EvalError, Value, Vm};

fn load(vm: &mut Vm, path: impl AsRef<Path>) -> Result<(), EvalError> {
    let file = fs::File::open(path)?;
    let parser = lexpr::Parser::from_reader(file);
    for res in vm.process(parser) {
        let _ = res?;
    }
    Ok(())
}

fn main() -> Result<(), EvalError> {
    env_logger::init();

    let args: Vec<_> = env::args_os().skip(1).collect();
    let mut vm = Vm::new();
    if args.is_empty() {
        let input = io::BufReader::new(io::stdin());
        for res in vm.process::<_, EvalError>(input.lines().map(|line| Ok(line?.parse()?))) {
            match res {
                Ok(value) => {
                    if value != Value::Unspecified {
                        println!("{}", value);
                    }
                }
                Err(e) => println!("; error: {}", e),
            }
        }
    } else {
        for filename in &args {
            load(&mut vm, filename)?;
        }
    }
    Ok(())
}
