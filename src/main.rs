use std::collections::HashMap;
use std::fs::File;

use pprof::ProfilerGuardBuilder;
use rustgrad_2::helpers::{extract_callers, Profiling};
use rustgrad_2::{create_new_context, parse_context_kwargs};

fn main() {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();
    let _ = fibonacci(8);
    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
    };
}

fn fibonacci(n: u32) -> Option<u128> {
    if n <= 1 {
        return Some(n as u128);
    }
    let (mut a, mut b) = (0u128, 1u128);
    for _ in 2..=n {
        let new_b = a.checked_add(b)?;
        a = b;
        b = new_b;
    }
    Some(b)
}

pub fn fn_name(args: String) -> Result<impl serde::Serialize, anyhow::Error> {
    use rustgrad_2::helpers::{diskcache_get, diskcache_put};
    use serde_json;
    // Generate the table name
    let table = format!("cache_{}", "some");

    // Generate the cache key by serializing the arguments
    let key = {
        let args_tuple = args;
        serde_json::to_string(&args_tuple).map_err(|e| anyhow::anyhow!(e))?
    };

    // Try to get the value from the cache
    if let Some(cached) = diskcache_get(&table, Box::new(key.clone())) {
        return Ok(cached);
    }

    // Execute the original function and cache the result
    let result = "".to_string();
    diskcache_put(&table, Box::new(key), result.clone())?;

    Ok(result)
}

