extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn diskcache(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();
    let block = &input.block;
    let args = &input.sig.inputs;

    // Extract function argument names
    let arg_names: Vec<_> = args.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(ident) = &*pat_type.pat {
                Some(ident.ident.clone())
            } else {
                None
            }
        } else {
            None
        }
    }).collect();

    let gen = quote! {

        pub fn #fn_name(#args) -> Result<impl serde::Serialize, anyhow::Error> {
            use serde::Serialize;
            use crate::helpers::{diskcache_put, diskcache_get};
            use serde_json;
            // Generate the table name
            let table = format!("cache_{}", #fn_name_str);
            
            // Generate the cache key by serializing the arguments
            let key = {
                let args_tuple = (#(#arg_names.clone()),*);
                serde_json::to_string(&args_tuple).map_err(|e| anyhow::anyhow!(e))?
            };

            // Try to get the value from the cache
            if let Some(cached) = diskcache_get(&table, Box::new(key.clone())) {
                return Ok(cached);
            }

            // Execute the original function and cache the result
            let result = (|| #block)();
            let _ = diskcache_put(&table, Box::new(key), result.clone())?;

            Ok(result)
        }
    };

    gen.into()
}
