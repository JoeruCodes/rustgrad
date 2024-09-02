extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemStruct, Type};

#[proc_macro_attribute]
pub fn init_c_struct_t(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut fields = Vec::new();

    let field_parser = syn::meta::parser(|meta| {
        let field_name = meta.path.get_ident().unwrap();
        let field_type = meta.value()?.parse::<Type>().unwrap();
        fields.push(quote! {
            pub #field_name : #field_type
        });
        Ok(())
    });

    parse_macro_input!(attr with field_parser);

    let input = parse_macro_input!(item as ItemStruct);

    let struct_name = &input.ident;
    let expanded = quote! {
        pub struct #struct_name {
            #(#fields),*
        }
    };

    TokenStream::from(expanded)
}
