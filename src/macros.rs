#[inline]
fn assert_same_len(a: (usize, Option<usize>), b: (usize, Option<usize>)) {
    debug_assert_eq!(a.1, Some(a.0));
    debug_assert_eq!(b.1, Some(b.0));
    debug_assert_eq!(a.0, b.0);
}

pub trait ZipChecked: IntoIterator + Sized {
    #[inline]
    fn zip_checked<B: IntoIterator>(
        self,
        b: B,
    ) -> core::iter::Zip<<Self as IntoIterator>::IntoIter, <B as IntoIterator>::IntoIter> {
        let a = self.into_iter();
        let b = b.into_iter();
        assert_same_len(a.size_hint(), b.size_hint());
        core::iter::zip(a, b)
    }
}

impl<A: IntoIterator> ZipChecked for A {}

// https://docs.rs/itertools/0.7.8/src/itertools/lib.rs.html#247-269
macro_rules! izip {
    // eg. __izip_closure!(((a, b), c) => (a, b, c) , dd , ee )
    (@ __closure @ $p:pat => $tup:expr) => {
        |$p| $tup
    };

    // The "b" identifier is a different identifier on each recursion level thanks to hygiene.
    (@ __closure @ $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )*) => {
        izip!(@ __closure @ ($p, b) => ( $($tup)*, b ) $( , $tail )*)
    };

    ( $first:expr $(,)?) => {
        {
            #[allow(unused_imports)]
            use $crate::macros::ZipChecked;
            ::core::iter::IntoIterator::into_iter($first)
        }
    };
    ( $first:expr, $($rest:expr),+ $(,)?) => {
        {
            #[allow(unused_imports)]
            use $crate::macros::ZipChecked;
            ::core::iter::IntoIterator::into_iter($first)
                $(.zip_checked($rest))*
                .map(izip!(@ __closure @ a => (a) $( , $rest )*))
        }
    };
}
