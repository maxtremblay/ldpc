use crate::css::{Css, CssOperator, CssSyndromeView};

use super::{ClassicalSyndromeDecoder, SyndromeDecoder};
pub type CssDecoder<D> = Css<D>;

impl<'a, D> SyndromeDecoder<CssSyndromeView<'a>, CssOperator> for CssDecoder<D>
where
    D: ClassicalSyndromeDecoder<'a>,
{
    fn correction_for(&self, syndrome: CssSyndromeView<'a>) -> CssOperator {
        self.as_ref()
            .pair(syndrome)
            .map(|(decoder, syndrome)| decoder.correction_for(syndrome.clone()))
            .swap_xz()
    }
}

#[cfg(test)]
mod test {
}
