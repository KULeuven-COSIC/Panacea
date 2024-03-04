use bitvec::prelude::*;
use dyn_stack::ReborrowMut;
use tfhe::core_crypto::prelude::{
    add_external_product_assign_mem_optimized, decrypt_glwe_ciphertext, encrypt_glwe_ciphertext,
    glwe_ciphertext_add_assign, glwe_ciphertext_sub_assign, par_encrypt_constant_ggsw_ciphertext,
    slice_algorithms::slice_wrapping_sub_assign, DecompositionBaseLog, DecompositionLevelCount,
    DispersionParameter, FourierGgswCiphertext, GgswCiphertext, GlweCiphertext, GlweSecretKey,
    LogStandardDev, Plaintext, PlaintextList, PolynomialSize,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::HashMap,
    fmt::{Display, Formatter},
    time::{Duration, Instant},
};

use crate::{
    context::{Context, FftBuffer},
    num_types::{AlignedScalarContainer, ComplexBox, One, Scalar, ScalarContainer, Zero},
    params::TFHEParameters,
    rlwe::{
        convert_standard_ggsw_to_fourier, expand, gen_all_subs_ksk, less_eq_than, neg_gsw_std,
        FourierRLWEKeyswitchKey,
    },
    utils::log2,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// Comparison operation in the decision node.
pub enum Op {
    LEQ,
    GT,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// A node in the tree.
pub enum Node {
    Internal(Box<Internal>),
    //label inside leaf
    Leaf(usize),
}

impl Default for Node {
    /// Create a tree with depth 1 (one internal node)
    fn default() -> Self {
        let mut processed_one_leaf = false;
        gen_full_tree(1, &mut processed_one_leaf)
    }
}
impl Clone for Node {
    fn clone(&self) -> Node {
        match self {
            Node::Internal(n) => Node::Internal(Box::new(Internal {
                threshold: n.threshold,
                feature: n.feature,
                index: n.index,
                op: n.op,
                left: n.left.clone(),
                right: n.right.clone(),
            })),

            Node::Leaf(label) => Node::Leaf(*label),
        }
    }
}

impl Node {
    /// Turn the node to an Internal, panic if it's a leaf.
    pub fn inner_value(self) -> Internal {
        match self {
            Self::Internal(x) => *x,
            Self::Leaf(_) => panic!("this is a leaf"),
        }
    }
    /// Create a tree with depth d
    pub fn new_with_depth(d: usize) -> Self {
        let mut processed_one_leaf = false;
        gen_full_tree(d, &mut processed_one_leaf)
    }

    /// Assign a unique index to every node in DFS order.
    pub fn fix_index(&mut self) -> usize {
        match self {
            Self::Internal(internal) => fix_index(internal, 0),
            Self::Leaf(_) => panic!("this is a leaf"),
        }
    }

    /// Return the flattened version of the tree.
    /// If `fix_index` is called prior, then the index should be ordered.
    pub fn flatten(&self) -> Vec<Internal> {
        match self {
            Self::Internal(internal) => {
                let mut out = Vec::new();
                // TODO reserve memory
                flatten_tree(&mut out, internal);
                out
            }
            Self::Leaf(_) => vec![],
        }
    }

    /// Evaluate the decision tree with a feature vector and output the final class.
    pub fn eval(&self, features: &Vec<Scalar>) -> usize {
        let mut out = 0;
        eval_node(&mut out, self, features, 1);
        out
    }

    // /// Evaluate the decision tree with a feature vector and output the sums of paths for each leaf
    // pub fn eval_sum(&self, features: &Vec<usize>) -> Vec<GlweCiphertext<AlignedScalarContainer>> {
    //     let mut out = 0;
    //     eval_node(&mut out, self, features, 1);
    //     out
    // }

    /// Count the number of leaves.
    pub fn count_leaf(&self) -> usize {
        match self {
            Self::Internal(internal) => {
                (match &internal.left {
                    Some(node) => node.count_leaf(),
                    None => 0,
                }) + (match &internal.right {
                    Some(node) => node.count_leaf(),
                    None => 0,
                })
            }
            Self::Leaf(_) => 1,
        }
    }

    /// Count the number of internal nodes.
    pub fn count_internal(&self) -> usize {
        match self {
            Self::Internal(internal) => {
                let c = 1
                    + internal
                        .left
                        .as_ref()
                        .unwrap_or(&Self::Leaf(0))
                        .count_internal()
                    + internal
                        .right
                        .as_ref()
                        .unwrap_or(&Self::Leaf(0))
                        .count_internal();
                c
            }
            Self::Leaf(_) => 0,
        }
    }

    /// Count the maximum depth.
    pub fn count_depth(&self) -> usize {
        match self {
            Self::Internal(internal) => {
                let l = internal
                    .left
                    .as_ref()
                    .unwrap_or(&Self::Leaf(0))
                    .count_depth();
                let r = internal
                    .right
                    .as_ref()
                    .unwrap_or(&Self::Leaf(0))
                    .count_depth();
                if l > r {
                    l + 1
                } else {
                    r + 1
                }
            }
            Self::Leaf(_) => 0,
        }
    }

    /// Find the maximum feature index in the tree.
    pub fn max_feature_index(&self) -> usize {
        match self {
            Self::Internal(internal) => {
                let i = internal.feature;
                i.max(
                    internal
                        .left
                        .as_ref()
                        .unwrap_or(&Self::Leaf(0))
                        .max_feature_index(),
                )
                .max(
                    internal
                        .right
                        .as_ref()
                        .unwrap_or(&Self::Leaf(0))
                        .max_feature_index(),
                )
            }
            Self::Leaf(_) => 0,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// An internal node in a decision tree.
pub struct Internal {
    pub threshold: Scalar,
    pub feature: usize,
    pub index: usize,
    pub op: Op,
    pub left: Option<Node>,
    pub right: Option<Node>,
}

fn fix_index(node: &mut Internal, i: usize) -> usize {
    node.index = i;
    let j = match &mut node.left {
        Some(Node::Leaf(_)) | None => i,
        Some(Node::Internal(left)) => fix_index(left, i + 1),
    };
    match &mut node.right {
        Some(Node::Leaf(_)) | None => j,
        Some(Node::Internal(right)) => fix_index(right, j + 1),
    }
}

fn flatten_tree(out: &mut Vec<Internal>, node: &Internal) {
    out.push(Internal {
        threshold: node.threshold,
        feature: node.feature,
        index: node.index,
        op: node.op,
        left: Some(Node::Leaf(0)),
        right: Some(Node::Leaf(0)),
    });
    match &node.left {
        Some(Node::Leaf(_)) | None => (),
        Some(Node::Internal(left)) => flatten_tree(out, left),
    }
    match &node.right {
        Some(Node::Leaf(_)) | None => (),
        Some(Node::Internal(right)) => flatten_tree(out, right),
    }
}

fn eval_node(out: &mut usize, node: &Node, features: &Vec<Scalar>, b: usize) {
    match node {
        Node::Leaf(x) => {
            *out += *x * b;
        }
        Node::Internal(node) => match node.op {
            Op::LEQ => {
                if features[node.feature] <= node.threshold {
                    match &node.left {
                        Some(left) => eval_node(out, left, features, b),
                        None => (),
                    };
                    match &node.right {
                        Some(right) => eval_node(out, right, features, b * (1 - b)),
                        None => (),
                    };
                } else {
                    match &node.left {
                        Some(left) => eval_node(out, left, features, b * (1 - b)),
                        None => (),
                    };
                    match &node.right {
                        Some(right) => eval_node(out, right, features, b),
                        None => (),
                    };
                }
            }
            Op::GT => todo!(),
        },
    }
}

fn gen_full_tree(d: usize, processed_one_leaf: &mut bool) -> Node {
    if d == 0 {
        if *processed_one_leaf {
            Node::Leaf(0)
        } else {
            *processed_one_leaf = true;
            Node::Leaf(1)
        }
    } else {
        Node::Internal(Box::new(Internal {
            threshold: 0,
            feature: 0,
            index: 0,
            op: Op::LEQ,
            left: Some(gen_full_tree(d - 1, processed_one_leaf)),
            right: Some(gen_full_tree(d - 1, processed_one_leaf)),
        }))
    }
}

pub fn trunc_tree(root: &Option<Node>) -> Option<Node> {
    match root {
        Some(Node::Internal(node)) => {
            let left_side = trunc_tree(&node.left);
            let right_side = trunc_tree(&node.right);

            match (&left_side, &right_side) {
                (None, None) => None,
                _ => Some(Node::Internal(Box::new(Internal {
                    threshold: node.threshold,
                    feature: node.feature,
                    index: node.index,
                    op: node.op,
                    left: left_side,
                    right: right_side,
                }))),
            }
        }
        Some(Node::Leaf(label)) => {
            if *label == 0 {
                None
            } else {
                Some(Node::Leaf(*label))
            }
        }
        None => None,
    }
}

/// Perform the comparison operation between (RLWE) encrypted features and the flattened plaintext decision tree
/// and then output an iterator of (RGSW) encrypted choice bits.
pub fn compare_expand<'a>(
    flat_nodes: &'a [Internal],
    client_cts: &'a [Vec<GlweCiphertext<AlignedScalarContainer>>],
    neg_sk_ct: &'a FourierGgswCiphertext<ComplexBox>,
    ksk_map: &'a HashMap<usize, FourierRLWEKeyswitchKey>,
    ctx: &'a Context<Scalar>,
) -> impl Iterator<Item = FourierGgswCiphertext<ComplexBox>> + 'a {
    let mut buf = ctx.gen_fft_ctx();
    flat_nodes.iter().map(move |node| {
        let cts = client_cts[node.feature]
            .iter()
            .map(|c| {
                let mut ct = c.clone();
                match node.op {
                    Op::LEQ => less_eq_than(&mut ct, node.threshold),
                    Op::GT => todo!(),
                }
                ct
            })
            .collect();
        expand(&cts, ksk_map, neg_sk_ct, ctx, &mut buf)
    })
}

/// An encrypted node.
pub enum EncNode {
    Internal(Box<EncInternal>),
    Leaf(usize),
}

impl EncNode {
    /// Create a new root from  a plaintext root and encrypted choice bits.
    pub fn new(
        clear_root: &Node,
        cts: &mut impl Iterator<Item = FourierGgswCiphertext<ComplexBox>>,
    ) -> Self {
        let ct = cts.next().unwrap();
        let mut out = EncInternal {
            ct,
            left: Some(Self::Leaf(0)),
            right: Some(Self::Leaf(0)),
        };
        match clear_root {
            Node::Internal(inner) => new_enc_node(&mut out, inner, cts),
            Node::Leaf(_) => panic!("this is a leaf"),
        }
        Self::Internal(Box::new(out))
    }

    /// Evaluate the tree.
    pub fn eval(
        &self,
        ctx: &Context<Scalar>,
        buf: &mut FftBuffer,
    ) -> Vec<GlweCiphertext<AlignedScalarContainer>> {
        let max_leaf_bits = ((self.max_leaf() + 1) as f64).log2().ceil() as usize;
        let mut out = vec![ctx.empty_glwe_ciphertext(); max_leaf_bits];
        let mut c = ctx.empty_glwe_ciphertext();
        (*c.get_mut_body().as_mut())[0] = Scalar::one();
        ctx.codec.encode(&mut (c.get_mut_body().as_mut())[0]);
        eval_enc_node(&mut out, self, c, ctx, buf);
        out
    }

    /// Every leaf stores a value, output the maximum value out of all the leaves.
    pub fn max_leaf(&self) -> usize {
        match self {
            Self::Internal(internal) => {
                let l = internal.left.as_ref().map(|left| left.max_leaf());

                let r = internal.right.as_ref().map(|right| right.max_leaf());

                match (l, r) {
                    (Some(left), Some(right)) => max(left, right),
                    (None, Some(right)) => right,
                    (Some(left), None) => left,
                    (None, None) => 0,
                }
            }
            Self::Leaf(x) => *x,
        }
    }
}

/// An encrypted internal node where the ciphertext is the choice bit.
pub struct EncInternal {
    pub ct: FourierGgswCiphertext<ComplexBox>,
    pub left: Option<EncNode>,
    pub right: Option<EncNode>,
}

fn new_enc_node(
    enc_node: &mut EncInternal,
    clear_node: &Internal,
    rgsw_cts: &mut impl Iterator<Item = FourierGgswCiphertext<ComplexBox>>,
) {
    match &clear_node.left {
        Some(Node::Leaf(x)) => enc_node.left = Some(EncNode::Leaf(*x)),
        Some(Node::Internal(left)) => match rgsw_cts.next() {
            None => panic!("missing RGSW ciphertext"),
            Some(ct) => {
                let mut new_node = EncInternal {
                    ct,
                    left: Some(EncNode::Leaf(0)),
                    right: Some(EncNode::Leaf(0)),
                };
                new_enc_node(&mut new_node, left, rgsw_cts);
                enc_node.left = Some(EncNode::Internal(Box::new(new_node)));
            }
        },
        None => (),
    }
    match &clear_node.right {
        Some(Node::Leaf(x)) => enc_node.right = Some(EncNode::Leaf(*x)),
        Some(Node::Internal(right)) => match rgsw_cts.next() {
            None => panic!("missing RGSW ciphertext"),
            Some(ct) => {
                let mut new_node = EncInternal {
                    ct,
                    left: Some(EncNode::Leaf(0)),
                    right: Some(EncNode::Leaf(0)),
                };
                new_enc_node(&mut new_node, right, rgsw_cts);
                enc_node.right = Some(EncNode::Internal(Box::new(new_node)));
            }
        },
        None => (),
    }
}

fn eval_enc_node(
    out: &mut Vec<GlweCiphertext<AlignedScalarContainer>>,
    node: &EncNode,
    mut b: GlweCiphertext<AlignedScalarContainer>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) {
    match node {
        EncNode::Leaf(x) => {
            for (bit, ct) in (*x).view_bits::<Lsb0>().iter().zip(out.iter_mut()) {
                if *bit {
                    glwe_ciphertext_add_assign(ct, &b);
                }
            }
        }
        EncNode::Internal(node) => {
            let mut left = ctx.empty_glwe_ciphertext();
            // let mut stack = buf.mem.stack();
            add_external_product_assign_mem_optimized(
                &mut left,
                &node.ct,
                &b,
                buf.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );
            // node.ct.external_product_with_buf(&mut left, &b, buf);

            match &node.right {
                Some(right_child) => {
                    slice_wrapping_sub_assign(b.as_mut(), left.as_ref());

                    eval_enc_node(out, right_child, b, ctx, buf);
                }
                None => (),
            }
            match &node.left {
                Some(left_child) => eval_enc_node(out, left_child, left, ctx, buf),
                None => (),
            }
        }
    }
}

/// Demultiplex a length n vector of RGSW ciphertexts
/// (representing an integer i) into a vector of length 2^n,
/// where the ith vector is `b`.
pub fn demux_with(
    b: GlweCiphertext<AlignedScalarContainer>,
    bits: &Vec<FourierGgswCiphertext<ComplexBox>>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> Vec<GlweCiphertext<AlignedScalarContainer>> {
    demux_rec(0, b, bits, ctx, buf)
}

/// Demultiplex a length n vector of RGSW ciphertexts into
/// a unit vector of length 2^n.
pub fn demux(
    bits: &Vec<FourierGgswCiphertext<ComplexBox>>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> Vec<GlweCiphertext<AlignedScalarContainer>> {
    // NOTE: no need to encrypt here since the "server" generates this ciphertext
    let mut c = ctx.empty_glwe_ciphertext();
    (*c.get_mut_body().as_mut())[0] = Scalar::one();
    ctx.codec.encode(&mut (c.get_mut_body().as_mut())[0]);

    demux_with(c, bits, ctx, buf)
}

fn demux_rec(
    level: usize,
    b: GlweCiphertext<AlignedScalarContainer>,
    bits: &Vec<FourierGgswCiphertext<ComplexBox>>,
    ctx: &Context<Scalar>,
    buf: &mut FftBuffer,
) -> Vec<GlweCiphertext<AlignedScalarContainer>> {
    assert!(level < bits.len());
    let mut left = ctx.empty_glwe_ciphertext();

    add_external_product_assign_mem_optimized(
        &mut left,
        &bits[level],
        &b,
        ctx.fft.as_view(),
        buf.mem.stack().rb_mut(),
    );

    let mut right = b;
    glwe_ciphertext_sub_assign(&mut right, &left);
    if level + 1 == bits.len() {
        // base case:
        vec![left, right]
    } else {
        // recursive case:
        // TODO: check efficiency of extend
        let mut left_output = demux_rec(level + 1, left, bits, ctx, buf);
        let right_output = demux_rec(level + 1, right, bits, ctx, buf);
        left_output.extend(right_output);
        left_output
    }
}

/// Every feature v is encrypted as RLWE(1/(B^j n) X^v) for j in 1...\ell
pub fn encrypt_feature_vector(
    sk: &GlweSecretKey<ScalarContainer>,
    vs: &[Scalar],
    ctx: &mut Context<Scalar>,
) -> Vec<Vec<GlweCiphertext<AlignedScalarContainer>>> {
    let mut pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    let logn = log2(ctx.poly_size.0) as Scalar;
    let mut out = Vec::with_capacity(vs.len());
    for v in vs {
        let mut tmp = Vec::new();
        for level in 1..=ctx.level_count.0 as Scalar {
            assert!(*v < ctx.poly_size.0 as Scalar);
            let shift: Scalar =
                (Scalar::BITS) as Scalar - (ctx.base_log.0 as Scalar) * level - logn;
            pt.as_mut().par_iter_mut().for_each(|x| *x = Scalar::zero());
            (*pt.as_mut())[*v as usize] = Scalar::one() << shift;

            let mut ct = ctx.empty_glwe_ciphertext();
            // TODO consider assign or list
            encrypt_glwe_ciphertext(sk, &mut ct, &pt, ctx.std, &mut ctx.encryption_generator);

            tmp.push(ct);
        }
        out.push(tmp);
    }
    out
}

pub struct SimulationResult {
    pub input_count: usize,
    pub setup_duration: Duration,
    pub server_duration: Duration,
    pub predictions: ScalarContainer,
    pub std: LogStandardDev,
    pub poly_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level_count: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level_count: DecompositionLevelCount,
    pub negs_base_log: DecompositionBaseLog,
    pub negs_level_count: DecompositionLevelCount,
}

impl SimulationResult {
    pub fn new(
        input_count: usize,
        setup_duration: Duration,
        server_duration: Duration,
        predictions: ScalarContainer,
        ctx: &Context<Scalar>,
    ) -> Self {
        Self {
            input_count,
            setup_duration,
            server_duration,
            predictions,
            std: ctx.std,
            poly_size: ctx.poly_size,
            base_log: ctx.base_log,
            level_count: ctx.level_count,
            ks_base_log: ctx.ks_base_log,
            ks_level_count: ctx.ks_level_count,
            negs_base_log: ctx.negs_base_log,
            negs_level_count: ctx.negs_level_count,
        }
    }
}

impl Display for SimulationResult {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "input_count={}, setup_duration={:?}, server_duration={:?}, q={:?}, \
            poly_size={:?}, log_std={:?}, default_decomp=({:?},{:?}), ks_decomp=({:?},{:?}), negs_decomp=({:?},{:?})",
               self.input_count, self.setup_duration, self.server_duration,
               Scalar::BITS, self.poly_size.0, self.std.get_log_standard_dev(),
               self.base_log.0, self.level_count.0, self.ks_base_log.0, self.ks_level_count.0, self.negs_base_log.0, self.negs_level_count.0)
    }
}

fn decrypt_and_recompose(
    sk: &GlweSecretKey<ScalarContainer>,
    cts: &Vec<GlweCiphertext<AlignedScalarContainer>>,
    ctx: &Context<Scalar>,
) -> Scalar {
    let mut bv: BitVec<Scalar, Lsb0> = BitVec::new();
    let mut pt = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
    for ct in cts {
        decrypt_glwe_ciphertext(sk, ct, &mut pt);
        ctx.codec.poly_decode(&mut pt.as_mut_polynomial());
        match pt.as_ref()[0] {
            0 => bv.push(false),
            1 => bv.push(true),
            _ => panic!("expected binary plaintext"),
        }
    }
    bv.load::<Scalar>()
}

/// Simulate PDTE evaluations by specifying a `model`, a set of `features`.
/// If `parallel` is set to true then different features are evaluated in parallel.
/// See the rayon documentation for how to control the number of threads.
/// Finally, return a `SimulationResult` which mainly consists of the timing
/// information and the evaluation result.
pub fn simulate(model: &Node, features: &[Vec<Scalar>], parallel: bool) -> SimulationResult {
    // Client side
    let setup_instant = Instant::now();
    let mut ctx = Context::new(TFHEParameters::default());
    let mut buf = ctx.gen_fft_ctx();
    let sk = GlweSecretKey::generate_new_binary(
        ctx.glwe_dimension,
        ctx.poly_size,
        &mut ctx.secret_generator,
    );

    let neg_sk_ct = convert_standard_ggsw_to_fourier(neg_gsw_std(&sk, &mut ctx), &ctx, &mut buf);
    let ksk_map = gen_all_subs_ksk(&sk, &mut ctx);
    let client_cts: Vec<Vec<Vec<GlweCiphertext<AlignedScalarContainer>>>> = features
        .iter()
        .map(|f| encrypt_feature_vector(&sk, f, &mut ctx))
        .collect();
    let flat_nodes = model.flatten();
    let setup_duration = setup_instant.elapsed();

    // Server side
    let server_f = |ct| {
        let enc_root = {
            let mut rgsw_cts = compare_expand(&flat_nodes, ct, &neg_sk_ct, &ksk_map, &ctx);
            EncNode::new(model, &mut rgsw_cts)
        };
        // NOTE: we need to create new buffers for every operation since it's not thread safe
        let mut buf = FftBuffer::new(ctx.poly_size);
        enc_root.eval(&ctx, &mut buf)
    };
    let server_instant = Instant::now();
    let output_cts: Vec<Vec<GlweCiphertext<AlignedScalarContainer>>> = if parallel {
        client_cts.par_iter().map(|ct| server_f(ct)).collect()
    } else {
        client_cts.iter().map(|ct| server_f(ct)).collect()
    };
    let server_duration = server_instant.elapsed();

    // Check correctness by doing the evaluation on plaintext model
    let mut predictions = vec![];
    for (ct, feature) in output_cts.iter().zip(features.iter()) {
        let actual_scalar = decrypt_and_recompose(&sk, ct, &ctx);
        let expected_scalar = model.eval(feature) as Scalar;
        assert_eq!(expected_scalar, actual_scalar);
        predictions.push(expected_scalar);
    }

    let input_count = features.len();
    SimulationResult::new(
        input_count,
        setup_duration,
        server_duration,
        predictions,
        &ctx,
    )
}

/// Convert `i` to a bit vector and pad it to length `n`,
/// and then encrypt it into RGSW ciphertexts.
pub fn bit_decomposed_rgsw(
    i: usize,
    n: usize,
    sk: &GlweSecretKey<ScalarContainer>,
    ctx: &mut Context<Scalar>,
) -> Vec<GgswCiphertext<ScalarContainer>> {
    let mut i_bits = i.view_bits::<Lsb0>()[..log2(n)].to_bitvec();
    assert!(i_bits.len() <= log2(n));
    i_bits.reverse();

    let mut a: Vec<GgswCiphertext<ScalarContainer>> = vec![
        GgswCiphertext::new(
            Scalar::zero(),
            ctx.glwe_size,
            ctx.poly_size,
            ctx.base_log,
            ctx.level_count,
            ctx.ciphertext_modulus,
        );
        i_bits.len()
    ];

    for i in 0..a.len() {
        par_encrypt_constant_ggsw_ciphertext(
            sk,
            &mut a[i],
            if i_bits[i] {
                Plaintext(Scalar::zero())
            } else {
                Plaintext(Scalar::one())
            },
            ctx.std,
            &mut ctx.encryption_generator,
        )
    }
    a
}

#[cfg(test)]
mod test {
    use tfhe::core_crypto::prelude::{
        convert_standard_ggsw_ciphertext_to_fourier_mem_optimized,
        par_encrypt_constant_ggsw_ciphertext, Plaintext,
    };

    use crate::{params::TFHEParameters, utils::compute_noise_encoded};

    use super::*;

    #[test]
    fn test_json() {
        let root = Node::Internal(Box::new(Internal {
            threshold: 1,
            feature: 2,
            index: 4,
            op: Op::LEQ,
            left: Some(Node::Internal(Box::new(Internal {
                threshold: 11,
                feature: 22,
                index: 44,
                op: Op::GT,
                left: Some(Node::Leaf(1)),
                right: Some(Node::Leaf(2)),
            }))),
            right: Some(Node::Leaf(3)),
        }));
        assert_eq!(
            r#"{"internal":{"threshold":1,"feature":2,"index":4,"op":"leq","left":{"internal":{"threshold":11,"feature":22,"index":44,"op":"gt","left":{"leaf":1},"right":{"leaf":2}}},"right":{"leaf":3}}}"#,
            serde_json::to_string(&root).unwrap()
        );
    }

    #[test]
    fn test_clear_node() {
        let mut root = Node::Internal(Box::new(Internal {
            threshold: 1,
            feature: 2,
            index: 4,
            op: Op::LEQ,
            left: Some(Node::Internal(Box::new(Internal {
                threshold: 11,
                feature: 22,
                index: 44,
                op: Op::GT,
                left: Some(Node::Leaf(0)),
                right: Some(Node::default()),
            }))),
            right: Some(Node::Leaf(0)),
        }));

        assert_eq!(root.fix_index(), 2);
        for (i, x) in root.flatten().iter().enumerate() {
            assert_eq!(x.index, i);
        }

        let internal = root.inner_value();
        assert_eq!(internal.index, 0);
        let left = internal.left.unwrap().inner_value();
        assert_eq!(left.index, 1);
        let right = left.right.unwrap().inner_value();
        assert_eq!(right.index, 2);
    }

    #[test]
    fn test_traversal_1() {
        // In this example we consider 2 features, 2 labels and and 3 nodes as shown below
        //        f_0           f_0 <= 2
        //       /   \
        //      f_1   l_0       f_1 <= 2
        //     /  \
        //   l_0   f_1          f_1 <= 3
        //        /  \
        //     l_10  l_0

        let root = {
            let mut tmp = Node::Internal(Box::new(Internal {
                threshold: 2,
                feature: 0,
                index: 0,
                op: Op::LEQ,
                left: Some(Node::Internal(Box::new(Internal {
                    threshold: 2,
                    feature: 1,
                    index: 0,
                    op: Op::LEQ,
                    left: Some(Node::Leaf(0)),
                    right: Some(Node::Internal(Box::new(Internal {
                        threshold: 3,
                        feature: 1,
                        index: 0,
                        op: Op::LEQ,
                        left: Some(Node::Leaf(10)),
                        right: Some(Node::Leaf(0)),
                    }))),
                }))),
                right: Some(Node::Leaf(0)),
            }));
            assert_eq!(tmp.fix_index(), 2);
            tmp
        };
        assert_eq!(root.count_leaf(), 4);
        assert_eq!(root.count_internal(), 3);
        assert_eq!(root.max_feature_index(), 1);

        let features = vec![2, 3]; // f_0 = 2, f_1 = 3
        assert_eq!(10, root.eval(&features));

        simulate(&root, &[features], false);
    }

    #[test]
    fn test_traversal_2() {
        // In this example we consider 2 features, 2 labels and and 3 nodes as shown below
        //        f_0           f_0 <= 2
        //       /   \
        //      f_1   l_1       f_1 <= 2
        //     /  \
        //   l_1   f_1          f_1 <= 1
        //        /  \
        //     l_1  l_0

        let root = {
            let mut tmp = Node::Internal(Box::new(Internal {
                threshold: 2,
                feature: 0,
                index: 0,
                op: Op::LEQ,
                left: Some(Node::Internal(Box::new(Internal {
                    threshold: 2,
                    feature: 1,
                    index: 0,
                    op: Op::LEQ,
                    left: Some(Node::Leaf(1)),
                    right: Some(Node::Internal(Box::new(Internal {
                        threshold: 1,
                        feature: 1,
                        index: 0,
                        op: Op::LEQ,
                        left: Some(Node::Leaf(1)),
                        right: Some(Node::Leaf(0)),
                    }))),
                }))),
                right: Some(Node::Leaf(1)),
            }));
            assert_eq!(tmp.fix_index(), 2);
            tmp
        };
        assert_eq!(root.count_leaf(), 4);
        assert_eq!(root.count_internal(), 3);
        assert_eq!(root.max_feature_index(), 1);

        let features = vec![2, 3]; // f_0 = 2, f_1 = 3
        assert_eq!(0, root.eval(&features));

        simulate(&root, &[features], false);
    }

    #[test]
    fn test_traversal_long() {
        const TH: Scalar = 10;
        const D: usize = 10;
        fn gen_line(d: usize) -> Node {
            if d == 0 {
                Node::Leaf(1)
            } else {
                Node::Internal(Box::new(Internal {
                    threshold: TH,
                    feature: 0,
                    index: 0,
                    op: Op::LEQ,
                    left: Some(gen_line(d - 1)),
                    right: Some(Node::Leaf(0)),
                }))
            }
        }
        let root = {
            let mut out = gen_line(D);
            assert_eq!(out.fix_index(), D - 1);
            out
        };
        {
            let features = vec![1];
            assert_eq!(1, root.eval(&features));
            simulate(&root, &[features], false);
        }
        {
            let features = vec![11];
            assert_eq!(0, root.eval(&features));
            simulate(&root, &[features], false);
        }
    }

    #[test]
    fn test_depth() {
        assert_eq!(Node::new_with_depth(0).count_depth(), 0);
        assert_eq!(Node::new_with_depth(1).count_depth(), 1);
        assert_eq!(Node::new_with_depth(3).count_depth(), 3);
    }

    #[test]
    fn test_bitvec() {
        // test conversion
        let mut bv_one: BitVec<usize, Lsb0> = BitVec::new();
        bv_one.push(true);
        let mut bv_two: BitVec<usize, Lsb0> = BitVec::new();
        bv_two.push(false);
        bv_two.push(true);

        assert_eq!(bv_one.load::<usize>(), 1);
        assert_eq!(bv_two.load::<usize>(), 2);

        // test decomposition
        let v = 10usize; // 1010
        let v_bits = v.view_bits::<Lsb0>().to_bitvec();
        assert!(!v_bits[0]);
        assert!(v_bits[1]);
        assert!(!v_bits[2]);
        assert!(v_bits[3]);
    }

    #[test]
    fn test_demux() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut buf = ctx.gen_fft_ctx();
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );
        let pt_zero = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        let mut pt_unit = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        (*pt_unit.as_mut())[0] = Scalar::one();

        let mut pt0 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        let mut pt1 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        let mut pt2 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());
        let mut pt3 = PlaintextList::new(Scalar::zero(), ctx.plaintext_count());

        {
            // test 11

            let mut bits = vec![
                FourierGgswCiphertext::new(
                    ctx.glwe_size,
                    ctx.poly_size,
                    ctx.base_log,
                    ctx.level_count
                );
                2
            ];

            let mut tmp = GgswCiphertext::new(
                Scalar::zero(),
                ctx.glwe_size,
                ctx.poly_size,
                ctx.base_log,
                ctx.level_count,
                ctx.ciphertext_modulus,
            );
            par_encrypt_constant_ggsw_ciphertext(
                &sk,
                &mut tmp,
                Plaintext(Scalar::one()),
                ctx.std,
                &mut ctx.encryption_generator,
            );

            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
                &tmp,
                &mut bits[0],
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );

            par_encrypt_constant_ggsw_ciphertext(
                &sk,
                &mut tmp,
                Plaintext(Scalar::one()),
                ctx.std,
                &mut ctx.encryption_generator,
            );

            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
                &tmp,
                &mut bits[1],
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );

            let cts = demux(&bits, &ctx, &mut buf);

            assert_eq!(cts.len(), 4);

            decrypt_glwe_ciphertext(&sk, &cts[0], &mut pt0);
            ctx.codec.poly_decode(&mut pt0.as_mut_polynomial());
            decrypt_glwe_ciphertext(&sk, &cts[1], &mut pt1);
            ctx.codec.poly_decode(&mut pt1.as_mut_polynomial());
            decrypt_glwe_ciphertext(&sk, &cts[2], &mut pt2);
            ctx.codec.poly_decode(&mut pt2.as_mut_polynomial());
            decrypt_glwe_ciphertext(&sk, &cts[3], &mut pt3);
            ctx.codec.poly_decode(&mut pt3.as_mut_polynomial());

            assert_eq!(pt0, pt_unit);
            assert_eq!(pt1, pt_zero);
            assert_eq!(pt2, pt_zero);
            assert_eq!(pt3, pt_zero);

            // check noise
            println!(
                "noise for depth=2: {}",
                compute_noise_encoded(&sk, &cts[1], &pt_zero, &ctx.codec)
            );
        }

        {
            // test 00

            let mut bits = vec![
                FourierGgswCiphertext::new(
                    ctx.glwe_size,
                    ctx.poly_size,
                    ctx.base_log,
                    ctx.level_count
                );
                2
            ];

            let mut tmp = GgswCiphertext::new(
                Scalar::zero(),
                ctx.glwe_size,
                ctx.poly_size,
                ctx.base_log,
                ctx.level_count,
                ctx.ciphertext_modulus,
            );
            par_encrypt_constant_ggsw_ciphertext(
                &sk,
                &mut tmp,
                Plaintext(Scalar::zero()),
                ctx.std,
                &mut ctx.encryption_generator,
            );

            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
                &tmp,
                &mut bits[0],
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );

            par_encrypt_constant_ggsw_ciphertext(
                &sk,
                &mut tmp,
                Plaintext(Scalar::zero()),
                ctx.std,
                &mut ctx.encryption_generator,
            );

            convert_standard_ggsw_ciphertext_to_fourier_mem_optimized(
                &tmp,
                &mut bits[1],
                ctx.fft.as_view(),
                buf.mem.stack().rb_mut(),
            );

            let cts = demux(&bits, &ctx, &mut buf);

            assert_eq!(cts.len(), 4);

            decrypt_glwe_ciphertext(&sk, &cts[0], &mut pt0);
            ctx.codec.poly_decode(&mut pt0.as_mut_polynomial());
            decrypt_glwe_ciphertext(&sk, &cts[1], &mut pt1);
            ctx.codec.poly_decode(&mut pt1.as_mut_polynomial());
            decrypt_glwe_ciphertext(&sk, &cts[2], &mut pt2);
            ctx.codec.poly_decode(&mut pt2.as_mut_polynomial());
            decrypt_glwe_ciphertext(&sk, &cts[3], &mut pt3);
            ctx.codec.poly_decode(&mut pt3.as_mut_polynomial());

            assert_eq!(pt0, pt_zero);
            assert_eq!(pt1, pt_zero);
            assert_eq!(pt2, pt_zero);
            assert_eq!(pt3, pt_unit);

            // check noise
            println!(
                "noise for depth=2: {}",
                compute_noise_encoded(&sk, &cts[1], &pt_zero, &ctx.codec)
            );
        }
    }

    #[test]
    fn test_demux_long() {
        let mut ctx = Context::new(TFHEParameters::default());
        let mut buf = ctx.gen_fft_ctx();
        let sk = GlweSecretKey::generate_new_binary(
            ctx.glwe_dimension,
            ctx.poly_size,
            &mut ctx.secret_generator,
        );

        let depth = 4usize;

        for i in 0..(1 << depth) {
            let bits = bit_decomposed_rgsw(i, 1 << depth, &sk, &mut ctx)
                .iter()
                .map(|ct| convert_standard_ggsw_to_fourier(ct.clone(), &ctx, &mut buf))
                .collect();

            let cts = demux(&bits, &ctx, &mut buf);
            assert_eq!(cts.len(), 1 << depth);

            let pt_unit = ctx.gen_unit_pt();
            let pt_zero = ctx.gen_zero_pt();

            let mut pt = ctx.gen_zero_pt();
            for (j, ct) in cts.iter().enumerate().take(1 << depth) {
                decrypt_glwe_ciphertext(&sk, ct, &mut pt);
                ctx.codec.poly_decode(&mut pt.as_mut_polynomial());

                if j == i {
                    assert_eq!(pt, pt_unit);
                } else {
                    assert_eq!(pt, pt_zero);
                }
            }

            // check noise
            println!(
                "first noise for depth={}, i={}: {}",
                depth,
                i,
                compute_noise_encoded(&sk, &cts[0], &pt_unit, &ctx.codec)
            );
            println!(
                "last noise for depth={}, i={}: {}",
                depth,
                i,
                compute_noise_encoded(&sk, &cts[(1 << depth) - 1], &pt_zero, &ctx.codec)
            );
        }
    }
}
