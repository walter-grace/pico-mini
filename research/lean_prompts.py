#!/usr/bin/env python3
"""
Generate Lean 4 proof prompts for Harmonic AI.

Flow: kv-lab finds winning technique → this generates a rigorous Lean prompt
→ paste into Harmonic → get verified .lean file back → proof becomes constraint.

Equations sourced from:
[1] Jacques (QJL mapping + Buffon's Needle)
[2] 1-bit QJL asymmetric inner product estimator
[3] PolarQuant recursive Cartesian-to-polar
[4] TurboQuant information-theoretic lower bounds (Shannon)
[5] RaBitQ unbiased distance estimation
[6] HRR / Vector Symbolic Architectures
"""

import os
import json
from datetime import datetime

PROOFS_DIR = os.path.join(os.path.dirname(__file__), "proofs")


# ─── Technique → proof mapping ───────────────────────────────────────────────

PROOF_REGISTRY = {
    "qjl_1bit": ["qjl_mapping", "qjl_asymmetric_unbiased", "jl_distance_preservation", "rabitq_unbiased_distance"],
    "hadamard_rotate": ["hadamard_norm_preservation"],
    "polar_quant": ["polar_reconstruction", "polar_angle_bound"],
    "baseline_minmax": ["minmax_error_bound", "turboquant_shannon_bound", "banaszczyk_rounding"],
    "residual_correction": ["residual_monotonic_improvement", "hyperbolic_residual_quantization"],
    "lloyd_max": ["lloyd_max_optimality", "turboquant_shannon_bound", "leech_lattice_quantization"],
    "mixed_kv": ["minmax_error_bound", "banaszczyk_rounding"],
    "per_layer_adaptive": ["minmax_error_bound"],
    # Advanced techniques
    "hyperbolic_rq": ["hyperbolic_residual_quantization", "hyperbolic_mobius_gyrogroup", "hyperbolic_error_bound"],
    "leech_lattice": ["leech_lattice_quantization", "leech_lattice_decoding", "leech_lattice_distortion_rate"],
    "hrr_vsa": ["hrr_circular_convolution", "hrr_retrieval_accuracy", "hrr_capacity_bound"],
    "banaszczyk_quant": ["banaszczyk_rounding", "banaszczyk_kv_cache", "banaszczyk_vs_naive"],
}


def generate_lean_prompts():
    """All proof prompts with rigorous mathematical formulations."""

    prompts = {
        # ─── QJL Mapping (Jacques / Buffon's Needle) ─────────────────────
        "qjl_mapping": {
            "technique": "qjl_1bit",
            "property": "Quantized Johnson-Lindenstrauss quasi-isometric embedding",
            "prompt": r"""Generate a Lean 4 proof for the Quantized Johnson-Lindenstrauss mapping.

DEFINITIONS:
- Random Gaussian matrix: Φ ~ N^{M×N}(0, 1)
- Uniform dithering vector: ξ ~ U^M([0, δ])
- Uniform quantizer: Q_δ(λ) = δ · ⌊λ/δ⌋
- QJL mapping: ψ_δ(x) := Q_δ(Φx + ξ)

THEOREM (Quasi-Isometric Embedding):
For any two vectors u, v ∈ ℝ^N, with high probability:

  (1 - ε)·‖u - v‖₂ - c·δ  ≤  (c'/M)·‖ψ(u) - ψ(v)‖₁  ≤  (1 + ε)·‖u - v‖₂ + c·δ

where both the multiplicative distortion ε and additive distortion c·δ decay as
O(√(log S / M)) as the reduced dimension M increases.

PROOF STRATEGY:
1. The dithering ξ makes quantization error uniform on [0, δ], independent of input
2. Apply standard JL concentration to Φx, then bound the quantization perturbation
3. The ℓ₁ norm estimator (not ℓ₂) arises from the connection to Buffon's Needle:
   the probability a quantization boundary falls between two projected values
   is proportional to their distance
4. Union bound over all S pairs gives the O(√(log S / M)) rate

Use Mathlib4 imports for:
- Mathlib.Probability.Distributions.Gaussian
- Mathlib.Analysis.NormedSpace.Basic
- Mathlib.MeasureTheory.Measure.Lebesgue
""",
        },

        # ─── 1-Bit QJL Asymmetric Estimator ──────────────────────────────
        "qjl_asymmetric_unbiased": {
            "technique": "qjl_1bit",
            "property": "1-bit QJL asymmetric inner product estimator is strictly unbiased",
            "prompt": r"""Generate a Lean 4 proof for the 1-bit QJL asymmetric estimator.

DEFINITIONS:
- Random projection matrix: S ∈ ℝ^{m×d} with i.i.d. entries from N(0,1)
- Key quantizer (1-bit): H_S(k) := sign(Sk)
- Asymmetric estimator: Prod_QJL(q, k) := √(π/2) · (1/m) · ‖k‖₂ · ⟨Sq, H_S(k)⟩

Only the cached key vector k is quantized. The query q stays in full precision.

THEOREM (Unbiased Estimation):
  𝔼_S[Prod_QJL(q, k)] = ⟨q, k⟩

PROOF:
1. Orthogonal decomposition: q = (⟨q,k⟩/‖k‖²)·k + q_{⊥k}
   where q_{⊥k} is the component orthogonal to k.

2. For the parallel component:
   ⟨S·(αk), sign(Sk)⟩ = α · ⟨Sk, sign(Sk)⟩ = α · ‖Sk‖₁
   where α = ⟨q,k⟩/‖k‖²

3. Key identity: For x ~ N(0, σ²), 𝔼[|x|] = σ·√(2/π)
   Therefore: 𝔼[‖Sk‖₁] = m · ‖k‖₂ · √(2/π)

4. For the perpendicular component:
   Each row sᵢ gives: 𝔼[⟨sᵢ, q_{⊥k}⟩ · sign(⟨sᵢ, k⟩)]
   Since q_{⊥k} ⊥ k, the projections ⟨sᵢ, q_{⊥k}⟩ and ⟨sᵢ, k⟩ are
   independent Gaussians. For independent X, Y ~ N(0,·):
   𝔼[X · sign(Y)] = 𝔼[X] · 𝔼[sign(Y)] = 0
   So cross-terms vanish.

5. Combining:
   𝔼[Prod_QJL] = √(π/2) · (1/m) · ‖k‖₂ · m · (⟨q,k⟩/‖k‖²) · ‖k‖₂ · √(2/π)
                = √(π/2) · √(2/π) · ⟨q,k⟩
                = ⟨q, k⟩  ∎

Use Mathlib4 imports for:
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.Probability.Distributions.Gaussian
- Mathlib.MeasureTheory.Integral.SetIntegral
""",
        },

        # ─── JL Distance Preservation ────────────────────────────────────
        "jl_distance_preservation": {
            "technique": "qjl_1bit",
            "property": "Johnson-Lindenstrauss distance preservation lemma",
            "prompt": r"""Generate a Lean 4 proof for the Johnson-Lindenstrauss Lemma.

THEOREM:
For n points in ℝ^d, ε ∈ (0, 1/2), and random linear map
f(x) = (1/√k)·Rx where R ∈ ℝ^{k×d} has i.i.d. N(0,1) entries
and k = O(ε⁻² log n):

With probability ≥ 1 - 1/n, for ALL pairs u, v:
  (1-ε)·‖u-v‖² ≤ ‖f(u)-f(v)‖² ≤ (1+ε)·‖u-v‖²

PROOF STRATEGY:
1. Fix a unit vector w = (u-v)/‖u-v‖. Then ‖f(u)-f(v)‖² = ‖u-v‖² · (1/k)·‖Rw‖²
2. ‖Rw‖² = Σᵢ (rᵢᵀw)² where each rᵢᵀw ~ N(0,1), so ‖Rw‖² ~ χ²(k)
3. Chi-squared concentration: P(|‖Rw‖²/k - 1| > ε) ≤ 2·exp(-kε²/8)
4. Union bound over all (n choose 2) ≤ n² pairs
5. Setting k = 8ε⁻²·ln(n) makes failure probability ≤ 2n²·exp(-ln n²) = 2/n²

Use Mathlib4 imports for:
- Mathlib.Probability.Moments
- Mathlib.Analysis.SpecificLimits.Basic
""",
        },

        # ─── Hadamard Norm Preservation ───────────────────────────────────
        "hadamard_norm_preservation": {
            "technique": "hadamard_rotate",
            "property": "Hadamard rotation preserves L2 norm (orthogonal invariance)",
            "prompt": r"""Generate a Lean 4 proof for Hadamard norm preservation.

DEFINITION:
Normalized Hadamard matrix H ∈ ℝ^{n×n} satisfying H·Hᵀ = I (orthogonal).

THEOREM:
  For all x ∈ ℝ^n: ‖Hx‖₂ = ‖x‖₂

PROOF:
  ‖Hx‖₂² = (Hx)ᵀ(Hx) = xᵀHᵀHx = xᵀIx = xᵀx = ‖x‖₂²
  Since norms are non-negative, taking square roots preserves equality. ∎

COROLLARY (Outlier Spreading):
If x has entries with high kurtosis (outliers), Hx distributes energy more
uniformly across coordinates. This means subsequent uniform quantization has
lower maximum per-element error.

Use Mathlib4 imports for:
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.LinearAlgebra.Matrix.NonsingularInverse
- Mathlib.LinearAlgebra.UnitaryGroup
""",
        },

        # ─── PolarQuant Recursive Reconstruction ─────────────────────────
        "polar_reconstruction": {
            "technique": "polar_quant",
            "property": "PolarQuant exact reconstruction from polar coordinates",
            "prompt": r"""Generate a Lean 4 proof for PolarQuant's recursive Cartesian-to-polar transformation.

DEFINITIONS:
For a vector x ∈ ℝ^d (where d is a power of 2):

Level-1 angles: ψⱼ^(1) := arctan(x_{2j} / x_{2j-1})  for j = 1, ..., d/2

The full recursive polar decomposition gives angles at each level ℓ = 1, ..., log₂(d).

THEOREM (Exact Reconstruction):
Each coordinate xᵢ can be perfectly reconstructed as:

  xᵢ = ‖x‖₂ · ∏_{ℓ=1}^{log₂ d} cos(ψ_{⌊i/2^ℓ⌋}^(ℓ))^{𝟙{i mod 2^ℓ ≤ 2^{ℓ-1}}}
             · ∏_{ℓ=1}^{log₂ d} sin(ψ_{⌊i/2^ℓ⌋}^(ℓ))^{𝟙{i mod 2^ℓ > 2^{ℓ-1}}}

PROOF STRATEGY:
1. Base case (d=2): x = ‖x‖₂ · (cos θ, sin θ) where θ = arctan(x₂/x₁)
2. Inductive step: split x into even/odd halves, each half has its own radius
   and angle. The radius of each half relates to the parent radius via cos/sin.
3. The product telescope reconstructs exactly because cos²θ + sin²θ = 1 at each level.

COROLLARY:
Quantizing only the angles (not per-block scales) avoids storing normalization
constants. A Lloyd-Max quantizer on angles captures geometric boundaries naturally.

Use Mathlib4 imports for:
- Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
- Mathlib.Analysis.InnerProductSpace.Basic
""",
        },

        # ─── PolarQuant Angle Error Bound ─────────────────────────────────
        "polar_angle_bound": {
            "technique": "polar_quant",
            "property": "PolarQuant angle quantization error is bounded",
            "prompt": r"""Generate a Lean 4 proof for PolarQuant's angle quantization error bound.

DEFINITIONS:
- Uniform quantizer with 2^b levels on [0, π]
- Step size: δ = π / 2^b
- Nearest-level quantizer: Q(θ) = δ · round(θ/δ)

THEOREM (Angle Error Bound):
  For any θ ∈ [0, π]: |θ - Q(θ)| ≤ δ/2 = π / 2^{b+1}

THEOREM (Inner Product Error Bound):
  For unit vectors v, v_q where v_q has quantized angles:
  |⟨v, v_q⟩ - 1| ≤ 1 - cos(π / 2^{b+1})

  For small angles: 1 - cos(x) ≈ x²/2, so the error scales as O(π² / 2^{2b+2}).

PROOF:
1. Uniform quantization to nearest level has max error δ/2 (pigeonhole)
2. Inner product between unit vectors: ⟨v, v_q⟩ = cos(angle between them)
3. The angle between v and v_q is at most π/2^{b+1} (from angle quantization error)
4. cos is monotone decreasing on [0, π], so cos(π/2^{b+1}) ≤ ⟨v, v_q⟩ ≤ 1

Use Mathlib4 imports for:
- Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
- Mathlib.Topology.Order.Basic
""",
        },

        # ─── Min-Max Quantization Error Bound ─────────────────────────────
        "minmax_error_bound": {
            "technique": "baseline_minmax",
            "property": "Per-group min-max quantization error is bounded",
            "prompt": r"""Generate a Lean 4 proof for per-group min-max quantization error.

DEFINITIONS:
- Values x₁, ..., xₙ ∈ ℝ with min a := min(xᵢ) and max b := max(xᵢ)
- Uniform quantizer with 2^B levels:
  Q(x) = a + round((x-a)/(b-a) · (2^B - 1)) · (b-a)/(2^B - 1)

THEOREM (Pointwise Error Bound):
  For all i: |xᵢ - Q(xᵢ)| ≤ (b - a) / (2 · (2^B - 1))

THEOREM (MSE Bound):
  (1/n) · Σᵢ (xᵢ - Q(xᵢ))² ≤ ((b - a) / (2 · (2^B - 1)))²

PROOF:
1. The quantization grid has spacing Δ = (b-a)/(2^B - 1)
2. Rounding to nearest grid point has max error Δ/2
3. Every xᵢ ∈ [a, b] is within Δ/2 of some grid point (pigeonhole on intervals)
4. MSE ≤ max error² because (1/n)Σeᵢ² ≤ max(eᵢ²) when all |eᵢ| ≤ Δ/2

Use Mathlib4 imports for:
- Mathlib.Analysis.MeanInequalities
- Mathlib.Order.Bounds.Basic
""",
        },

        # ─── TurboQuant Shannon Lower Bound ───────────────────────────────
        "turboquant_shannon_bound": {
            "technique": "baseline_minmax",
            "property": "TurboQuant MSE is within ~2.7x of Shannon Lower Bound",
            "prompt": r"""Generate a Lean 4 proof for TurboQuant's information-theoretic optimality.

DEFINITIONS:
- Random vector x ∈ ℝ^d with differential entropy h(x)
- Total bit budget: B bits
- Distortion-rate function D(B) = MSE achievable with B bits

THEOREM (Shannon Lower Bound on Distortion):
  D(p_X, B) ≥ (d / 2πe) · 2^{(2/d)(h(x) - B)}

This is the absolute minimum MSE ANY quantizer can achieve with B bits.

THEOREM (TurboQuant Near-Optimality):
  MSE_TurboQuant ≤ 2.7 · D_Shannon(B)

TurboQuant's per-group asymmetric quantization achieves MSE within a small
constant factor (~2.7) of the Shannon lower bound.

PROOF STRATEGY:
1. Shannon lower bound follows from rate-distortion theory:
   R(D) = h(x) - (d/2)·log₂(2πeD/d) for Gaussian sources
   Inverting: D(R) = (d/2πe)·2^{-2R/d} where R = B/d bits per dimension
2. For the general case, replace Gaussian entropy with h(x)
3. TurboQuant's gap comes from: (a) uniform vs optimal quantization within groups,
   (b) finite group size overhead for storing min/max per group
4. The 2.7 factor is: 2^{2·(entropy gap)} where entropy gap accounts for
   uniform quantizer on bounded support vs optimal Lloyd-Max

Use Mathlib4 imports for:
- Mathlib.MeasureTheory.Measure.MeasureSpace
- Mathlib.Analysis.SpecificLimits.Basic
- Mathlib.InformationTheory.Entropy
""",
        },

        # ─── Residual Correction ──────────────────────────────────────────
        "residual_monotonic_improvement": {
            "technique": "residual_correction",
            "property": "Residual correction monotonically reduces error",
            "prompt": r"""Generate a Lean 4 proof for residual correction improvement.

DEFINITIONS:
- Original vector: x ∈ ℝ^d
- Base quantizer Q_B with error: e_B = x - Q_B(x)
- Residual quantizer Q_R applied to the error e_B
- Reconstruction: x̂ = Q_B(x) + Q_R(e_B)

THEOREM (Error Reduction):
  ‖x - x̂‖ = ‖e_B - Q_R(e_B)‖

If Q_R has any compression at all (i.e., it's not the zero function):
  ‖e_B - Q_R(e_B)‖ ≤ ‖e_B‖

THEOREM (QJL Residual — Expected Improvement):
If Q_R is the 1-bit QJL estimator with projection dimension m, then:
  𝔼[‖x - x̂‖²] ≤ 𝔼[‖e_B‖²] · (1 - 1/m)

So each residual correction step reduces expected squared error by factor (1-1/m).

PROOF:
1. x - x̂ = x - Q_B(x) - Q_R(e_B) = e_B - Q_R(e_B) (substitution)
2. If Q_R is unbiased: 𝔼[Q_R(e_B)] = e_B
3. Variance decomposition: 𝔼[‖e_B - Q_R(e_B)‖²] = Var(Q_R(e_B))
4. For QJL with m projections: Var = ‖e_B‖² · (1 - 1/m) (from chi-squared variance)

Use Mathlib4 imports for:
- Mathlib.Probability.Variance
- Mathlib.Analysis.InnerProductSpace.Basic
""",
        },

        # ─── Lloyd-Max Optimality ─────────────────────────────────────────
        "lloyd_max_optimality": {
            "technique": "lloyd_max",
            "property": "Lloyd-Max quantizer minimizes MSE for fixed codebook size",
            "prompt": r"""Generate a Lean 4 proof for Lloyd-Max quantizer optimality.

DEFINITIONS:
- Random variable X with PDF f(x)
- Quantizer with K levels: {y₁, ..., y_K} (reconstruction points)
- Decision boundaries: {b₀, b₁, ..., b_K} where b₀ = -∞, b_K = +∞
- MSE distortion: D = 𝔼[(X - Q(X))²] = Σₖ ∫_{b_{k-1}}^{b_k} (x - yₖ)² f(x) dx

THEOREM (Necessary Conditions for Optimality):
The MSE-optimal quantizer satisfies two conditions simultaneously:

1. Centroid condition (optimal reconstruction):
   yₖ = 𝔼[X | b_{k-1} < X ≤ bₖ] = ∫_{b_{k-1}}^{b_k} x·f(x)dx / ∫_{b_{k-1}}^{b_k} f(x)dx

2. Nearest-neighbor condition (optimal boundaries):
   bₖ = (yₖ + y_{k+1}) / 2

THEOREM (Convergence):
The Lloyd-Max algorithm (alternating between conditions 1 and 2) produces a
monotonically non-increasing sequence of distortions D₀ ≥ D₁ ≥ D₂ ≥ ...
that converges to a local minimum.

PROOF:
1. Each centroid update minimizes D with boundaries fixed (calculus of variations)
2. Each boundary update minimizes D with centroids fixed (nearest-neighbor decoding)
3. D is bounded below by 0, and each step is non-increasing → convergence by
   monotone convergence theorem

Use Mathlib4 imports for:
- Mathlib.MeasureTheory.Integral.SetIntegral
- Mathlib.Order.Filter.Basic
""",
        },

        # ─── RaBitQ Unbiased Distance ────────────────────────────────────
        "rabitq_unbiased_distance": {
            "technique": "qjl_1bit",
            "property": "RaBitQ unbiased distance estimation via binary quantization",
            "prompt": r"""Generate a Lean 4 proof for RaBitQ's unbiased distance estimation.

DEFINITIONS:
- Centroid: c (e.g., k-means centroid of the database)
- Raw data vector: o_r, Raw query vector: q_r
- Normalized: o = (o_r - c)/‖o_r - c‖, q = (q_r - c)/‖q_r - c‖

THEOREM (Distance Decomposition):
  ‖o_r - q_r‖² = ‖o_r - c‖² + ‖q_r - c‖² - 2·‖o_r - c‖·‖q_r - c‖·⟨q, o⟩

The first two terms are scalar (precomputed). Only ⟨q, o⟩ needs estimation.

THEOREM (Unbiased Binary Estimation):
By quantizing o into a binary vector b ∈ {-1, +1}^d with appropriate random
rounding, the estimator:
  ⟨q, o⟩_est = ⟨q, b⟩ · correction_factor

is unbiased: 𝔼[⟨q, o⟩_est] = ⟨q, o⟩

And can be computed using bitwise POPCNT operations in O(d/64) time.

PROOF:
1. Expand ‖o_r - q_r‖² using the parallelogram identity around c
2. Factor out the norms to isolate ⟨q, o⟩ as the only non-scalar term
3. Random rounding: P(bᵢ = 1) = (oᵢ + 1)/2 makes 𝔼[bᵢ] = oᵢ
4. Linearity of expectation: 𝔼[⟨q, b⟩] = ⟨q, 𝔼[b]⟩ = ⟨q, o⟩

Use Mathlib4 imports for:
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.Probability.Independence.Basic
""",
        },

        # ─── HRR / VSA Binding ────────────────────────────────────────────
        "hrr_circular_convolution": {
            "technique": "hrr_vsa",
            "property": "HRR circular convolution is invertible and capacity-preserving",
            "prompt": r"""Generate a Lean 4 proof for Holographic Reduced Representations.

DEFINITIONS:
- Vectors x, y ∈ ℝ^d (hyperdimensional, d ≥ 1000)
- Binding via circular convolution:
  (x ⊛ y)_j = Σ_{k=0}^{d-1} x_k · y_{(j-k) mod d}
- Factors through the DFT:
  x ⊛ y = F⁻¹(F(x) ⊙ F(y))
  where F is the d-point DFT and ⊙ is Hadamard (elementwise) product.
- Bundling (superposition): s = Σ_{i=1}^n (aᵢ ⊛ vᵢ)
- Unbinding via circular correlation:
  (x ⋆ y)_j = Σ_{k=0}^{d-1} x_k · y_{(j+k) mod d}
  equivalently: x ⋆ y = F⁻¹(conj(F(x)) ⊙ F(y))

RETRIEVAL IDENTITY:
For bundle s = Σ_{i=1}^n (aᵢ ⊛ vᵢ), querying with aⱼ yields:
  aⱼ ⋆ s = (aⱼ ⋆ aⱼ) ⊛ vⱼ  +  Σ_{i≠j} aⱼ ⋆ (aᵢ ⊛ vᵢ)
            \_____________/      \________________________/
               signal ≈ vⱼ          cross-talk noise η

Signal term: aⱼ ⋆ aⱼ = F⁻¹(|F(aⱼ)|²), concentrates at index 0 for i.i.d. aⱼ.

DISTANCE ESTIMATION from bundle:
  ⟨q, vⱼ⟩_est = ⟨aⱼ ⋆ s, q⟩ = ⟨vⱼ, q⟩ · (1/d · ‖F(aⱼ)‖²) + ⟨ηⱼ, q⟩

THEOREM (Approximate Inverse):
  aⱼ ⋆ (aⱼ ⊛ vⱼ) ≈ vⱼ with entrywise error O(1/√d)

PROOF (via Fourier):
1. F(x ⊛ y) = F(x) · F(y) (convolution theorem)
2. aⱼ ⋆ (aⱼ ⊛ vⱼ) = F⁻¹(|F(aⱼ)|² · F(vⱼ))
3. For i.i.d. aⱼ ~ N(0, 1/d): |F(aⱼ)_k|² are i.i.d. χ²(2)/(2d)
4. By LLN: (1/d)Σ|F(aⱼ)_k|² → 1/d, off-diagonal auto-correlation has var O(1/d²)
5. After rescaling, signal ≈ vⱼ with O(1/√d) error per entry

Use Mathlib4 imports for:
- Mathlib.Analysis.Fourier.FourierTransform
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.Probability.Moments
""",
        },

        # ─── HRR Retrieval Accuracy ──────────────────────────────────────
        "hrr_retrieval_accuracy": {
            "technique": "hrr_vsa",
            "property": "HRR holographic bundle retrieval accuracy bound",
            "prompt": r"""Generate a Lean 4 proof for HRR retrieval accuracy.

DEFINITIONS:
- Address vectors: a₁, ..., aₙ ~iid N(0, d⁻¹·I_d)
- Value vectors: v₁, ..., vₙ with ‖vᵢ‖ = 1 (fixed, unit norm)
- Holographic bundle: s = Σ_{i=1}^n (aᵢ ⊛ vᵢ) ∈ ℝ^d
- Inner product estimate: ⟨q, vⱼ⟩_est from unbinding aⱼ ⋆ s

NOISE ANALYSIS:
Each cross-talk term aⱼ ⋆ (aᵢ ⊛ vᵢ) for i ≠ j has entries that are sums of
products of independent Gaussians. Conditioned on vᵢ:
  Var[(ηⱼ)_l] = (n-1)/d² · v̄²
where v̄² = (1/(n-1))·Σ_{i≠j} ‖vᵢ‖².

Projected noise variance:
  Var[⟨ηⱼ, q⟩] = (n-1)/d · v̄² · ‖q‖² · 1/d

Signal-to-noise ratio:
  SNR = |⟨vⱼ, q⟩|² / Var[⟨ηⱼ, q⟩] = Θ(d / (n-1))

THEOREM (Retrieval Accuracy):
For unit query q and target index j:
  Pr[|⟨q, vⱼ⟩_est - ⟨q, vⱼ⟩| > ε] ≤ 2·exp(-c·ε²·d/n)

for universal constant c > 0.

COROLLARY (Capacity):
- To store n vectors with error ≤ ε at failure prob ≤ δ:
  d = Ω(n/ε² · log(1/δ))
- At fixed d, max storable vectors: n* = O(ε²·d / log(1/δ))
- For (1+ε)-ANN over n vectors: d = Ω(n/ε² · log n)

PROOF:
1. Noise ηⱼ is a sum of (n-1) independent sub-Gaussian random vectors
2. ⟨ηⱼ, q⟩ is sub-Gaussian with parameter σ² = O(n/d)
3. Apply Hoeffding/sub-Gaussian tail bound
4. Union bound over n candidates for ANN application

Use Mathlib4 imports for:
- Mathlib.Probability.Moments
- Mathlib.MeasureTheory.Measure.MeasureSpace
""",
        },

        # ─── HRR Capacity Bound ──────────────────────────────────────────
        "hrr_capacity_bound": {
            "technique": "hrr_vsa",
            "property": "HRR compression ratio and query complexity",
            "prompt": r"""Generate a Lean 4 proof for HRR compression and query complexity.

THEOREM (Compression):
A holographic bundle s = Σ_{i=1}^n (aᵢ ⊛ vᵢ) maps n×d floats to a single
d-dimensional vector — a compression ratio of n:1.

THEOREM (Query Complexity):
Unbinding requires one circular correlation: O(d·log d) time via FFT.
This is independent of n (the number of stored vectors).

THEOREM (Compression-Accuracy Tradeoff):
At compression ratio n:1 with dimension d, retrieval error satisfies:
  𝔼[|⟨q, vⱼ⟩_est - ⟨q, vⱼ⟩|²] = O(n/d)

So to achieve MSE ≤ ε² at compression ratio n:1:
  d ≥ C · n / ε²

Equivalently, at fixed d, the maximum compression ratio with error ≤ ε is:
  n_max = O(ε² · d)

Use Mathlib4 imports for:
- Mathlib.Analysis.Fourier.FourierTransform
- Mathlib.Probability.Variance
""",
        },

        # ═══════════════════════════════════════════════════════════════════
        # I. HYPERBOLIC RESIDUAL QUANTIZATION
        # ═══════════════════════════════════════════════════════════════════

        "hyperbolic_residual_quantization": {
            "technique": "hyperbolic_rq",
            "property": "Hyperbolic residual quantization in Poincaré ball model",
            "prompt": r"""Generate a Lean 4 proof for Hyperbolic Residual Quantization.

DEFINITIONS:
- Poincaré ball: 𝔹^d = { x ∈ ℝ^d : ‖x‖₂ < 1 }
- Riemannian metric: g_x^𝔹 = λ_x² · g^E  where λ_x = 2/(1 - ‖x‖²)
- Geodesic distance:
  d_𝔹(x, y) = arccosh(1 + 2·‖x-y‖² / ((1-‖x‖²)(1-‖y‖²)))

- Möbius addition (gyrogroup operation on 𝔹^d):
  x ⊕_M y = ((1 + 2⟨x,y⟩ + ‖y‖²)·x + (1 - ‖x‖²)·y) / (1 + 2⟨x,y⟩ + ‖x‖²·‖y‖²)

- Möbius negation: ⊖_M x = -x
- Möbius subtraction: x ⊖_M y = x ⊕_M (-y)

RESIDUAL QUANTIZATION (M stages, codebooks C₁,...,C_M ⊂ 𝔹^d):
- Initialize r₀ = x
- Stage m: c_m = argmin_{c ∈ C_m} d_𝔹(r_{m-1}, c)
- Hyperbolic residual: r_m = (⊖_M c_m) ⊕_M r_{m-1}
- Reconstruction: x̂ = c₁ ⊕_M c₂ ⊕_M ... ⊕_M c_M

THEOREM (Left Cancellation):
  The Möbius residual r_m = (⊖_M c_m) ⊕_M r_{m-1} is the unique point satisfying
  c_m ⊕_M r_m = r_{m-1}
  (exploiting the left-cancellation law of the Möbius gyrogroup)

THEOREM (Residual Concentration):
  Successive residuals are gyro-translated toward the origin, where λ_x ≈ 2
  and local geometry is approximately Euclidean. This means later RQ stages
  operate where standard VQ (Lloyd's algorithm) converges stably.

LOSS FUNCTION:
  L(C₁,...,C_M) = 𝔼_{x~D}[d_𝔹(x, x̂)²]

RIEMANNIAN GRADIENT for codebook training:
  grad_𝔹 L = (1/λ_{c_m}²) · ∇_E L
Updates via exponential map:
  exp_{c_m}(v) = c_m ⊕_M (tanh(λ_{c_m}·‖v‖/2) · v/‖v‖)

Use Mathlib4 imports for:
- Mathlib.Geometry.Manifold.Instances.Real
- Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
- Mathlib.Topology.MetricSpace.Basic
""",
        },

        "hyperbolic_mobius_gyrogroup": {
            "technique": "hyperbolic_rq",
            "property": "Möbius addition forms a gyrogroup with left cancellation",
            "prompt": r"""Generate a Lean 4 proof that Möbius addition forms a gyrogroup.

DEFINITIONS:
- Poincaré ball: 𝔹^d = { x ∈ ℝ^d : ‖x‖₂ < 1 }
- Möbius addition:
  x ⊕_M y = ((1 + 2⟨x,y⟩ + ‖y‖²)·x + (1 - ‖x‖²)·y) / (1 + 2⟨x,y⟩ + ‖x‖²·‖y‖²)

THEOREM (Gyrogroup Properties):
1. Closure: x, y ∈ 𝔹^d ⟹ x ⊕_M y ∈ 𝔹^d
   Proof: Show ‖x ⊕_M y‖ < 1 using the identity:
   1 - ‖x ⊕_M y‖² = (1-‖x‖²)(1-‖y‖²) / (1 + 2⟨x,y⟩ + ‖x‖²·‖y‖²)²
   · (1 + 2⟨x,y⟩ + ‖x‖²·‖y‖²)
   The denominator is > 0 (by Cauchy-Schwarz: 2|⟨x,y⟩| ≤ 2‖x‖·‖y‖ < ‖x‖²+‖y‖²+1)

2. Identity: 0 ⊕_M y = y and x ⊕_M 0 = x

3. Left cancellation: (⊖_M a) ⊕_M (a ⊕_M b) = b
   This is the key property enabling residual quantization.

4. Gyroassociativity:
   a ⊕_M (b ⊕_M c) = (a ⊕_M b) ⊕_M gyr[a,b](c)
   where gyr[a,b] is the Thomas gyration (a rotation in ℝ^d).

PROOF OF LEFT CANCELLATION:
Direct algebraic verification: expand (⊖_M a) ⊕_M (a ⊕_M b) using the Möbius
formula twice, simplify the numerator and denominator, and show it equals b.
The key algebraic identity used:
  1 + 2⟨-a, a⊕_M b⟩ + ‖a⊕_M b‖² = (1-‖a‖²)² · (1+2⟨a,b⟩+‖b‖²) / denom²

Use Mathlib4 imports for:
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.Algebra.Group.Defs
""",
        },

        "hyperbolic_error_bound": {
            "technique": "hyperbolic_rq",
            "property": "Hyperbolic RQ distortion-rate bound via volume growth",
            "prompt": r"""Generate a Lean 4 proof for hyperbolic RQ error bounds.

KEY INSIGHT — EXPONENTIAL VOLUME GROWTH:
In Euclidean ℝ^d: Vol(ball of radius r) ∝ r^d
In hyperbolic ℍ^d:
  Vol(B_ℍ(r)) = ω_{d-1} · ∫₀ʳ sinh^{d-1}(t) dt ~ (ω_{d-1} / (2^{d-1}·(d-1))) · e^{(d-1)r}

where ω_{d-1} is the surface area of the Euclidean unit (d-1)-sphere.

This exponential growth means K centroids cover exponentially more volume in ℍ^d
than in ℝ^d — far fewer centroids needed for hierarchical/tree-like data.

COVERING NUMBER:
  N(X, ε, d_𝔹) ≤ Vol(X_ε) / Vol(B_ℍ(ε))
where X_ε is the ε-thickening in ℍ^d.

THEOREM (Distortion Bound):
For M-stage RQ with K centroids per stage, data with hyperbolic doubling
dimension δ_ℍ (smallest δ s.t. any ball of radius 2r coverable by 2^δ balls of radius r):

  𝔼[d_𝔹(x, x̂)²] ≤ C_d · K^{-2/δ_ℍ} · ρ^{2(M-1)}

where ρ < 1 is a contraction factor from curvature-adapted regularity.

DISTORTION-RATE:
Total bitrate R = M·log₂(K) bits per vector:
  D(R) ≲ C_d · 2^{-2R/δ_ℍ}

EUCLIDEAN LIMIT:
When curvature κ → 0: δ_ℍ → d, ρ → K^{-1/d}, recovering classical O(K^{-2M/d}).

KEY ADVANTAGE:
When δ_ℍ ≪ d (data has low hyperbolic doubling dimension — tree-like structure):
  D(R) ~ 2^{-2R/δ_ℍ}  ≫  2^{-2R/d} (Euclidean rate)
Dramatically better compression for hierarchical data.

Use Mathlib4 imports for:
- Mathlib.Geometry.Manifold.Instances.Real
- Mathlib.MeasureTheory.Measure.Hausdorff
- Mathlib.Analysis.SpecialFunctions.ExpDeriv
""",
        },

        # ═══════════════════════════════════════════════════════════════════
        # II. LEECH LATTICE SPHERE-PACKING QUANTIZATION
        # ═══════════════════════════════════════════════════════════════════

        "leech_lattice_quantization": {
            "technique": "leech_lattice",
            "property": "Leech lattice Λ₂₄ achieves near-optimal quantization (0.8% of optimal)",
            "prompt": r"""Generate a Lean 4 proof for Leech lattice quantization optimality.

DEFINITIONS:
- Leech lattice Λ₂₄: unique even unimodular lattice in ℝ²⁴ with no vectors of squared norm 2
- Construction via extended binary Golay code G₂₄:
  Λ₂₄ = (1/√8) · { x ∈ ℤ²⁴ ∪ (ℤ+½)²⁴ : x ≡ C₀ (mod 2), Σᵢxᵢ ≡ 0 (mod 4) }
  where C₀ ranges over codewords of G₂₄

- Lattice quantizer: Q_Λ(x) = argmin_{λ ∈ Λ₂₄} ‖x - λ‖₂²
- For d-dim data (d = 24k): partition into k blocks of 24, quantize each:
  x̂ = (Q_Λ(x⁽¹⁾/Δ), ..., Q_Λ(x⁽ᵏ⁾/Δ)) · Δ

NORMALIZED SECOND MOMENT (NSM):
  G(Λ) = (1/n · ∫_{V₀} ‖x‖² dx) / Vol(V₀)^{1+2/n}
  where V₀ is the Voronoi region and n is lattice dimension.

For Leech lattice: G(Λ₂₄) ≈ 0.06583

ZADOR LOWER BOUND (any quantizer in dimension n):
  G_n* ≥ (1/2πe) · n/(n+2) · Γ(1+n/2)^{2/n} · π⁻¹
  which → 1/(2πe) ≈ 0.0585 as n → ∞

At n=24: lower bound ≈ 0.0653, so Λ₂₄ is within 0.8% of optimal.

SPACE-FILLING LOSS:
  D_{Λ₂₄}(R) / D*(R) = 2πe · G(Λ₂₄) ≈ 0.06583/0.05855 ≈ 1.124
  i.e., only 0.51 dB above the Shannon limit.

Use Mathlib4 imports for:
- Mathlib.NumberTheory.Lattice
- Mathlib.Analysis.MeanInequalities
- Mathlib.MeasureTheory.Measure.Lebesgue
""",
        },

        "leech_lattice_decoding": {
            "technique": "leech_lattice",
            "property": "Leech lattice O(n log n) closest-vector decoding via Golay code",
            "prompt": r"""Generate a Lean 4 proof for Leech lattice decoding complexity.

DECODER (Conway-Sloane / Vardy-Be'ery via Golay code):
The Golay code G₂₄ is a [24, 12, 8] binary code.

Phase 1 — Reduce modulo sublattice:
  f = x - 2·⌊x/2⌉  (fractional part in [-1, 1]²⁴)

Phase 2 — Syndrome computation:
  s = f mod G₂₄ via parity check matrix H of Golay code
  Identifies the correct coset of Λ₂₄/(2ℤ²⁴)

Phase 3 — Coset search:
  |(ℤ²⁴/2ℤ²⁴)/G₂₄| = 2²⁴/2¹² = 4096 cosets
  Golay's minimum distance 8 ensures O(1) coset leaders to check per coset

THEOREM (Complexity):
  Total decoding: O(n·log n) where n = 24
  - Syndrome decode: O(24·12) for matrix-vector product
  - Coset enumeration: O(24·log 24) for sorting-based search
  - Bounded enumeration per coset: O(1) due to min distance 8

THEOREM (Correctness):
  Q_Λ(x) returns the closest lattice point to x in Euclidean distance.
  For any x ∈ ℝ²⁴, the decoded point λ* satisfies:
  ‖x - λ*‖₂ = min_{λ ∈ Λ₂₄} ‖x - λ‖₂

Use Mathlib4 imports for:
- Mathlib.InformationTheory.Code.Binary
- Mathlib.Combinatorics.Designs
""",
        },

        "leech_lattice_distortion_rate": {
            "technique": "leech_lattice",
            "property": "Leech lattice distortion-rate with reverse water-filling",
            "prompt": r"""Generate a Lean 4 proof for Leech lattice distortion-rate performance.

DEFINITIONS:
- Single block: X ~ N(0, σ²·I₂₄), rate R bits/dimension
- Full vector: d = 24k, blocks x⁽¹⁾,...,x⁽ᵏ⁾ each in ℝ²⁴

THEOREM (Per-Block Distortion-Rate):
  D(R) = 24 · G(Λ₂₄) · σ² · 2^{-2R} ≤ 1.580 · σ² · 2^{-2R}

THEOREM (Shannon Comparison):
  D*(R) = σ² · 2^{-2R}  (Gaussian source Shannon DRF)
  Ratio: D_{Λ₂₄}/D* = 24·G(Λ₂₄) ≈ 1.580, i.e., 0.99 dB gap

THEOREM (Full Vector with Optimal Bit Allocation):
Under reverse water-filling across k blocks with per-block variances σᵢ²:
  D_total(R_total) = G(Λ₂₄) · 24 · (∏ᵢσᵢ²)^{1/k} · 2^{-2R_total/k}

PROOF:
1. For Gaussian X, MSE = 24·G(Λ)·Δ²·det(Λ)^{1/12}
2. Rate constraint: R = (1/24)·log₂(σ²⁴/det(Δ·Λ₂₄)) bits/dim
3. Optimize Δ: ∂D/∂Δ = 0 gives Δ* in terms of σ and R
4. Substitute back: D(R) = 24·G(Λ₂₄)·σ²·2^{-2R}
5. Multi-block: Lagrange multiplier on total rate → water-filling

Use Mathlib4 imports for:
- Mathlib.Analysis.Calculus.MeanValue
- Mathlib.MeasureTheory.Integral.SetIntegral
""",
        },

        # ═══════════════════════════════════════════════════════════════════
        # IV. BANASZCZYK'S VECTOR BALANCING AND QUANTIZATION ROUNDING
        # ═══════════════════════════════════════════════════════════════════

        "banaszczyk_rounding": {
            "technique": "banaszczyk_quant",
            "property": "Banaszczyk's theorem: vector balancing with Gaussian potential",
            "prompt": r"""Generate a Lean 4 proof for Banaszczyk's Vector Balancing Theorem.

THEOREM (Banaszczyk, 1998):
Let v₁, ..., vₙ ∈ ℝ^d with ‖vᵢ‖₂ ≤ 1 for all i.
Let K ⊂ ℝ^d be a convex body with γ_d(K) ≥ 1/2, where:
  γ_d(K) = ∫_K (2π)^{-d/2} · e^{-‖x‖²/2} dx  (standard Gaussian measure)

Then there exist signs ε₁, ..., εₙ ∈ {-1, +1} such that:
  Σ_{i=1}^n εᵢ·vᵢ ∈ 5K

APPLICATION TO QUANTIZATION ROUNDING:
Given x ∈ ℝ^d, constraint matrix A ∈ ℝ^{m×d} with columns a₁,...,a_d, ‖aⱼ‖₂ ≤ 1.
Find x̂ ∈ ℤ^d such that:
  ‖A(x - x̂)‖_∞ ≤ O(√(log(2m)))

Write x = ⌊x⌋ + f with f ∈ [0,1)^d. Find ξ ∈ {0,1}^d minimizing ‖A(f-ξ)‖_∞.
Define wⱼ = (2fⱼ - 1)·aⱼ. Finding signs for wⱼ = choosing rounding direction.

Set K = [-t, t]^m. Gaussian measure condition:
  γ_m([-t,t]^m) = (erf(t/√2))^m ≥ 1/2
  Satisfied when t ≥ √(2·ln(2m)) + O(1)

Banaszczyk guarantees: ‖A(f - ξ)‖_∞ ≤ 5t = O(√(log m))

PROOF STRATEGY (Gaussian Random Walk):
1. Define partial sums S_t = Σ_{i=1}^t εᵢ·vᵢ
2. Gaussian potential: Φ_t(K) = γ_d(K - S_t)
3. Two-point inequality (from Prékopa-Leindler):
   max{γ_d(K+v), γ_d(K-v)} ≥ γ_d(K) · e^{-‖v‖²/2}
4. Greedy sign selection: choose εₜ to maximize Φ_t(K)
5. After n steps: Φ_n(K) ≥ γ_d(K) · e^{-n/2} > 0
6. Therefore S_n ∈ 5K (with tighter tracking of Gaussian mass)

Use Mathlib4 imports for:
- Mathlib.Probability.Distributions.Gaussian
- Mathlib.Analysis.Convex.Body
- Mathlib.MeasureTheory.Measure.GaussianMeasure
""",
        },

        "banaszczyk_kv_cache": {
            "technique": "banaszczyk_quant",
            "property": "Banaszczyk rounding for KV cache: dimension-free attention logit bound",
            "prompt": r"""Generate a Lean 4 proof for Banaszczyk-based KV cache quantization.

FORMULATION:
- Queries q₁, ..., q_T ∈ ℝ^d, Keys k₁, ..., k_T ∈ ℝ^d
- Quantize key matrix K ∈ ℝ^{T×d} to K̂ ∈ Δ·ℤ^{T×d}
- Goal: control attention logit distortion

THEOREM (Dimension-Free Logit Bound):
Setting A = normalized query matrix (columns = qᵢ/‖qᵢ‖), and applying Banaszczyk
rounding to each key kⱼ column-by-column:

  max_{i,j} |qᵢᵀ·kⱼ - qᵢᵀ·k̂ⱼ| ≤ O(‖qᵢ‖ · Δ/2^b · √(log T))

where Δ is the dynamic range of key entries and b is the bit width.

KEY INSIGHT: This bound depends only logarithmically on T (number of tokens),
NOT on d (head dimension). This is dimension-free in d.

COMPARISON WITH NAIVE ROUNDING:
Coordinate-wise independent rounding gives:
  |qᵢᵀ·(kⱼ - k̂ⱼ)| ≤ O(‖qᵢ‖ · √d · Δ/2^b)

Banaszczyk improvement factor: √(d/log T)

For typical transformer: d = 128, T = 4096 → improvement ≈ √(128/12) ≈ 3.3x
This means Banaszczyk rounding at 3 bits matches naive rounding at ~5 bits.

PROOF:
1. Each key column kⱼ decomposed as kⱼ = ⌊kⱼ/Δ⌋·Δ + fⱼ·Δ
2. Rounding direction for each coordinate chosen via Banaszczyk signs
3. Constraint matrix A = [q₁/‖q₁‖, ..., q_T/‖q_T‖]ᵀ has columns of norm ≤ 1
4. Apply Banaszczyk: ‖A·(fⱼ - ξⱼ)‖_∞ ≤ O(√(log T))
5. Scale back by Δ/2^b to get the attention logit bound

CONSTRUCTIVE ALGORITHM (Lovett-Meka / Bansal-Dadush-Garg-Lovett):
Runtime: O(n³·d) per vector = poly(T, d) — practical for offline KV cache quantization.

Use Mathlib4 imports for:
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.Analysis.Convex.Body
- Mathlib.Probability.Distributions.Gaussian
""",
        },

        "banaszczyk_vs_naive": {
            "technique": "banaszczyk_quant",
            "property": "Banaszczyk rounding vs coordinate-wise: √(d/log T) improvement",
            "prompt": r"""Generate a Lean 4 proof comparing Banaszczyk vs naive rounding.

THEOREM (Naive Rounding Error):
For independent coordinate-wise rounding of k ∈ ℝ^d to k̂ with step Δ:
  Each coordinate error |kⱼ - k̂ⱼ| ≤ Δ/2
  Attention logit error: |qᵀ(k - k̂)| ≤ Σⱼ |qⱼ|·Δ/2 ≤ ‖q‖₁·Δ/2 ≤ ‖q‖·√d·Δ/2
  (by Cauchy-Schwarz: ‖q‖₁ ≤ √d·‖q‖₂)

THEOREM (Banaszczyk Rounding Error):
With query-aware rounding using Banaszczyk's theorem:
  |qᵢᵀ(k - k̂)| ≤ O(‖qᵢ‖·Δ·√(log T))

THEOREM (Improvement Factor):
  naive_error / banaszczyk_error = Θ(√(d / log T))

PROOF:
1. Naive: worst case is q aligned with the all-ones vector, giving √d factor
2. Banaszczyk: constraint K = [-t,t]^T with t = √(2·log(2T))
   ensures ALL query-key pairs simultaneously bounded by O(√(log T))
3. The ratio √d / √(log T) can be large:
   - d=64, T=2048: ratio ≈ √(64/11) ≈ 2.4x
   - d=128, T=8192: ratio ≈ √(128/13) ≈ 3.1x
   - d=128, T=131072: ratio ≈ √(128/17) ≈ 2.7x

COROLLARY (Bit Savings):
Since quantization error ∝ 2^{-b}, achieving the same error with fewer bits:
  Banaszczyk at b bits ≈ Naive at (b + ½·log₂(d/log T)) bits
  For d=128, T=4096: savings ≈ 1.7 bits

Use Mathlib4 imports for:
- Mathlib.Analysis.InnerProductSpace.Basic
- Mathlib.Analysis.MeanInequalities
""",
        },
    }

    return prompts


def get_proofs_for_technique(technique_name):
    """Get the relevant proof prompts for a winning technique."""
    proof_keys = PROOF_REGISTRY.get(technique_name, [])
    all_prompts = generate_lean_prompts()
    return {k: all_prompts[k] for k in proof_keys if k in all_prompts}


def generate_lean_file(technique_name, metrics, output_dir=None):
    """
    Generate a Lean prompt file for a winning experiment.
    Called by kv-lab when an experiment KEEPs.

    Returns: path to generated prompt file, or None if no proofs apply.
    """
    proofs = get_proofs_for_technique(technique_name)
    if not proofs:
        return None

    if output_dir is None:
        output_dir = PROOFS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{technique_name}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"# Lean 4 Proof Prompts for: {technique_name}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        if metrics:
            f.write(f"Experiment metrics: cosine={metrics.get('cosine_sim', 0):.4f}, ")
            f.write(f"ratio={metrics.get('compression_ratio', 0):.2f}x\n")
        f.write(f"\nPaste each section into Harmonic AI to generate verified Lean 4 proofs.\n")
        f.write(f"Save the resulting .lean files in research/proofs/verified/\n\n")

        for name, info in proofs.items():
            f.write(f"---\n\n")
            f.write(f"## {info['property']}\n\n")
            f.write(f"**Proof ID:** `{name}`\n")
            f.write(f"**Technique:** `{info['technique']}`\n\n")
            f.write(f"### Prompt for Harmonic AI\n\n")
            f.write(f"```\n{info['prompt'].strip()}\n```\n\n")

    return filepath


def print_prompts():
    """Print all Lean prompts for copy-paste into Harmonic AI."""
    prompts = generate_lean_prompts()

    for name, info in prompts.items():
        print(f"\n{'='*70}")
        print(f"  PROOF: {info['property']}")
        print(f"  Technique: {info['technique']}")
        print(f"  ID: {name}")
        print(f"{'='*70}")
        print(info["prompt"])

    print(f"\n{'='*70}")
    print(f"  TECHNIQUE → PROOF MAPPING")
    print(f"{'='*70}")
    for tech, proofs in PROOF_REGISTRY.items():
        print(f"  {tech}: {', '.join(proofs)}")


if __name__ == "__main__":
    print_prompts()
