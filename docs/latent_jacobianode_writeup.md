# Latent-space JacobianODE for partially-observed dynamical systems

**Scope.** This document is a self-contained technical write-up of the extensions
we built on top of the vanilla JacobianODE framework
[Eisen, Ostrow, Chandra, Kozachkov, Miller, Fiete, NeurIPS 2025] in order to apply
Jacobian-based dynamics learning to partially-observed neural recordings (Mary
propofol resting-state LFP). It is intended as a first draft of supplementary
material for the thesis chapter; it should be self-contained enough to be edited
into final supp or handed to a paper-writing pipeline.

**Companion reference.** The base method is described in
`Characterizing_control_bw_interacting_subsystems.pdf` (Sections 3 and Appendix
A–D). We do not re-derive results that already appear there; we cite the
sections and focus on the extensions.

**Notation summary.** Throughout:

- $\mathbf{x}(t) \in \mathbb{R}^{d_{\mathrm{obs}}}$ — observed signal at time $t$ (per-electrode LFP, per-condition normalized).
- $\Delta t$ — fixed sampling interval. Observations are on a uniform grid $\{t_0 + k\Delta t\}$.
- $n_d$ — number of delays (`n_delays`); $\tau$ — delay spacing (`delay_spacing`).
- $w = (n_d - 1)\,\tau + 1$ — total length of the delay-embedding window in samples (`time_window`).
- $\mathbf{y}(t) \in \mathbb{R}^{d_{\mathrm{emb}}}$ with $d_{\mathrm{emb}} = n_d \cdot d_{\mathrm{obs}}$ — delay-embedded vector.
- $\mathbf{z}(t) \in \mathbb{R}^{d_{\mathrm{emb}}}$ — output of the (dimension-preserving) latent encoder.
- $\mathbf{z}_{\mathrm{dyn}} \in \mathbb{R}^{d_{\mathrm{dyn}}}$, $\mathbf{z}_{\mathrm{null}} \in \mathbb{R}^{d_{\mathrm{emb}} - d_{\mathrm{dyn}}}$ — partition of $\mathbf{z}$ into the dynamic and null subspaces.
- $\mathbf{f}: \mathbb{R}^{d_{\mathrm{dyn}}} \to \mathbb{R}^{d_{\mathrm{dyn}}}$ — true latent dynamics (unobserved); $\mathbf{J} = \partial \mathbf{f}/\partial \mathbf{z}_{\mathrm{dyn}}$ is its Jacobian.
- $\hat{\mathbf{J}}^{\theta}$ — neural Jacobian estimator with parameters $\theta$ (an MLP, as in the base paper).
- $\mathbf{c} \in \mathbb{R}^{d_c}$ — per-trial condition vector (e.g., $\pm 1$ for awake / maintenance dose).

---

## 1. Background: what JacobianODE does in observation space

The vanilla JacobianODE method learns a neural function
$\hat{\mathbf{J}}^{\theta}(\mathbf{x})$ that approximates the Jacobian of an
unknown autonomous dynamical system $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$
**directly in observation space**. From an estimated Jacobian along a trajectory
one recovers $\hat{\mathbf{f}}$ via path integration

$$
\hat{\mathbf{f}}(\mathbf{x}(t_f)) - \hat{\mathbf{f}}(\mathbf{x}(t_i))
= \int_{\mathcal{C}} \hat{\mathbf{J}}^{\theta}\, d\mathbf{s},
$$

so that gradients of the trajectory-reconstruction loss flow back through the
Jacobian network alone (no separate $\mathbf{f}$ estimator). The training
objective is

$$
\mathcal{L}(\theta;\mathbf{x}) = \mathcal{L}_{\mathrm{traj}}(\theta;\mathbf{x}) + \lambda_{\mathrm{loop}}\,\mathcal{L}_{\mathrm{loop}}(\theta;\mathbf{x}),
$$

with a generalized teacher-forcing schedule on the rollout
($\tilde{\mathbf{x}}(t+\Delta t)=(1-\alpha)\hat{\mathbf{x}}(t+\Delta t) +
\alpha \mathbf{x}(t+\Delta t)$, App. D.8.1) and a self-supervised loop-closure
constraint $\oint_{\mathcal{C}_{\mathrm{loop}}} \hat{\mathbf{J}}^{\theta}\,d\mathbf{s}=\mathbf{0}$
that enforces the conservativity of each Jacobian row.

The base method assumes **full observation of the state**, i.e. that
$\mathbf{x}$ is a coordinate system on the same manifold as the latent dynamics.
The Discussion of the paper flags exactly the gap we close:

> "A future challenge for JacobianODE models is partially observed dynamics.
> Recent work has identified that it is possible to learn latent embeddings
> that approximately recover the true state from partial observation [113–117].
> Jacobian-based dynamics learning has been performed in a latent space
> [92, 118], however it is unclear whether this translates to accurate Jacobian
> estimation."

The remainder of this document describes the extensions that take JacobianODE
from observation space into a learned latent space, in a way that (a) preserves
the path-integration / loop-closure machinery, (b) preserves attractor topology
between observation and latent space, and (c) admits a clean way to estimate
how many latent dimensions actually carry the dynamics.

---

## 2. Delay embedding

### 2.1 Setup

Each "observation" in our setting is a $d_{\mathrm{obs}}$-channel LFP vector
sampled at $1\,\mathrm{kHz}$. The instantaneous LFP vector is a partial,
low-pass-filtered projection of a far higher-dimensional neural state; a single
$\mathbf{x}(t)$ does not in general coordinatize the underlying attractor.

A standard remedy is **delay embedding**: for a fixed number of delays $n_d$
and delay spacing $\tau\in \mathbb{N}_{>0}$, form

$$
\mathbf{y}(t) =
\Big[\mathbf{x}(t),\ \mathbf{x}(t-\tau \Delta t),\ \mathbf{x}(t-2\tau \Delta t),
\ldots,\ \mathbf{x}(t-(n_d-1)\tau \Delta t)\Big] \in \mathbb{R}^{n_d d_{\mathrm{obs}}}.
$$

### 2.2 Takens' theorem (in words)

For a generic smooth observation map $h:\mathcal{M}\to \mathbb{R}$ on a smooth
compact $d$-dimensional attractor $\mathcal{M}$, Takens' theorem
[Takens 1981] states that with $n_d \geq 2d + 1$ the map
$\phi:\mathbf{x} \mapsto (h(\mathbf{x}),h(F^{-1}\mathbf{x}),\ldots,
h(F^{-(n_d-1)}\mathbf{x}))$
is generically an embedding — a diffeomorphism between $\mathcal{M}$ and its
image $\phi(\mathcal{M})\subset \mathbb{R}^{n_d}$. The vector-valued case
(LFP) only relaxes the bound on $n_d$: more independent channels mean fewer
delays are required to reconstruct the topology.

The practical consequence is that, once delay-embedded, the trajectory
$\{\mathbf{y}(t)\}_{t}$ lies on a manifold $\phi(\mathcal{M})$ that is
**diffeomorphic** to the true attractor $\mathcal{M}$. In particular,
tangent-space dynamics on $\phi(\mathcal{M})$ are related to those on
$\mathcal{M}$ by an invertible linear change of basis (the pushforward of
$\phi$). Jacobian-based analyses on $\phi(\mathcal{M})$ — eigenvalues, Lyapunov
spectra, reachability Gramians up to similarity — are therefore meaningful as
geometric properties of the underlying system, not artifacts of the choice of
observable.

### 2.3 Implementation

We treat $(n_d, \tau)$ as hyperparameters and choose them empirically. The
embedded sequence is constructed by `Tensor.unfold` over the time axis. We
align the encoder, the dynamics loss, and the decoder all on the embedded grid:
the encoder ingests $\mathbf{y}(t)$ and emits $\mathbf{z}(t)$ at the same rate,
so trajectory predictions and loop closures are computed step-for-step in the
delay-embedded coordinate system.

**Practical note: $\tau=1$ vs $\tau>1$ — a tradeoff.** Larger $\tau$ spreads
the embedding window over a longer wall-clock interval, which means each
delay-embedded vector $\mathbf{y}(t)$ carries **more dynamical information**
about the underlying trajectory (a longer history is sampled). However, that
same information has to be **compressed into the same-dimensional latent
space** — the dynamic subspace dimension $d_{\mathrm{dyn}}$ does not scale
with $\tau$, so each encoded latent step must summarize a temporally wider
window. Larger $\tau$ also requires a longer total raw-sample window $w =
(n_d-1)\tau+1$ for each encoded latent step, which increases activation
memory. The right $\tau$ trades off "more information to encode" against
"harder compression and more compute." For 1 kHz LFP filtered to $\leq 80$ Hz,
$\tau=1$ has been competitive or better than $\tau>1$ in our sweeps, suggesting
consecutive samples already span enough information for the dynamics MLP at
this filtering. Slower / less band-limited systems would likely benefit from
$\tau>1$.

---

## 3. Latent encoder: volume-preserving additive coupling flows

### 3.1 Why a diffeomorphism?

The encoder must map $\mathbf{y} \in \mathbb{R}^{d_{\mathrm{emb}}}$ to a latent
$\mathbf{z}$ where dynamics learning happens. Three desiderata pull in the
same direction:

1. **Topology preservation.** The delay embedding has already given us a
   manifold $\phi(\mathcal{M})\subset \mathbb{R}^{d_{\mathrm{emb}}}$ that is
   diffeomorphic to the true attractor. If the encoder is itself a
   diffeomorphism, the latent manifold is diffeomorphic to both — so Jacobian
   spectra and reachability properties remain coordinate-invariant geometric
   quantities, not features of the encoder. The base JacobianODE method relies
   on this implicitly by working in observation space; we need it explicitly
   in latent space.
2. **Dimension preservation.** A diffeomorphism on $\mathbb{R}^{d_{\mathrm{emb}}}$
   keeps the *full* delay-embedding information available, even when only a
   subset of latent dimensions ($\mathbf{z}_{\mathrm{dyn}}$, §4) is asked to
   carry the dynamics. The remaining dimensions can absorb non-dynamic
   structure (off-manifold noise, residual delay-embedding redundancy) without
   destroying any information needed for reconstruction.
3. **Invertibility.** Exact left-invertibility makes the decoder unnecessary
   to encode the trajectory: $\mathbf{z}_{\mathrm{dyn}}$ is recovered from
   $\mathbf{y}$ by a deterministic forward pass, and the decoder is only
   needed to map *predicted* future latents back to observation space for the
   trajectory loss.

We use a stack of **additive (NICE-style) coupling layers**
[Dinh, Krueger, Bengio 2014] interleaved with volume-preserving rotational
mixers. This satisfies all three desiderata by construction.

### 3.2 The additive coupling layer

Split the last dimension into two halves $\mathbf{x} = (\mathbf{x}_a,
\mathbf{x}_b)$ with $\mathbf{x}_a \in \mathbb{R}^{D/2}$,
$\mathbf{x}_b \in \mathbb{R}^{D-D/2}$. The layer is

$$
\mathbf{y}_a = \mathbf{x}_a, \qquad
\mathbf{y}_b = \mathbf{x}_b + \mathbf{m}_\vartheta(\mathbf{x}_a, \mathbf{c}),
$$

where $\mathbf{m}_\vartheta$ is an MLP (the "conditioner") with parameters $\vartheta$ (distinct from the Jacobian MLP's parameters $\theta$).
The conditioner's input is $\mathbf{x}_a$ alone (or $[\mathbf{x}_a;\mathbf{c}]$
when the encoder is conditioned, §6.1). Crucially, $\mathbf{m}_\vartheta$ does not
depend on $\mathbf{x}_b$.

**Jacobian.** Writing the per-sample Jacobian
$\mathbf{J}_{\mathrm{cpl}} = \partial \mathbf{y}/\partial \mathbf{x}$ in
$(a,b)$ blocks:

$$
\mathbf{J}_{\mathrm{cpl}} =
\begin{bmatrix} \mathbf{I}_a & \mathbf{0} \\ \partial \mathbf{m}_\vartheta/\partial \mathbf{x}_a & \mathbf{I}_b \end{bmatrix},
$$

so $\det \mathbf{J}_{\mathrm{cpl}} = 1$ identically. The layer is **volume-preserving**
(unit Jacobian determinant) regardless of the conditioner's parameters, its
nonlinearities, the input value, or the conditioning vector $\mathbf{c}$. No
log-det term is ever computed or backpropagated.

**Exact inverse.** Given $\mathbf{y}$, recover
$\mathbf{x}_a = \mathbf{y}_a$ and
$\mathbf{x}_b = \mathbf{y}_b - \mathbf{m}_\vartheta(\mathbf{y}_a, \mathbf{c})$. This
inverse is closed-form and uses the **same forward pass** of $\mathbf{m}_\vartheta$
— so a stack of $L$ coupling layers can be inverted exactly in $L$ forward
passes (no fixed-point iteration as in NeurIPS-style ResFlow inverses).

### 3.3 Why volume preservation matters

Four independent reasons.

**(a) Topology.** A volume-preserving smooth bijection of $\mathbb{R}^D$ is a
diffeomorphism. So the composition of an arbitrary number of additive coupling
layers (with smooth conditioners) is a diffeomorphism by construction. This
gives us the topology-preservation property requested in §3.1.1 without any
constraint on the conditioner architecture.

**(b) Observation-space noise volume is preserved.** When we inject
observation-space Gaussian noise $\boldsymbol{\eta} \sim \mathcal{N}(0,\sigma^2 \mathbf{I}_{d_{\mathrm{obs}}})$
during training (§7), the induced perturbation in latent space at the
encoder's operating point is approximately $\mathbf{E}_*\boldsymbol{\eta}$,
where $\mathbf{E}_*$ is the encoder's local Jacobian. With a unit-Jacobian
encoder, the latent noise covariance has determinant $\sigma^{2d_{\mathrm{emb}}}$
— the same volume of perturbation that the obs-space noise specified — so the
noise's *informational content* (its differential entropy contribution) is
preserved across the encoder, rather than being amplified or compressed by an
arbitrary scale that the encoder happens to pick up during training. For a
regularizer whose magnitude is set in obs space and whose effect we want to
reason about in latent space, this matters.

**(c) Gradient flow.** For a generic encoder, gradients propagated through
the latent-to-obs decoding path scale by the encoder's local volume change.
With unit Jacobian determinant, $\|\nabla_\mathbf{x}\| \approx \|\nabla_\mathbf{z}\|$
in a *coordinate-aggregate* sense (the determinant pins the product of
singular values to 1). Individual singular values can still diverge from 1
inside an additive coupling layer — they are bounded above and below by
$1 \pm \|\partial \mathbf{m}_\vartheta/\partial \mathbf{x}_a\|$ — but on average the
encoder cannot trivially absorb or amplify gradients by changing scale.
Combined with the near-identity initialization in §3.4, this gives the
training a well-conditioned starting regime.

**(d) MSE losses in obs and latent space are on the same scale.** Because
the encoder preserves the geometric variance of the data (unit Jacobian
determinant means the volume of any neighborhood — and hence the variance
contained in that neighborhood — is invariant under the encoder map), an
MSE loss computed in latent space is numerically comparable in magnitude to
an MSE loss computed in observation space. This is the property that lets
us set the **weights of all three MSE-style losses** — latent prediction,
decoded prediction, and decoded reconstruction (§5) — to 1 without
needing per-term scale tuning. With a non-volume-preserving encoder, those
three losses could live on wildly different scales and the loss weights
would become a non-trivial hyperparameter sweep. The loop-closure loss is
separate scale (it is a squared path integral with units of $f^2 \cdot
\text{length}^2$, not an MSE) and still requires its own weight
$\lambda_{\mathrm{loop}}$.

### 3.4 Near-identity initialization (`zero_init=True`)

We initialize the last linear layer of each conditioner MLP to zero
(`zero_init=True` in `AdditiveCouplingLayer`). With zero output from
$\mathbf{m}_\vartheta$, the layer reduces to $\mathbf{y} = \mathbf{x}$ — the
identity. Stacking $L$ such layers gives the identity map at initialization.
With learnable orthogonal mixers (§3.5) initialized to the identity, the
full encoder is initialized as $\mathbf{z} = \mathbf{y}$.

**Why this is preferred.** Two reasons:

1. *No prior on the latent geometry.* The latent space is not asked to do
   anything fancy at initialization. The PCA-determined dynamic-subspace dim
   $d_{\mathrm{dyn}}$ (§4) is computed on the delay-embedded data directly, so
   at initialization $\mathbf{z}_{\mathrm{dyn}}$ is literally the
   highest-variance PCs of the embedded observations — already a sensible
   starting representation.
2. *Conditioning.* The exact identity map is the well-conditioned limit of the
   coupling stack (all singular values are 1, gradients pass through
   unchanged). Random init is the opposite: a stack of $L \sim 16$ coupling
   layers with arbitrary random conditioners produces an encoder whose
   singular value spectrum is poorly conditioned and varies sharply across
   the input space.

An optional `near_identity_std > 0` perturbs the last conditioner layer with
small Gaussian noise instead of pure zero — this can break the strict
"identity at the start of the first gradient step" degeneracy and helps
gradients flow through the conditioner. In practice we run with
`near_identity_std = 1e-3`, which preserves the spirit of the initialization
while breaking the symmetry.

### 3.5 Cayley orthogonal mixers between coupling layers

Each additive coupling layer leaves $\mathbf{x}_a$ untouched, so a single
layer cannot couple all $d_{\mathrm{emb}}$ dimensions. Interleaving a learnable
*linear, volume-preserving* mixer between layers solves this. We use a
**Cayley parameterization of the orthogonal group**: a learnable skew-symmetric
matrix $\mathbf{A}\in\mathbb{R}^{D\times D}$ ($\mathbf{A}^\top = -\mathbf{A}$)
gives the orthogonal matrix

$$
\mathbf{Q} = (\mathbf{I}-\mathbf{A})(\mathbf{I}+\mathbf{A})^{-1},
$$

which is in $\mathrm{SO}(D)$ for any $\mathbf{A}$ (so $\det \mathbf{Q}=+1$,
not just $\pm 1$). Initializing $\mathbf{A}=\mathbf{0}$ gives
$\mathbf{Q}=\mathbf{I}$, matching the rest of the near-identity
initialization. Compared to a fixed random permutation (the standard NICE
choice), the Cayley mixer is *learnable* and lets the optimizer adjust which
dimensions get coupled at each layer — empirically this lets us use fewer
coupling layers than a permutation-only stack for the same expressivity.

We additionally use `final_perm_identity=True` in all production runs. With
Cayley orthogonal mixers initialized to the identity (skew-symmetric matrix
$\mathbf{A}=\mathbf{0}$ gives $\mathbf{Q}=\mathbf{I}$), the composition of
all inter-layer mixers is the identity at init; `final_perm_identity` then
appends a *fixed* permutation at the end that is itself the identity, so the
**full encoder is exactly the identity map at initialization**. Combined with
delay embedding's convention of placing the most recent timestep first,
this means $\mathbf{z}_{\mathrm{dyn}}$ at init contains the first
$d_{\mathrm{dyn}}$ entries of the delay-embedded vector $\mathbf{y}$ — i.e.,
the most recent observation slice. After training, the Cayley layers learn
non-identity rotations and the dynamic subspace becomes a learned mixture of
the delay-embedding axes. The role of `final_perm_identity` is therefore
purely an *initialization* device: it gives a deterministic, sensible
starting representation regardless of how the trainable mixers happen to
initialize.

### 3.6 Per-area direct-sum encoder (optional)

When the observation vector has a natural partition into $K$ groups (e.g.
electrodes in different brain areas), the encoder can be built as a
**direct sum** of $K$ per-area coupling stacks:

$$
\mathbf{z}^{(k)} = \mathcal{E}_k(\mathbf{y}^{(k)},\mathbf{c}),\qquad k=1,\ldots,K,
$$

with $\mathbf{z} = (\mathbf{z}^{(1)},\ldots,\mathbf{z}^{(K)})$. This is exposed
as `DirectSumCouplingEncoder`. Each per-area block has its own coupling stack,
its own per-area PCA-based dimension assignment (§4.1), and is independently
volume-preserving. The full encoder remains a diffeomorphism (composition of
direct-sum block diffeomorphisms is itself a diffeomorphism). For analyses
that ask block-Jacobian questions ("how does area B's state push area A?"),
this structure preserves the block partition straight through to latent space.

The Mary propofol runs reported here use a *single* encoder rather than a
direct-sum one (there is no natural area partition for the LFP setup), but
the same coupling-flow machinery underlies both.

---

## 4. Dynamic / null subspace partition

The encoder is dimension-preserving:
$\mathbf{z}\in\mathbb{R}^{d_{\mathrm{emb}}}$. We do not, however, ask the
dynamics network to model all $d_{\mathrm{emb}}$ dimensions — for a chaotic
attractor of low intrinsic dimensionality embedded in a high-$D$ delay
space, most of the embedding dimensions are redundant. We split

$$
\mathbf{z} \;=\; (\mathbf{z}_{\mathrm{dyn}},\, \mathbf{z}_{\mathrm{null}}),
\qquad \mathbf{z}_{\mathrm{dyn}}\in\mathbb{R}^{d_{\mathrm{dyn}}},
\quad \mathbf{z}_{\mathrm{null}}\in\mathbb{R}^{d_{\mathrm{emb}}-d_{\mathrm{dyn}}},
$$

and ask the Jacobian / dynamics machinery to act *only* on
$\mathbf{z}_{\mathrm{dyn}}$. The null subspace is unconstrained except that its
contribution to observation-space reconstruction is suppressed to zero
(§4.3), forcing the encoder to put all reconstructable information into
$\mathbf{z}_{\mathrm{dyn}}$.

### 4.1 Choice of $d_{\mathrm{dyn}}$: PCA upper bound on intrinsic dimensionality

We pick $d_{\mathrm{dyn}}$ automatically from the training data, before any
encoder is trained, using linear PCA on the delay-embedded sequences:

$$
d_{\mathrm{dyn}} \;=\; \min\Big\{k : \sum_{i=1}^{k}\frac{\lambda_i}{\sum_j \lambda_j} \geq \alpha\Big\},
$$

where $\{\lambda_i\}$ are the eigenvalues of the empirical covariance of the
training $\mathbf{y}(t)$, sorted descending, and $\alpha$ is a user-set
variance threshold (`n_target_var_threshold`). In our Mary runs we use
$\alpha \in \{0.99, 0.95\}$ as the primary sweep dimension.

**Why this is a (reasonable) upper bound on intrinsic dimensionality.** Let
$\mathcal{M}\subset\mathbb{R}^{d_{\mathrm{emb}}}$ be the true latent
attractor of intrinsic dimension $d^\star$. Its affine hull has dimension at
least $d^\star$, so for sufficiently dense sampling the rank of the empirical
covariance is at least $d^\star$. As $\alpha \to 1$, the PCA dimension
$d_{\mathrm{dyn}}^{(\alpha)} \to \mathrm{rank}(\mathrm{Cov})\geq d^\star$.
For practical thresholds $\alpha<1$, this is **not a strict theorem**: a
manifold with axes of very small but nonzero curvature/extent can have a true
intrinsic dimension that exceeds the PCA-$\alpha$ dimension. We use the
threshold heuristically — $\alpha=0.99$ is a "safe upper bound on the
dimensions we can hope to model" rather than "the exact intrinsic dim" —
and treat it as a hyperparameter that controls a trade-off between latent
expressiveness and model size.

**Empirical findings.** In the Mary propofol resting-state work, the relevant
range of $d_{\mathrm{dyn}}$ at $\alpha=0.99$ on delay-embedded LFP is a few
hundred dimensions, and we see monotone degradation as $\alpha$ drops to
$0.95$, $0.90$, $0.75$, $0.50$ — both trajectory val loss and
one-step MASE worsen approximately monotonically with smaller $\alpha$,
consistent with $\alpha=0.99$ being closer to the true intrinsic dimensionality
than the smaller thresholds. We do not (currently) have a clean theoretical
bound that pins $\alpha^\star$.

### 4.2 Dynamics losses act only on $\mathbf{z}_{\mathrm{dyn}}$

This is a strict architectural rule, not a soft penalty. Concretely:

1. The encoder produces $\mathbf{z} = (\mathbf{z}_{\mathrm{dyn}},
   \mathbf{z}_{\mathrm{null}})$ via the standard slice
   $\mathbf{z}_{\mathrm{dyn}} = \mathbf{z}[:d_{\mathrm{dyn}}]$,
   $\mathbf{z}_{\mathrm{null}} = \mathbf{z}[d_{\mathrm{dyn}}:]$.
2. The neural Jacobian $\hat{\mathbf{J}}^{\theta}$ maps
   $\mathbb{R}^{d_{\mathrm{dyn}}} \to \mathbb{R}^{d_{\mathrm{dyn}}\times d_{\mathrm{dyn}}}$
   (not $\mathbb{R}^{d_{\mathrm{emb}}\times d_{\mathrm{emb}}}$). Path-integration,
   ODE integration, and teacher forcing all happen in
   $\mathbf{z}_{\mathrm{dyn}}$ space.
3. The loop-closure loss $\mathcal{L}_{\mathrm{loop}}$ is computed on loops in
   $\mathbf{z}_{\mathrm{dyn}}$ space — i.e., contains no information about
   $\mathbf{z}_{\mathrm{null}}$.
4. The trajectory prediction loss decodes $\mathbf{z}_{\mathrm{dyn}}^{\mathrm{pred}}$
   back to observation space (§4.3) and compares against the obs-space target.

The null subspace receives gradients **only** from the reconstruction loss
(§4.3), which pulls $\mathbf{z}_{\mathrm{null}}$ toward values that, *combined
with $\mathbf{z}_{\mathrm{dyn}}$ set to its true encoded value*, decode back
to the observation. But because the decoder also sees $\mathbf{z}_{\mathrm{null}}$
set to zero in the dynamics-prediction path (§4.3), the encoder is
incentivized to push all reconstruction-relevant information into
$\mathbf{z}_{\mathrm{dyn}}$ and to leave $\mathbf{z}_{\mathrm{null}}$
unconstrained.

### 4.3 Zero-padded reconstruction in the latent dimension

The decoder is built to consume the **full** $d_{\mathrm{emb}}$-dimensional
latent. When we decode a *prediction* — produced by the Jacobian dynamics, in
$\mathbf{z}_{\mathrm{dyn}}$ space — we must give the decoder a full-dim input.
We do this by **zero-padding the null subspace**:

$$
\mathrm{decode}(\mathbf{z}_{\mathrm{dyn}}^{\mathrm{pred}}) \;\triangleq\;
\mathcal{D}\big(\,[\,\mathbf{z}_{\mathrm{dyn}}^{\mathrm{pred}};\;\mathbf{0}_{d_{\mathrm{emb}}-d_{\mathrm{dyn}}}\,]\,\big).
$$

This is the same operation applied in the reconstruction loss path
(`_reconstruction_loss` at `latent_jacobian.py:1183-1185`): we encode the
input, take only $\mathbf{z}_{\mathrm{dyn}}$ from the result, zero-pad the
null subspace, then decode. The decoder is **never** asked to use a non-zero
null subspace at prediction time.

This has two consequences worth stating clearly:

- *Information bottleneck for dynamics.* Anything that the dynamics needs to
  forecast must be visible at the decoder's input with
  $\mathbf{z}_{\mathrm{null}}=\mathbf{0}$. The encoder is *implicitly*
  trained — by the chain of (i) trajectory loss penalizes obs-space rollout
  error, (ii) rollouts use zero-padded decodes, (iii) gradients propagate
  through both encoder and decoder — to place all dynamically-relevant
  information in $\mathbf{z}_{\mathrm{dyn}}$.
- *Soft, not hard, separation of dynamic and null.* The architecture does not
  *force* $\mathbf{z}_{\mathrm{null}}$ to be zero on the data manifold; the
  encoder may put noise / off-manifold residual / per-trial nuisance variation
  there, where it harmlessly does not affect predicted decodes. The
  reconstruction loss tries to use $\mathbf{z}_{\mathrm{null}}$ to improve
  encoder-only reconstruction fidelity, but in the zero-padded decoding path
  (which is what the dynamics ever sees) the null subspace contributes nothing.

---

## 5. Loss functions

We minimize **four** loss terms: three MSE-style losses (latent prediction,
decoded prediction, decoded reconstruction) plus the loop-closure loss
inherited from the base paper. The three MSE-style losses all live on
approximately the same numerical scale — a consequence of the encoder's
volume preservation (§3.3 (d)) — so we weight them all at $1$. The
loop-closure loss is a squared path integral on a different scale and gets
its own swept weight $\lambda_{\mathrm{loop}}$. The total objective is

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{lat\text{-}pred}} + \mathcal{L}_{\mathrm{dec\text{-}pred}} + \mathcal{L}_{\mathrm{dec\text{-}rec}} + \lambda_{\mathrm{loop}}\,\mathcal{L}_{\mathrm{loop}}.
$$

Throughout this section we use $\mathbf{x}(t)$ for obs-space samples,
$\mathbf{y}(t)$ for the delay-embedded vector, and
$\mathbf{z}(t) = \mathcal{E}(\mathbf{y}(t),\mathbf{c})$ for the encoded
latent (with dynamic / null split $\mathbf{z} = (\mathbf{z}_{\mathrm{dyn}},
\mathbf{z}_{\mathrm{null}})$). We write
$[\mathbf{z}_{\mathrm{dyn}};\mathbf{0}]$ for the zero-padded full-dim vector
fed into the decoder (§4.3). $\hat{\mathbf{f}}$ denotes the path-integrated
ODE rollout of $\hat{\mathbf{J}}^{\theta}$ over one step
(via JacobianODEint), so $\hat{\mathbf{f}}(\mathbf{z}_{\mathrm{dyn}}(t))$
approximates $\mathbf{z}_{\mathrm{dyn}}(t+\Delta t)$.

### 5.1 Latent prediction loss $\mathcal{L}_{\mathrm{lat\text{-}pred}}$

Compares the dynamics MLP's rollout against the encoded ground truth,
**directly in latent space** (no decoder):

$$
\mathcal{L}_{\mathrm{lat\text{-}pred}}
= \mathbb{E}_t\Big[\big\| \hat{\mathbf{f}}(\mathbf{z}_{\mathrm{dyn}}(t))
- \mathrm{sg}\big[\mathbf{z}_{\mathrm{dyn}}(t+\Delta t)\big] \big\|^2 \Big].
$$

The encoded target $\mathbf{z}_{\mathrm{dyn}}(t+\Delta t)$ is stop-grad'd
(`sg`) when training-time observation noise is injected (§7), so that
gradients from $\mathcal{L}_{\mathrm{lat\text{-}pred}}$ flow only through
the noisy encoder + dynamics path and not through the target itself —
the standard "noise2clean" target convention. Without obs noise the
target is just $\mathbf{z}_{\mathrm{dyn}}(t+\Delta t)$ from the same
forward pass.

This term is the most direct measurement of whether the dynamics model is
accurate in latent space, independent of the encoder–decoder reconstruction
floor. It is what makes the dynamics network "see" the latent geometry
the encoder is currently producing.

### 5.2 Decoded prediction loss $\mathcal{L}_{\mathrm{dec\text{-}pred}}$

Compares the **decoded** rollout against the raw observation:

$$
\mathcal{L}_{\mathrm{dec\text{-}pred}}
= \mathbb{E}_t\Big[\big\|
\mathcal{D}\big([\hat{\mathbf{f}}(\mathbf{z}_{\mathrm{dyn}}(t));\mathbf{0}],\mathbf{c}\big)
- \mathbf{x}(t+\Delta t)
\big\|^2 \Big].
$$

The expectation is over rollout steps as well as time-points; in practice
sub-windows of length $w_{\mathrm{init}} + n_{\mathrm{pred}}$ are extracted,
JacobianODEint integrates the latent through `prediction_steps` ahead with
the generalized teacher-forcing schedule (App. D.8.1 of the base paper),
and each predicted latent is zero-padded and decoded to compare against the
corresponding raw observation.

The loss is evaluated with the **most-recent reconstruction mode** (§5.5):
the decoder outputs a flattened
delay-embedding-shaped vector and we compare only the most-recent slice
$\hat{\mathbf{x}}(t+k\Delta t)$ against $\mathbf{x}(t+k\Delta t)$.

This term is the obs-space forecast objective: it forces the encoder + dynamics +
decoder pipeline to produce predictions that are correct in observation
units. It is the loss that the user-monitored val/traj_loss tracks.

### 5.3 Decoded reconstruction loss $\mathcal{L}_{\mathrm{dec\text{-}rec}}$

The encoder–decoder autoencoder loss, applied at every encoded latent step
in the **zero-padded null** decoding path (§4.3):

$$
\mathcal{L}_{\mathrm{dec\text{-}rec}}
= \mathbb{E}_{t}\Big[ \big\| \mathcal{D}\big([\mathbf{z}_{\mathrm{dyn}}(t);\mathbf{0}],\mathbf{c}\big) - \mathbf{y}_{\mathrm{rec}}(t)\big\|^2 \Big].
$$

where $\mathbf{y}_{\mathrm{rec}}(t)$ is the reconstruction target — by
default the full delay-embedded window in `uniform` mode (§5.5), but
configurable per the same reconstruction-mode mechanism as §5.2.

This loss is the *only* place where the null subspace
$\mathbf{z}_{\mathrm{null}}$ would have any influence on the optimization
target — except that we explicitly zero it before decoding, so in effect
the reconstruction loss trains the encoder + decoder *plus* the implicit
"use only $\mathbf{z}_{\mathrm{dyn}}$ for reconstruction" channel. It is
the term that anchors the encoder and decoder to obs-space units.

### 5.4 Loop-closure loss $\mathcal{L}_{\mathrm{loop}}$

Inherited verbatim from the base JacobianODE paper (paper §3.3; App. A.3)
but applied to $\mathbf{z}_{\mathrm{dyn}}$ rather than to $\mathbf{x}$.
Loops are constructed by concatenating line segments between randomly
selected in-batch $\mathbf{z}_{\mathrm{dyn}}$ pairs, and the loss penalizes
the squared $\ell_2$ norm of the path integral of $\hat{\mathbf{J}}^{\theta}$
around each loop:

$$
\mathcal{L}_{\mathrm{loop}}(\theta;\mathbf{x})
= \Big\| \oint_{\mathcal{C}_{\mathrm{loop}}} \hat{\mathbf{J}}^{\theta}\, d\mathbf{s} \Big\|_2^2.
$$

This enforces that each row of $\hat{\mathbf{J}}^{\theta}$ is a conservative
vector field — equivalently, that $\hat{\mathbf{J}}^{\theta}$ is the
Jacobian of *some* latent vector field. This is a structural property of
any honest Jacobian; the loop-closure regularizer pushes the learned MLP
toward respecting it.

**Why this transfers cleanly to latent space.** The true latent Jacobian
$\mathbf{J}_{\mathrm{lat}}(\mathbf{z})$ is, by construction, the Jacobian
of the pushforward vector field $\mathbf{f}_{\mathrm{lat}} = D\phi \cdot
\mathbf{f}\circ\phi^{-1}$, where $\phi$ is the encoder. So its rows are
gradients of components of $\mathbf{f}_{\mathrm{lat}}$ tautologically, and
loop closure is satisfied automatically by the **true** answer
$\mathbf{J}_{\mathrm{lat}}$ for any choice of $\phi$. Loop closure in
latent space targets exactly the same structural property as in obs space
— "be a valid Jacobian" — and is well-motivated for any smooth diffeomorphic
encoder.

### 5.5 Reconstruction modes for the obs-space targets

The decoded prediction loss (§5.2) and decoded reconstruction loss (§5.3)
both operate on the **delay-embedded** observation space, so the decoder
output naturally has shape $(w, d_{\mathrm{obs}})$ — a stack of $w$
timeslices from oldest to newest. We use one of three weightings to
collapse this into a scalar MSE:

- **`uniform`**: equal weight on all $w$ slices. The decoder is asked to
  reproduce the entire delay-embedded window.
- **`most_recent`**: weight only on the most recent slice (first
  $d_{\mathrm{obs}}$ entries of the flattened output). The decoder is asked
  to reproduce only the current timestep $\hat{\mathbf{x}}(t)$.
- **`harmonic`**: $1/k$ decaying weights. Available; rarely used.

**Production setting**:

- $\mathcal{L}_{\mathrm{dec\text{-}rec}}$ uses `uniform`: we want the
  autoencoder to faithfully reproduce the whole window.
- $\mathcal{L}_{\mathrm{dec\text{-}pred}}$ uses `most_recent`: prediction is
  a forecast task, and at rollout step $k$ the quantity of interest is
  $\hat{\mathbf{x}}(t+k\Delta t)$ vs $\mathbf{x}(t+k\Delta t)$ on the live
  frame. Averaging over the $w$ slices of the predicted window would
  dilute that forecast signal by $1/w$ and would mix in slices that are
  "in the past relative to the rollout's anchor" — easy to reconstruct from
  teacher-forced or recent input — with the genuinely-forecasted future
  slice. Validation hardcodes `most_recent` for the same reason: we want
  monitored val loss to reflect forward-rollout fidelity on the live frame.

This mode-switching ensures train-time gradient and val-time monitored loss
both diagnose the forecastable quantity — an inconsistency in early
versions of the code that produced val losses numerically smaller than
train losses by a factor of $w$ for the same model.

(A related but distinct option, `decode_only_recent=True`, builds the
decoder to only produce the most recent slice as its native output rather
than slicing it from a full-window output. We use the full-window output
plus most-recent slicing at the loss; the two are roughly equivalent but
the slicing-at-loss path lets the same model also be evaluated under
`uniform` mode for diagnostics without retraining.)

### 5.6 Gradient flow summary

| Component                              | $\nabla \to \hat{\mathbf{J}}^{\theta}$ | $\nabla \to \mathcal{E}$                | $\nabla \to \mathcal{D}$ |
|---------------------------------------|----------------------------------------|-----------------------------------------|--------------------------|
| $\mathcal{L}_{\mathrm{lat\text{-}pred}}$    | yes                                    | yes (the noisy-encoder pass; target is stop-grad'd)   | no |
| $\mathcal{L}_{\mathrm{dec\text{-}pred}}$    | yes                                    | yes (through encoded init + TF steps)   | yes |
| $\mathcal{L}_{\mathrm{dec\text{-}rec}}$     | no                                     | yes (only on $\mathbf{z}_{\mathrm{dyn}}$ — null is fed $\mathbf{0}$) | yes |
| $\mathcal{L}_{\mathrm{loop}}$               | yes                                    | yes (loops built from gradient-bearing $\mathbf{z}_{\mathrm{dyn}}$) | no |

The only `no_grad` in this entire pipeline is on the encoder pass used to
compute Jacobian norms for the *teacher-forcing α update*
(`latent_jacobian.py:1432-1437`); that no_grad is scoped to the α schedule
and does not affect any of the four loss terms above. The encoder gradient
from $\mathcal{L}_{\mathrm{loop}}$ is *not* a problem (see §5.4, "Why this
transfers cleanly to latent space"): the encoder cannot warp toward
"easier-LC-but-wrong" because the true latent Jacobian is automatically
LC-satisfying for any encoder. The LC encoder gradient is just a standard
regularizer effect, pushing the joint (encoder + Jacobian MLP) toward
representations on which the MLP can globally express a conservative
matrix field.

---

## 6. Conditioning and per-condition data normalization

### 6.1 Condition vector $\mathbf{c}$

Each trial carries a one-dimensional condition vector
$\mathbf{c}\in\mathbb{R}^{d_c}$. In Mary propofol resting-state runs, $d_c=1$
with values $\mathbf{c}=-1$ for awake trials and $\mathbf{c}=+1$ for
maintenance-dose anesthetized trials. The condition vector is **broadcast**
into multiple places in the network:

- Each coupling layer's conditioner sees $[\mathbf{x}_a;\mathbf{c}]$ as input
  (when the encoder is built with `condition_dim>0`). This is a
  *per-coupling-layer* condition input, not a global gate.
- The Jacobian MLP $\hat{\mathbf{J}}^{\theta}(\mathbf{z}_{\mathrm{dyn}},
  \mathbf{c})$ also takes $\mathbf{c}$ as an explicit input. So the *learned
  Jacobian field* is condition-dependent: awake and anesthetized trajectories
  have separate (but parameter-shared) Jacobian functions of latent state.

In practice this means the model can learn one latent geometry shared
across awake/anesthesia (the encoder is condition-aware but not
condition-partitioned) while expressing the **two different dynamical regimes**
on that shared geometry through condition-dependent Jacobians.

### 6.2 Per-condition normalization

We normalize the observation tensor **per condition** before delay-embedding.
For each electrode and each condition value $c\in\{-1,+1\}$ we compute mean
and std across all timepoints in trials with that condition, and z-score:

$$
\tilde{\mathbf{x}}_e(t) \;=\; \frac{\mathbf{x}_e(t) - \mu_{e,c}}{\sigma_{e,c}}.
$$

This removes the trivial cross-condition mean shift (anesthesia and awake
LFPs have different DC offsets and overall power per electrode) from the
modeling task, so that the dynamics MLP sees similar-scale inputs in both
conditions and must explain genuine *dynamical* differences rather than
trivial scale differences.

A potential failure mode of this normalization is that we lose information
about the absolute *amplitude* difference between conditions. We accept this
trade-off because the conditioning vector $\mathbf{c}$ tells the dynamics
network which condition the data came from anyway, and amplitude alone is not
a feature of dynamics that the JacobianODE machinery is well-suited to
characterize.

---

## 7. Training-time observation noise

A controlled amount of obs-space Gaussian noise is injected at the encoder
input during training:

$$
\mathbf{y}_{\mathrm{noisy}}(t) = \mathbf{y}(t) + \boldsymbol{\eta}(t),
\qquad \boldsymbol{\eta}(t) \sim \mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I}).
$$

There is a subtle point: when noise is injected, the encoded
$\mathbf{z}_{\mathrm{noisy}}=\mathcal{E}(\mathbf{y}_{\mathrm{noisy}})$ is a
noisy version of the true latent. Using it directly as the *target* in the
latent-prediction loss (§5.1) would pull $\mathbf{f}$ toward random latent
points. So we maintain a parallel **clean-encoder pass** under
`torch.no_grad()` and use
$\mathbf{z}_{\mathrm{clean}}=\mathcal{E}(\mathbf{y})$ as the target. Gradients
from $\mathcal{L}_{\mathrm{lat\text{-}pred}}$ flow only through the noisy
encoder + dynamics path. This is the standard "noise2clean" target
convention.

The role of obs noise here is dynamical-system regularization: the
JacobianODE paper observes that 5% training noise *improves* Jacobian
estimation in the simulated benchmarks (Table 1), interpreted as noise
encouraging the model to explore tangent-space dynamics around the observed
trajectory. We are currently sweeping $\sigma \in \{0, 0.01, 0.05\}$ as a
post-hoc question (data is per-condition-normalized, so these correspond to
$0$, $1\%$, $5\%$ of unit variance).

---

## 8. Evaluation metrics

In addition to the loss values themselves, validation logs three Mean Absolute
Scaled Error (MASE) variants that decouple different sources of error.

Define the obs-space targets and decoded predictions:

- $\mathbf{x}_{t+1}$: true next observation.
- $\hat{\mathbf{x}}_{t+1} = \mathcal{D}([\hat{\mathbf{f}}(\mathbf{z}_{\mathrm{dyn}}(t));\mathbf{0}])$: the model's one-step forecast in obs space.
- $\mathbf{x}_t$: persistence (no-change) forecast.
- $\mathcal{D}(\mathbf{z}_{\mathrm{dyn}}(t+1))$: decoded encoded ground truth — the "best the decoder could do" if dynamics were perfect.

Then:

**Standard one-step MASE** (raw forecast quality):

$$
\mathrm{MASE}_{\mathrm{1step}} = \frac{|\hat{\mathbf{x}}_{t+1} - \mathbf{x}_{t+1}|}{|\mathbf{x}_t - \mathbf{x}_{t+1}|}.
$$

This measures the *raw* obs-space forecast against the persistence baseline.
MASE $<1$ means the model beats persistence; MASE $\gg 1$ is bad.

**Dynamics-only MASE** (`dynamics_only_one_step_mase`) — isolates the
contribution of $\hat{\mathbf{f}}$:

$$
\mathrm{MASE}_{\mathrm{dyn}} = \frac{|\mathcal{D}(\hat{\mathbf{f}}(\mathbf{z}_{\mathrm{dyn}}(t))) - \mathcal{D}(\mathbf{z}_{\mathrm{dyn}}(t+1))|}{|\mathbf{x}_t - \mathbf{x}_{t+1}|}.
$$

The numerator compares two decoded latents — one produced by dynamics, one
the ground-truth decoded latent — so it measures how well dynamics alone
captures latent-space displacement, *independent of* the encoder–decoder
reconstruction floor.

**Decoder-corrected MASE** (`decoder_corrected_one_step_mase`) — compares
model forecast against the decoder's no-dynamics baseline:

$$
\mathrm{MASE}_{\mathrm{dec\text{-}corr}} = \frac{|\hat{\mathbf{x}}_{t+1} - \mathbf{x}_{t+1}|}{|\mathcal{D}(\mathbf{z}_{\mathrm{dyn}}(t)) - \mathbf{x}_{t+1}|}.
$$

The denominator is **decoded persistence**: what you would predict by
encoding $\mathbf{x}_t$, decoding $\mathbf{z}_{\mathrm{dyn}}(t)$, and reporting
that as $\hat{\mathbf{x}}_{t+1}$. This subtracts off both the persistence
baseline and any encoder–decoder error, isolating the contribution of the
dynamics model to forecasting quality.

The three together let us read off whether a high obs-space MASE is
"dynamics' fault," "decoder's fault," or "both" — important because the
latent partition (§4) explicitly separates these two failure modes
architecturally.

---

## 9. Training protocol (current sweeps)

For reference, the production training recipe used in the Mary propofol
resting-state sweeps:

- **Encoder.** `latent_direct_sum_coupling` with `hidden_dim=1024`,
  `n_coupling_layers=16`, `use_cayley_perms=True`, `final_perm_identity=True`,
  `zero_init=True`, `near_identity_std=1e-3`.
- **Dynamics MLP** ($\hat{\mathbf{J}}^{\theta}$). 4 hidden layers,
  widths `(2048, 2048, 4096, 4096)`. Input dim $d_{\mathrm{dyn}} + d_c$,
  output dim $d_{\mathrm{dyn}}^2$.
- **Embedding.** $\tau=1$ (sample step), $n_d$ swept over $\{5, 10, 15, 20\}$
  for each $\alpha$.
- **Latent dim.** PCA-based: $\alpha\in\{0.99,0.95\}$.
- **Reconstruction.** `reconstruction_mode='uniform'`, trajectory loss uses
  `most_recent`. `decode_only_recent=False`.
- **Loss weights.** Latent-prediction, decoded-prediction, and
  decoded-reconstruction weights all set to $1$ (justified by the
  same-scale argument in §3.3 (d)). $\lambda_{\mathrm{loop}}$ swept
  (currently a $7$-point geometric sweep $\{0, 10^{-6}, 10^{-5}, 10^{-4},
  10^{-3}, 10^{-2}, 10^{-1}\}$ for the upcoming LC ablation).
- **Trajectory.** `seq_length=25`, `seq_spacing=15`, `prediction_steps=10`,
  `traj_init_steps=15`.
- **Optimizer & scheduler.** AdamW + the `TeacherForcingLRScheduler` adopted
  from the base JacobianODE paper.
- **Obs noise.** Currently swept; production default $0$ pending the
  obs-noise pilot.
- **Early stopping.** Patience 5 on `mean val loss`, min epochs 15.
- **Filtering.** LFP low-pass filtered at 80 Hz before delay embedding.
- **Normalization.** Per-condition z-score (§6.2).
- **Conditioning.** $d_c=1$, $\mathbf{c}\in\{-1,+1\}$ (awake / maintenance dose).

---

## 10. Summary of departures from the vanilla JacobianODE paper

For ease of reference when comparing this work to the published paper:

| Component | Vanilla JacobianODE | This work |
|-----------|--------------------|-----------|
| Operating space | Observation $\mathbf{x}$ | Latent $\mathbf{z}_{\mathrm{dyn}}$ |
| Input pre-processing | None (state assumed observed) | Delay embedding ($n_d$, $\tau$); per-condition z-score |
| Encoder | None | Stack of additive coupling flows with learnable Cayley orthogonal mixers; volume-preserving; near-identity init |
| Latent partition | n/a | $\mathbf{z}=(\mathbf{z}_{\mathrm{dyn}},\mathbf{z}_{\mathrm{null}})$ with $d_{\mathrm{dyn}}$ from PCA at threshold $\alpha\in\{0.95,0.99\}$ |
| Decoder | n/a | Maps full $\mathbf{z}$ (with $\mathbf{z}_{\mathrm{null}}=\mathbf{0}$ in dynamics path) back to delay-embedded observation; loss is `uniform` (recon) / `most_recent` (trajectory) |
| Jacobian model | $\hat{\mathbf{J}}^{\theta}(\mathbf{x})$ | $\hat{\mathbf{J}}^{\theta}(\mathbf{z}_{\mathrm{dyn}}, \mathbf{c})$, condition-aware |
| Loss | $\mathcal{L}_{\mathrm{traj}}+\lambda_{\mathrm{loop}}\mathcal{L}_{\mathrm{loop}}$ | Four terms: latent-pred + decoded-pred + decoded-recon (all weight $1$, on the same MSE scale by §3.3 (d)) + $\lambda_{\mathrm{loop}}\mathcal{L}_{\mathrm{loop}}$ |
| Validation metrics | MSE / R² / Frobenius-J error | Trajectory val loss + three MASE variants (raw, dynamics-only, decoder-corrected); fast-eigenvalue fraction; per-condition Lyapunov / reachability diagnostics |
| Training noise | Fixed at $\{1\%,5\%,10\%\}$ in the paper experiments | Per-sweep, currently being studied |

The path-integration ODE solver, teacher-forcing schedule, loop-closure
construction, and Jacobian MLP backbone are all inherited unchanged from the
base paper.

---

## 11. Open questions / things flagged for the chapter narrative

A short list of things that are not yet locked down and where the chapter
discussion should be honest about uncertainty:

1. **PCA threshold is heuristic, not theorem.** §4.1 argues $\alpha\to 1$
   gives an upper bound on intrinsic dim, but the practical $\alpha=0.99$
   choice is empirical. We have not derived a finite-$\alpha$ bound.
2. **Most-recent reconstruction for the decoded prediction loss is an
   empirical choice.** The argument in §5.5 (avoiding $1/w$ dilution of
   forecast signal) is plausible but not proven optimal. Ablating against
   `uniform` and `harmonic` on a small grid would either confirm or refute
   it.
3. **Encoder gradients from loop closure: standard regularizer, no
   pathology.** The pushforward Jacobian under any smooth diffeomorphism is
   row-conservative by construction (it remains the Jacobian of an honest
   vector field), so the loop-closure target is automatically satisfied by
   the true latent Jacobian regardless of which encoder $\phi$ we pick.
   Encoder gradients from $\mathcal{L}_{\mathrm{loop}}$ therefore *cannot*
   warp the encoder toward an "easier but wrong" target — the right target
   is LC-satisfying for every encoder. The remaining question is whether
   those gradients act as a useful regularizer (encouraging the encoder to
   produce latent geometries on which the MLP can globally approximate a
   conservative matrix field) or as marginal noise. An ablation that runs
   $\mathcal{L}_{\mathrm{loop}}$ under `no_grad` on the encoder would
   isolate this; we have not run it.
4. **Encoder warmup.** The codebase supports a "encoder warmup" phase
   (`encoder_warmup_epochs>0`) where reconstruction and KL losses are
   trained first, before dynamics losses are activated. Current Mary runs
   use `encoder_warmup_epochs=0` (everything trained jointly from epoch 1).
   This is worth flagging if any final-version runs use a non-zero warmup.

---

*End of write-up.* The chapter / supplementary draft can pull from any of
§1–§10 as needed; §11 is private to-do for the narrative, not for inclusion.
