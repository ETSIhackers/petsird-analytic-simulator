# Randoms estimation from delayed coincidences

This note describes the randoms model and the estimators implemented in
[`python/randoms.py`](../python/randoms.py) and used by
`02_reconstruct_petsird.py --randoms_from_delayeds {ml,casey}`.

## 1. Randoms model

Consider two detection bins $a$ and $b$ in coincidence with singles rates $S_a(t)$ and
$S_b(t)$, and a coincidence window of total width $2\tau$. If the singles are independent
Poisson processes, a single in $b$ falls into the window of a single in $a$ with
probability $2\tau S_b$, so the rate of random coincidences is $2\tau S_a S_b$. Integrated
over the acquisition, the expected number of randoms is

$$
E[R_{ab}] = 2\tau \int S_a(t)\, S_b(t)\, dt .
$$

If all singles rates share the same time course, $S_a(t) = S_a\, g(t)$ (e.g. decay of a
single tracer with a static distribution), the integral factorizes. Absorbing all
constants into scaled singles $s_a = \sqrt{2\tau \int g^2\, dt}\; S_a$ gives

$$
E[R_{ab}] = s_a\, s_b \qquad \text{for all pairs } (a, b) \in \mathcal{P},
$$

where $\mathcal{P}$ is the set of (unordered) pairs of detection bins in coincidence. This
counts every pair of singles within the window, which matches the PETSIRD coincidence
policy `MULTIPLES_AS_ALL_COINCIDENCES` used by the simulator.

**TOF.** The arrival time difference of a random pair is uniformly distributed over the
coincidence window. If the $n_\text{TOF}$ TOF bins tile the window, the randoms of a pair
are uniformly distributed over the TOF bins:
$E[R_{ab,t}] = s_a s_b / n_\text{TOF}$.

**Fans.** The fan of $a$ is $\mathrm{fan}(a) = \{b : (a,b) \in \mathcal{P}\}$. In PETSIRD,
coincidences are defined per module pair (SGID $\ge 0$ in the module pair SGID LUT). The fan
of $a$ is therefore all detection bins of all modules in coincidence with the module of $a$.
Fan sums of a vector $v$,

$$
(\mathcal{F} v)_a = \sum_{b \in \mathrm{fan}(a)} v_b ,
$$

can be computed from the module totals of $v$ in $O(\#\text{bins} + \#\text{modules}^2)$
operations.

## 2. Delayed coincidences and the likelihood

The delayed window pairs singles of $a$ with singles of $b$ delayed by $\Delta \gg \tau$. It
contains no true coincidences, and its pairs follow the same statistics as the randoms in the
prompt window. The delayeds $D_{ab}$ are therefore independent Poisson variables with

$$
D_{ab} \sim \mathrm{Poisson}(s_a s_b), \qquad (a,b) \in \mathcal{P},
$$

which are independent of the randoms in the prompts. Their log-likelihood is

$$
\ell(s) = \sum_{(a,b)\in\mathcal{P}} \big[ D_{ab} \log(s_a s_b) - s_a s_b \big] + \text{const}
        = \sum_a F_a \log s_a - \sum_{(a,b)\in\mathcal{P}} s_a s_b + \text{const},
$$

with the fan sums of the delayeds

$$
F_a = \sum_{b \in \mathrm{fan}(a)} D_{ab} \qquad (\text{number of delayeds with one end in } a).
$$

So $F$ is a sufficient statistic. The estimation only needs one count per detection bin (one
`bincount` over both ends of all delayed events), not a histogram of the delayeds. There are
$\#\text{bins}$ unknowns (2160 for the simulated scanner) compared with $|\mathcal{P}|$ data
bins (1.36M LORs, times the TOF bins). This is why the estimated randoms are much less noisy
than the delayeds themselves.

With $u_a = \log s_a$, $\ell = \sum_a F_a u_a - \sum_{\mathcal{P}} e^{u_a + u_b}$ is a linear
term minus a sum of convex functions, so it is **concave in $\log s$**. Consequences:

- Every stationary point is a global maximum.
- The randoms $s_a s_b$ of all pairs in $\mathcal{P}$ are unique. If the module coincidence
  graph is bipartite, $s$ itself is only unique up to $s_a \to c\, s_a$ on one side and
  $s_b \to s_b / c$ on the other, which leaves all products unchanged. The graph of the
  simulated scanner contains triangles (e.g. modules 0, 4, 8), so there the singles are
  unique too.
- If $F_a = 0$, the maximum is at the boundary $s_a = 0$.

## 3. Maximum likelihood equations

Setting $\partial \ell / \partial s_a = F_a / s_a - G_a = 0$ with $G_a = (\mathcal{F}s)_a$ gives

$$
s_a\, G_a = F_a \qquad \text{for all } a \text{ with } F_a > 0 ,
$$

which says that the expected fan sums equal the measured fan sums. Summing over $a$ gives
$2 \sum_{\mathcal{P}} s_a s_b = \sum_a F_a = 2 N_D$. So the **total of the ML randoms
estimate equals the number of delayeds** $N_D$.

## 4. Algorithm (`ml_singles`)

**Fixed-point step.** $s_a^{n+1} = F_a / G_a(s^n)$. The overall scale is not damped by this
step: for $s^n = c\, s^\ast$ (with $s^\ast$ the ML solution) we get $G(s^n) = c\, G(s^\ast)$
and hence $s^{n+1} = s^\ast / c$. Linearized in $\log s$, the error propagates as
$\delta u^{n+1} = -M \delta u^n$ with the row-stochastic matrix
$M_{ab} = s_b / G_a$ for $b \in \mathrm{fan}(a)$. Its eigenvalue $1$ (the scale) becomes
$-1$, an undamped oscillation, so the plain fixed-point iteration never converges. In the
study, it stalls with a total that stays 0.09% off.

**Scale step.** The log-likelihood along the direction $s$,
$\ell(c\, s) = \sum_a F_a \log c - c^2 \sum_{\mathcal{P}} s_a s_b + \text{const}$, is maximal for

$$
c^2 = \frac{N_D}{\sum_{\mathcal{P}} s_a s_b} .
$$

This rescaling makes the total estimated randoms equal to the number of delayeds, which every
ML solution satisfies (Section 3). It only uses the measured delayeds and always increases the
likelihood. Applied after every fixed-point step, it removes the oscillating mode. The
remaining modes have $|\lambda| < 1$. For module-based fans, the variations within a module
have $\lambda = 0$ and converge in a single step.

**Safeguard (MM step).** The fixed-point step is not guaranteed to increase $\ell$. If it does
not, the iteration uses a minorize-maximize step instead. With $r = s_b^n / s_a^n$,

$$
s_a s_b \le \tfrac12 \left( s_a^2\, r + s_b^2 / r \right) \quad (\text{equality at } s = s^n),
$$

so $\ell$ is minorized by the separable function
$\sum_a \big[ F_a \log s_a - \tfrac12 s_a^2\, G_a^n / s_a^n \big]$, whose maximizer is

$$
s_a^{n+1} = \sqrt{ s_a^n\, F_a / G_a^n } .
$$

The MM step increases $\ell$ monotonically but converges slowly: in $\log s$ it is half a
fixed-point step, with error factor about $\tfrac12$ per iteration. In the implementation it is
only triggered near machine precision. The full iteration (fixed-point or MM step, followed by
the scale step) is monotone in $\ell$.

Convergence on the simulated scanner (RMS difference of the LOR randoms to the converged ML,
relative to the mean randoms per LOR; see `tests/study_randoms_estimation.py`):

| iterations | 0 (Casey) | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| fixed-point + scale (implemented) | 6e-2 | 3e-2 | 3e-3 | 3e-4 | 3e-6 | 2e-12 |
| MM only | 6e-2 | 4e-2 | 1e-2 | 4e-3 | 5e-4 | 1e-6 |
| fixed-point only | 6e-2 | 2e-2 | 3e-3 | 1.4e-3 | 1.3e-3 | 1.3e-3 |

## 5. Casey-type fan-sum estimate (`casey_singles`, also the initialization of ML)

$$
\hat s_a = \frac{F_a}{\sqrt{\sum_{b \in \mathrm{fan}(a)} F_b}} .
$$

In expectation, $E[F_a] = s_a G_a$ and
$E\big[\sum_{b\in\mathrm{fan}(a)} F_b\big] = \sum_{b \in \mathrm{fan}(a)} s_b G_b = G_a\, \bar G_a$,
where $\bar G_a$ is the $s$-weighted mean of $G_b$ over $\mathrm{fan}(a)$. Hence (ignoring the
noise in the denominator)

$$
\frac{\hat s_a}{s_a} \approx \sqrt{\frac{G_a}{\bar G_a}}, \qquad
\frac{\hat s_a \hat s_b}{s_a s_b} \approx \sqrt{\frac{G_a\, G_b}{\bar G_a\, \bar G_b}} .
$$

The estimate is exact if all fan sums of the singles $G$ are equal (the "Casey assumption").
Its bias depends on how much $G$ varies between fans. It does not depend on the number of
counts. On the simulated scanner, the strongly different module efficiencies and the
off-center activity give module-pair biases of −11% to +14%.

## 6. Use in the LM reconstruction

The additive contamination of a prompt event in detection bins $(a, b)$ is

$$
\bar r_e = \hat s_a \hat s_b / n_\text{TOF} \;\;(\text{TOF}), \qquad
\bar r_e = \hat s_a \hat s_b \;\;(\text{non-TOF}),
$$

and the total over all data bins, needed for the value of the log-likelihood, is
$\sum_{\mathcal{P}} \hat s_a \hat s_b$ (equal to $N_D$ for ML).

**Noise.** To first order, the relative variance of $\hat s_a \hat s_b$ is about
$1/F_a + 1/F_b$. For the raw delayeds of a single bin it is $1/(s_a s_b)$. With fan sums of
hundreds to thousands of counts, compared with $\ll 1$ count per bin, the estimate is orders
of magnitude less noisy (see the study for numbers).

## 7. Assumptions and limitations

- Independent (Poisson) singles. The randoms of a pair are the product of two per-bin factors.
- Delayed and prompt windows have the same width and the same losses (dead time, pile-up).
- All singles share one time course. This fails, for example, for dynamic scans with a
  changing activity distribution or for mixed isotopes. Estimating per time frame helps.
- Coincidences (fans) are defined per module pair (PETSIRD SGID LUT). Element pairs with zero
  module pair efficiency are not excluded from the fans.
- The randoms are uniform over the TOF bins, i.e. the TOF bins tile the coincidence window.
- Detection bins without delayeds get $\hat s_a = 0$ (the ML solution), which only happens
  for very low numbers of delayeds.
