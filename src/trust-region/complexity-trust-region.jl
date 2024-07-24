export ComplexityTrustRegion

"""
    ComplexityTrustRegion{T, V} <: AbstractTrustRegion{T, V}

Trust region type that contains the following fields:
- `initial_radius::T`: initial radius;
- `Δ::T`: parameter updated to provide the radius;
- `α::T`: power of the gradient norm in the radius;
- `β::T`: power of the Hessian norm in the radius;
- `radius::T`: current radius;
- `max_radius::T`: upper bound on the radius (default `1 / sqrt(eps(T))`);
- `acceptance_threshold::T`: decrease radius if ratio is below this threshold between 0 and 1 (default `1e-4`);
- `increase_threshold::T`: increase radius if ratio is beyond this threshold between 0 and 1  (default `0.95`);
- `decrease_factor::T`: decrease factor less between 0 and 1 (default `1 / 3`);
- `increase_factor::T`: increase factor greater than one (default `3 / 2`);
- `ratio::T`: current ratio `ared / pred`;
- `gt::V`: pre-allocated memory vector to store the gradient of the objective function;
- `good_grad::Bool`: `true` if `gt` is the gradient of the objective function at the trial point.

The actual trust-region radius is computed as

    Δₖ * ‖∇f(xₖ)‖^α / (1 + ‖Bₖ‖)^β,

where Δₖ is the parameter that is updated when a step is accepted or rejected, and ∇f(xₖ) and Bₖ are the gradient and Hessian approximation at iteration k, respectively.

The following constructors are available:

    ComplexityTrustRegion(gt::T, Δ₀::T; kwargs...)

If `gt` is not known, it is possible to use the following constructors:

    TrustRegion(::Type{V}, n::Int, Δ₀::T; kwargs...)
    TrustRegion(n::Int, Δ₀::T; kwargs...)

that will allocate a vector of size `n` and type `V` or `Vector{T}`.
"""
mutable struct ComplexityTrustRegion{T, V} <: AbstractTrustRegion{T, V}
  Δ₀::T
  Δ::T
  α::T
  β::T
  radius::T
  max_radius::T
  acceptance_threshold::T
  increase_threshold::T
  decrease_factor::T
  increase_factor::T
  ratio::T
  gt::V
  good_grad::Bool

  function ComplexityTrustRegion(
    gt::V,
    Δ₀::T;
    α::T = zero(T),
    β::T = zero(T),
    max_radius::T = one(T) / sqrt(eps(T)),
    acceptance_threshold::T = T(1.0e-4),
    increase_threshold::T = T(0.95),
    decrease_factor::T = one(T) / 3,
    increase_factor::T = 3 * one(T) / 2,
  ) where {T, V}
    Δ₀ > 0 || (Δ₀ = one(T))
    (0 < acceptance_threshold < increase_threshold < 1) ||
      throw(TrustRegionException("Invalid thresholds"))
    (0 < decrease_factor < 1 < increase_factor) ||
      throw(TrustRegionException("Invalid decrease/increase factors"))

    return new{T, V}(
      Δ₀,
      Δ₀,
      α,
      β,
      Δ₀,  # should be updated by calling set_radius!()
      max_radius,
      acceptance_threshold,
      increase_threshold,
      decrease_factor,
      increase_factor,
      zero(T),
      gt,
      false,
    )
  end
end

ComplexityTrustRegion(::Type{V}, n::Int, Δ₀::T; kwargs...) where {T, V} =
  ComplexityTrustRegion(V(undef, n), Δ₀; kwargs...)
ComplexityTrustRegion(n::Int, Δ₀::T; kwargs...) where {T} = ComplexityTrustRegion(Vector{T}, n, Δ₀; kwargs...)

function set_radius!(tr::ComplexityTrustRegion{T, V}, gradient_norm::T, hessian_norm::T) where {T, V}
  tr.radius = compute_radius(tr.Δ, gradient_norm, hessian_norm, tr.α, tr.β, tr.max_radius)
  return tr
end

function compute_radius(Δ::T, gradient_norm::T, hessian_norm::T, α::T, β::T, max_radius::T) where T
  radius = Δ * gradient_norm^α / (1 + hessian_norm)^β
  radius = min(radius, max_radius)
  return radius
end

function update!(tr::ComplexityTrustRegion{T, V}, gradient_norm::T, hessian_norm::T) where {T, V}
  if tr.ratio < tr.acceptance_threshold
    tr.Δ *= tr.decrease_factor
  elseif tr.ratio >= tr.increase_threshold
    tr.Δ *= tr.increase_factor
  end
  set_radius!(tr, gradient_norm, hessian_norm)
  return tr
end
