"""
# Schroedigner's equation in 1D with Dirichlet boundary conditions.

Schroedigner's equation in 1D with Dirichlet boundary conditions
It models the wavefunction of an electron in an infinite dimensional well.

It is initialized with a sinusoidal profile.
The equation is given by:

```math
i \\epsilon \\frac{\\partial \\psi}{\\partial t} = - \\frac{\\epsilon^2}{2}\\frac{\\partial^2 \\psi}{\\partial x^2}
```
"""
function schroedinger_1d()
    @variables x t ψ(..)
    @parameters ϵ

    Dxx = Differential(x)^2
    Dt = Differential(t)
    V(x) = 0
    ψ0 = x -> sin(2π*x) #exp((im/ϵ)*1e-1*sum(x))
    eqs = [(im*ϵ)*Dt(ψ(t,x)) ~ (-0.5*ϵ^2)Dxx(ψ(t,x)) + V(x)*ψ(t,x)]
    bcs = [ψ(0,x) ~ ψ0(x),
        ψ(t,xmin) ~ 0, #exp((im/ϵ)*(1e-1*sum(xmin) - 0.5e-2*t)),
        ψ(t,xmax) ~ 0] #exp((im/ϵ)*(1e-1*sum(xmax) - 0.5e-2*t))]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]


    tags = ["1D", "Dirichlet", "Linear", "Schroedigner"]

    @named schroedinger_1d = PDESystem(eqs, bcs, domains, [t, x], [u(t, x)], [ϵ => 1.0],
                                metadata = tags)

    schroedinger_1d
end
push!(all_systems, schroedinger_1d())