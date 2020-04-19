module Transform

# ---------------------
# Elementwise transform
# ---------------------

export ElementwiseTransform

"""
    ElementwiseTransform{F}

A simple transformation that applies a function elementwise.

# Example
```julia
double_elements = ElementwiseTransform(x -> x * 2)
```
"""
struct ElementwiseTransform{F}
    func::F
end

@inline ElementwiseTransform() = ElementwiseTransform(identity)

@inline (transf::ElementwiseTransform)(x, tile) = transf.func.(x)

end
