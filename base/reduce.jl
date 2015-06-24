# This file is a part of Julia. License is MIT: http://julialang.org/license

## reductions ##

###### Generic (map)reduce functions ######

if Int === Int32
typealias SmallSigned Union(Int8,Int16)
typealias SmallUnsigned Union(UInt8,UInt16)
else
typealias SmallSigned Union(Int8,Int16,Int32)
typealias SmallUnsigned Union(UInt8,UInt16,UInt32)
end

typealias CommonReduceResult Union(UInt64,UInt128,Int64,Int128,Float32,Float64)
typealias WidenReduceResult Union(SmallSigned, SmallUnsigned, Float16)

# r_promote: promote x to the type of reduce(op, [x])
r_promote(op, x::WidenReduceResult) = widen(x)
r_promote(op, x) = x
r_promote(::AddFun, x::WidenReduceResult) = widen(x)
r_promote(::MulFun, x::WidenReduceResult) = widen(x)
r_promote(::AddFun, x::Number) = x + zero(x)
r_promote(::MulFun, x::Number) = x * one(x)
r_promote(::AddFun, x) = x
r_promote(::MulFun, x) = x
r_promote(::MaxFun, x::WidenReduceResult) = x
r_promote(::MinFun, x::WidenReduceResult) = x
r_promote(::MaxFun, x) = x
r_promote(::MinFun, x) = x


## foldl && mapfoldl

function mapfoldl_impl(f, op, v0, itr, i)
    # Unroll the while loop once; if v0 is known, the call to op may
    # be evaluated at compile time
    if done(itr, i)
        return v0
    else
        (x, i) = next(itr, i)
        v = op(v0, f(x))
        while !done(itr, i)
            (x, i) = next(itr, i)
            v = op(v, f(x))
        end
        return v
    end
end

mapfoldl(f, op, v0, itr) = mapfoldl_impl(f, op, v0, itr, start(itr))

mapfoldl(f, op::Function, v0, itr) = mapfoldl_impl(f, specialized_binary(op), v0, itr, start(itr))

function mapfoldl(f, op, itr)
    i = start(itr)
    if done(itr, i)
        return Base.mr_empty(f, op, eltype(itr))
    end
    (x, i) = next(itr, i)
    v0 = f(x)
    mapfoldl_impl(f, op, v0, itr, i)
end

foldl(op, v0, itr) = mapfoldl(IdFun(), op, v0, itr)
foldl(op, itr) = mapfoldl(IdFun(), op, itr)

## foldr & mapfoldr

function mapfoldr_impl(f, op, v0, itr, i::Integer)
    # Unroll the while loop once; if v0 is known, the call to op may
    # be evaluated at compile time
    if i == 0
        return v0
    else
        x = itr[i]
        v  = op(f(x), v0)
        while i > 1
            x = itr[i -= 1]
            v = op(f(x), v)
        end
        return v
    end
end

mapfoldr(f, op, v0, itr) = mapfoldr_impl(f, op, v0, itr, endof(itr))
mapfoldr(f, op, itr) = (i = endof(itr); mapfoldr_impl(f, op, f(itr[i]), itr, i-1))

foldr(op, v0, itr) = mapfoldr(IdFun(), op, v0, itr)
foldr(op, itr) = mapfoldr(IdFun(), op, itr)

## reduce & mapreduce

# mapreduce_***_impl require ifirst < ilast
function mapreduce_seq_impl(f, op, A::AbstractArray, ifirst::Int, ilast::Int)
    @inbounds fx1 = r_promote(op, f(A[ifirst]))
    @inbounds fx2 = f(A[ifirst+=1])
    @inbounds v = op(fx1, fx2)
    while ifirst < ilast
        @inbounds fx = f(A[ifirst+=1])
        v = op(v, fx)
    end
    return v
end

function mapreduce_pairwise_impl(f, op, A::AbstractArray, ifirst::Int, ilast::Int, blksize::Int)
    if ifirst + blksize > ilast
        return mapreduce_seq_impl(f, op, A, ifirst, ilast)
    else
        imid = (ifirst + ilast) >>> 1
        v1 = mapreduce_pairwise_impl(f, op, A, ifirst, imid, blksize)
        v2 = mapreduce_pairwise_impl(f, op, A, imid+1, ilast, blksize)
        return op(v1, v2)
    end
end

mapreduce(f, op, itr) = mapfoldl(f, op, itr)
mapreduce(f, op, v0, itr) = mapfoldl(f, op, v0, itr)
mapreduce_impl(f, op, A::AbstractArray, ifirst::Int, ilast::Int) =
    mapreduce_pairwise_impl(f, op, A, ifirst, ilast, 1024)

# handling empty arrays
mr_empty(f, op, T) = throw(ArgumentError("reducing over an empty collection is not allowed"))
# use zero(T)::T to improve type information when zero(T) is not defined
mr_empty(::IdFun, op::AddFun, T) = r_promote(op, zero(T)::T)
mr_empty(::AbsFun, op::AddFun, T) = r_promote(op, abs(zero(T)::T))
mr_empty(::Abs2Fun, op::AddFun, T) = r_promote(op, abs2(zero(T)::T))
mr_empty(::IdFun, op::MulFun, T) = r_promote(op, one(T)::T)
mr_empty(::AbsFun, op::MaxFun, T) = abs(zero(T)::T)
mr_empty(::Abs2Fun, op::MaxFun, T) = abs2(zero(T)::T)
mr_empty(f, op::AndFun, T) = true
mr_empty(f, op::OrFun, T) = false

function _mapreduce{T}(f, op, A::AbstractArray{T})
    n = Int(length(A))
    if n == 0
        return mr_empty(f, op, T)
    elseif n == 1
        return r_promote(op, f(A[1]))
    elseif n < 16
        @inbounds fx1 = r_promote(op, f(A[1]))
        @inbounds fx2 = r_promote(op, f(A[2]))
        s = op(fx1, fx2)
        i = 2
        while i < n
            @inbounds fx = f(A[i+=1])
            s = op(s, fx)
        end
        return s
    else
        return mapreduce_impl(f, op, A, 1, n)
    end
end

mapreduce(f, op, A::AbstractArray) = _mapreduce(f, op, A)
mapreduce(f, op, a::Number) = f(a)

mapreduce(f, op::Function, A::AbstractArray) = _mapreduce(f, specialized_binary(op), A)

reduce(op, v0, itr) = mapreduce(IdFun(), op, v0, itr)
reduce(op, itr) = mapreduce(IdFun(), op, itr)
reduce(op, a::Number) = a

### short-circuiting specializations of mapreduce

## helper functions

# obtain return type of functions
# maybe merge with return_types in reflection.jl?
returntype(f, types::Tuple)           = return_types(call, (typeof(f), types...))
returntype(f::Predicate, ::Tuple)     = [Bool]
returntype(f::EqX, types::Tuple)      = [Bool]
returntype(f::IdFun, types::Tuple)    = collect(types)
returntype(f::Function, types::Tuple) = return_types(f, types)

returntype(f::UnspecializedFun, types::Tuple) = return_types(f.f, types)

# conditions and results of short-circuiting
const ShortCircuits = Union{AndFun, OrFun}

iszero(x::Bool) = !x
iszero{T}(x::T) =  x == zero(T)

ismax(x::Bool) = x
ismax{T}(x::T) = x == typemax(T)

shortcircuits{T <: Integer}(::AndFun, x::T) = iszero(x)
shortcircuits{T <: Integer}(::OrFun,  x::T) = ismax(x)

shortcircuits(::ShortCircuits, x) = false

shorted(::AndFun) = false
shorted(::OrFun)  = true

# temporary support for the deprecated (Char,Char) methods of & and |
shortcircuits(::AndFun, x::Char) = iszero(x, T)
shortcircuits(::OrFun,  x::Char) = ismax(x, T)
#---

sc_finish(::AndFun) = true
sc_finish(::OrFun)  = false

# utility macro
# maybe this could be somewhere else?
macro inbounds_if_array(itr, block)
    quote
        if isa($itr, $AbstractArray)
            @inbounds $(esc(block))
        else
            $(esc(block))
        end
    end
end

## short-circuiting definitions

# preliminaries
function mapreduce_sc(f::Func{1}, op::ShortCircuits, itr)
    isempty(itr) && return mr_empty(f,op,itr)

    x = first(itr)
    r = f(x)
    t = typeof(r)

    ftypes = returntype(f, (eltype(itr),))

    if ftypes == [t] && t <: Integer && isleaftype(t)
        shortcircuits(op,r) && return r
        mapreduce_sc(f, op, itr, r)
    else
        _mapreduce(f, op, itr)
    end
end

# loop
function mapreduce_sc{T}(f::Func{1}, op::ShortCircuits, itr, r::T)
    local v::T = r
    @inbounds_if_array itr begin
        for x in itr
            v = op(v, f(x)::T)
            shortcircuits(op, v) && return v
        end
    end
    return v
end

# optimization for boolean results: No need to keep track of previous results
function mapreduce_sc(f::Func{1}, op::ShortCircuits, itr, r::Bool)
    @inbounds_if_array itr begin
        for x in itr
            v::Bool = f(x)::Bool
            shortcircuits(op, v) && return shorted(op)
        end
    end
    return sc_finish(op)
end

mapreduce_sc(f::Function, op::ShortCircuits, itr) =
    mapreduce_sc(specialized_unary(f), op, itr)


# resolve ambiguities
mapreduce(f, op::ShortCircuits, itr::Number) = f(a)

mapreduce(f, op::ShortCircuits, itr::AbstractArray) =
    mapreduce_sc(f, op, itr)

# entry method
mapreduce(f, op::ShortCircuits, itr) =
    mapreduce_sc(f, op, itr)


###### Specific reduction functions ######

## sum

function mapreduce_seq_impl(f, op::AddFun, a::AbstractArray, ifirst::Int, ilast::Int)
    @inbounds begin
        s = r_promote(op, f(a[ifirst])) + f(a[ifirst+1])
        @simd for i = ifirst+2:ilast
            s += f(a[i])
        end
    end
    s
end

# Note: sum_seq usually uses four or more accumulators after partial
# unrolling, so each accumulator gets at most 256 numbers
sum_pairwise_blocksize(f) = 1024

# This appears to show a benefit from a larger block size
sum_pairwise_blocksize(::Abs2Fun) = 4096

mapreduce_impl(f, op::AddFun, A::AbstractArray, ifirst::Int, ilast::Int) =
    mapreduce_pairwise_impl(f, op, A, ifirst, ilast, sum_pairwise_blocksize(f))

sum(f::Union(Callable,Func{1}), a) = mapreduce(f, AddFun(), a)
sum(a) = mapreduce(IdFun(), AddFun(), a)
sum(a::AbstractArray{Bool}) = countnz(a)
sumabs(a) = mapreduce(AbsFun(), AddFun(), a)
sumabs2(a) = mapreduce(Abs2Fun(), AddFun(), a)

# Kahan (compensated) summation: O(1) error growth, at the expense
# of a considerable increase in computational expense.
function sum_kbn{T<:FloatingPoint}(A::AbstractArray{T})
    n = length(A)
    c = r_promote(AddFun(), zero(T)::T)
    if n == 0
        return c
    end
    s = A[1] + c
    for i in 2:n
        @inbounds Ai = A[i]
        t = s + Ai
        if abs(s) >= abs(Ai)
            c += ((s-t) + Ai)
        else
            c += ((Ai-t) + s)
        end
        s = t
    end
    s + c
end


## prod

prod(f::Union(Callable,Func{1}), a) = mapreduce(f, MulFun(), a)
prod(a) = mapreduce(IdFun(), MulFun(), a)

prod(A::AbstractArray{Bool}) =
    error("use all() instead of prod() for boolean arrays")

## maximum & minimum

function mapreduce_impl(f, op::MaxFun, A::AbstractArray, first::Int, last::Int)
    # locate the first non NaN number
    v = f(A[first])
    i = first + 1
    while v != v && i <= last
        @inbounds v = f(A[i])
        i += 1
    end
    while i <= last
        @inbounds x = f(A[i])
        if x > v
            v = x
        end
        i += 1
    end
    v
end

function mapreduce_impl(f, op::MinFun, A::AbstractArray, first::Int, last::Int)
    # locate the first non NaN number
    v = f(A[first])
    i = first + 1
    while v != v && i <= last
        @inbounds v = f(A[i])
        i += 1
    end
    while i <= last
        @inbounds x = f(A[i])
        if x < v
            v = x
        end
        i += 1
    end
    v
end

maximum(f::Union(Callable,Func{1}), a) = mapreduce(f, MaxFun(), a)
minimum(f::Union(Callable,Func{1}), a) = mapreduce(f, MinFun(), a)

maximum(a) = mapreduce(IdFun(), MaxFun(), a)
minimum(a) = mapreduce(IdFun(), MinFun(), a)

maxabs(a) = mapreduce(AbsFun(), MaxFun(), a)
minabs(a) = mapreduce(AbsFun(), MinFun(), a)

## extrema

extrema(r::Range) = (minimum(r), maximum(r))
extrema(x::Real) = (x, x)

function extrema(itr)
    s = start(itr)
    done(itr, s) && throw(ArgumentError("collection must be non-empty"))
    (v, s) = next(itr, s)
    while v != v && !done(itr, s)
        (x, s) = next(itr, s)
        v = x
    end
    vmin = v
    vmax = v
    while !done(itr, s)
        (x, s) = next(itr, s)
        if x > vmax
            vmax = x
        elseif x < vmin
            vmin = x
        end
    end
    return (vmin, vmax)
end

## all & any

# make sure that the specializable unary functions are defined before `any` or `all` are used
# move to functors.jl?
for fun in [:identity, :abs, :abs2, :exp, :log]
    eval(Expr(:function, fun))
end

any(itr) = any(IdFun(), itr)
all(itr) = all(IdFun(), itr)

function any(f, itr)
    specf  = isgeneric(f)? specialized_unary(f) : Predicate(f)
    any(specf, itr)
end

function any(f::Func{1}, itr)
    result = mapreduce_sc(f, OrFun(), itr)
    isa(result, Bool)? result : nonboolean_any(result)
end

function all(f, itr)
    specf  = isgeneric(f)? specialized_unary(f) : Predicate(f)
    all(specf, itr)
end

function all(f::Func{1}, itr)
    result = mapreduce_sc(f, AndFun(), itr)
    isa(result, Bool)? result : nonboolean_all(result)
end

## in & contains

in(x, itr) = any(EqX(x), itr)

const ∈ = in
∉(x, itr)=!∈(x, itr)
∋(itr, x)= ∈(x, itr)
∌(itr, x)=!∋(itr, x)

function contains(eq::Function, itr, x)
    for y in itr
        eq(y, x) && return true
    end
    return false
end


## countnz & count

function count(pred::Union(Callable,Func{1}), itr)
    n = 0
    for x in itr
        pred(x) && (n += 1)
    end
    return n
end

function count(pred::Union(Callable,Func{1}), a::AbstractArray)
    n = 0
    for i = 1:length(a)
        @inbounds if pred(a[i])
            n += 1
        end
    end
    return n
end

immutable NotEqZero <: Func{1} end
call(::NotEqZero, x) = x != 0

countnz(a) = count(NotEqZero(), a)
