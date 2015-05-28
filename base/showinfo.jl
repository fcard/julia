module Info

abstract Parent
abstract Child <: Parent
abstract OperatorCall <: Child

immutable TopLevel <: Parent
end

immutable InfixOperation <: OperatorCall
    prec::Integer
end

immutable UnaryOperation <: OperatorCall
    prec::Integer
end

immutable Block <: Child
end

immutable Default <: Child
end

end
