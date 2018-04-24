# subgradient of obj
function stochastic_grad_obj(X, Y, W, A, C, lambda, i,j)
    N, D = size(X)
    K = size(W, 2)
    W = sparse(W)
    # i = rand(1:N)    # i -> examples
    # j = rand(1:K)    # j -> classes
    # @printf("i=%d, j=%d\n",i,j)
    Yij = Y[i,j]*2-1
    Xi = X[i,:]
    Wj = W[:,j]
    Aj = A[:,j]
    # L2-hinge loss
    gj = full(-2*C * max(0,1-Yij*Xi'*Wj) * Yij*Xi) * N*K
    # L1-regularization
    gj += lambda * sign.(Wj) * K
    # recursive regularization
    neighbours = find(Aj); l = length(neighbours)
    gn = zeros(D,l)
    for n = 1:l
        gn[:,n] = (W[:,neighbours[n]] - Wj) * K
        # @printf("    neighbour=%d\n",neighbours[n])
    end
    gj -= sum(gn,2)[:]
    # construct gradient
    rows = repmat(1:D, l+1)
    cols = zeros(Int64,(l+1)*D)
    for n = 1:l
        cols[(n-1)*D+1:n*D] = neighbours[n]
    end
    cols[l*D+1:(l+1)*D] = j
    G = sparse(rows,cols,[gn[:]; gj],D,K)
end


function mainStochastic(X, Y, K, A, C, lambda;
                        stepsize=(i)->1e-10, maxIter = 1e4)
    N, D = size(X)
    W = sparse([1],[1],[0.0],D,K)
    obj_val = obj(X, Y, W, A, C, lambda)
    println(@sprintf("Iter = %5d, Obj = %.5e ",0,obj_val))
    for iter=1:maxIter
        i = rand(1:N)    # i -> examples
        j = rand(1:K)    # j -> classes
        G = stochastic_grad_obj(X, Y, W, A, C, lambda, i, j)
        W = W - stepsize(iter)*G
        if (iter%500)==0
            obj_val = obj(X, Y, W, A, C, lambda)
            println(@sprintf("Iter = %5d, i = %d, j = %d, Obj = %.5e ",iter,i,j,obj_val))
        end
    end
    return W
end
