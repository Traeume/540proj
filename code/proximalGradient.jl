function grad_obj(X,Y,W,A,C,lambda)
    # computes the smooth part of obj
    N, D = size(X)
    K = size(W, 2)
    W = sparse(W)
    G = sparse([1],[1],[0.0],D,K)
    # find gradient
    # G = get_grad(X, Y, W, L, C, lambda)
    for j=1:K
        wj = W[:,j]
        Yj = full(Y[:,j])*2 - 1
        # L2 hinge loss
        tmp = Yj .* max.(0,1-Yj.*(X*wj))
        g = -2 * C * (X' * tmp)
        # Graph recursive regularization
        neighbours = find(A[j,:])
        l = length(neighbours)
        g = g + 4*l*wj
        g = g - sparse(4*W[:,neighbours]*ones(l))
        G[:,j] = g
    end
    return G
end

function proxGradUpdate(X, Y, W, A, C, lambda; alpha=1, eta=0.01, maxIter=10)
    # alpha is step size
    N, D = size(X)
    K = size(W, 2)
    W = sparse(W)
    L = sparse(1:K,1:K,A*ones(K)) - A
    obj_old = obj(X, Y, W, A, C, lambda)
    # find gradient
    G = grad_obj(X, Y, W, A, C, lambda)
    G1 = get_grad(X, Y, W, L, C, lambda)
    if !(G ≈ G1)
        println("Two gradients differ.")
    end

    # line search
    for iter=1:maxIter
        # take gradient step (stepsize alpha)
        Whalf = W - alpha*G
        # take prox
        WNext = sparse(sign.(Whalf) .*
                    max.(0,abs.(Whalf)-alpha*lambda))
        # http://www.seas.ucla.edu/~vandenbe/236C/lectures/proxgrad.pdf 23
        Gm = 1/alpha * (W - WNext) # Gradient Map
        LHS = obj(X, Y, WNext, A, C, lambda)
        RHS = obj_old - alpha * sum(G.*Gm) + alpha/2 * vecnorm(Gm)^2
        println(@sprintf("  iter = %5d, alpha = %.5e, LHS = %.5e, RHS = %.5e", iter, alpha, LHS, RHS))
        if (LHS <= RHS) || iter==maxIter
            W = WNext
            break
        else
            alpha = alpha * eta
        end
    end
    return W
end

function mainProx(X, Y, K, A, C, lambda, alpha; maxIter = 100)
    N, D = size(X)
    W = sparse([1],[1],[0.0],D,K)
    obj_val = obj(X, Y, W, A, C, lambda)
    println(@sprintf("Iter = %5d, Obj = %.5e ",0,obj_val))
    for iter=1:maxIter
        W = proxGradUpdate(X, Y, W, A, C, lambda)
        obj_val = obj(X, Y, W, A, C, lambda)
        println(@sprintf("Iter = %5d, Obj = %.5e ",iter,obj_val))
    end
    return W
end

X, Y = read("../data/diatoms/train_remap.txt")
K = size(Y,1)
Y = Y'
adj_mat = read_cat_hier("../data/diatoms/hr_remap.txt", K)
mainProx(X, Y, K, adj_mat, 1, 1, 0.00005,maxIter = 25)
