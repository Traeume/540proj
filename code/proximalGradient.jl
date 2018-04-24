# stolen from Mark Schmidt findmin.jl and misc.jl
### A function to compute the gradient numerically
# func = (W) -> obj(X, Y, W, adj_mat, C, lambda)
function numGrad(func,W;rows=1:size(W,1),cols=1:size(W,2))
	D, K = size(W);
	delta = 2*sqrt(1e-12)*(1+vecnorm(W))
	g = zeros(D, K)
	e_ij = zeros(D, K)
    for j = cols
	    for i = rows
    		e_ij[i,j] = 1
    		(fxp,) = func(W + delta*e_ij)
    		(fxm,) = func(W - delta*e_ij)
    		g[i,j] = (fxp - fxm)/(2*delta)
    		e_ij[i,j] = 0
            println(@sprintf("g[%d,%d]=%e where delta=%e, fxp-fxm=%e",i,j,g[i,j],delta,fxp-fxm))
        end
	end
	return g
end

# computes gradient of the smooth part of obj
function grad_obj(X,Y,W,A,C,lambda)
    N, D = size(X)
    K = size(W, 2)
    W = sparse(W)
    G = sparse([1],[1],[0.0],D,K)
    # find gradient
    for j=1:K
        wj = W[:,j]
        Yj = full(Y[:,j])*2 - 1
        # L2 hinge loss
        tmp = Yj .* max.(0,1-Yj.*(X*wj))
        g = -2 * C * (X' * tmp)
        # Graph recursive regularization
        neighbours = find(A[j,:])
        l = length(neighbours)
        g = g + 2*l*wj
        g = g - 2*sparse(W[:,neighbours]*ones(l))
        G[:,j] = g
    end
    return G
end

# # computes obj and gradient of smooth part of obj
# function obj_grad(X, Y, W, adj_mat, C, lambda)
#     N, D = size(X)
#     K = size(W, 2)
#     res = 0
#     G =
#     for i = 1:K
#         #@show(i)
#         wi = W[:,i]
#         score = X*wi
#         Yi = full(Y[:,i])
#         Yi = (Yi-0.5)*2
#         # L2-hinge loss
#         res += C* sum((max.(0, 1 - score.*Yi)).^2 )
#         # L1 regularization
#         res += lambda*vecnorm(wi, 1)
#     end
#     # recursive regularization
#     rows = rowvals(adj_mat)
#     for i = 1:K
#         tmp = nzrange(adj_mat, i)
#         neighbours = rows[tmp]
#         for j in neighbours
#             res += vecnorm( W[:,i]-W[:,j], 2 )^2 / 2
#         end
#     end
#     return res
# end

function proxGradUpdate(X, Y, W, A, C, lambda; alpha=1, eta=0.01, maxIter=10)
    # alpha is step size
    N, D = size(X)
    K = size(W, 2)
    W = sparse(W)
    L = sparse(1:K,1:K,A*ones(K)) - A
    obj_old = obj(X, Y, W, A, C, lambda)
    # find gradient
    G = get_grad(X, Y, W, L, C, lambda)
    # G1 = grad_obj(X, Y, W, A, C, lambda)
    # if !(G ≈ G1)
    #     println("Two gradients differ.")
    # end

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

function mainProx(X, Y, K, A, C, lambda; maxIter = 100)
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
#
# X, Y = read("../data/diatoms/train_remap.txt")
# K = size(Y,1)
# Y = Y'
# adj_mat = read_cat_hier("../data/diatoms/hr_remap.txt", K)
# mainProx(X, Y, K, adj_mat, 1, 1,maxIter = 25)
