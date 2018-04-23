include("misc.jl")

function Lasso(maxiter=1)

end


function proximal_newtom(X, Y, A, C, lambda, tol=1e-4, maxiter = 100)
    N, D = size(X)
    K = size(Y, 2)
    
    W = zeros(D, K)
    
    iter = 1
    for iter = 1:maxiter
        
    end
    
end

function proximal_gradient(X, Y, A, C, lambda, tol=1e-4, maxiter = 100, max_inner_iter = 20)
    N, D = size(X)
    K = size(Y, 2)
    
    W = zeros(D, K)
    # get graph laplacian
    Diag = zeros(K)
    for i = 1:K
        Diag[i] = sum( A[:,i] ) 
    end
    DG = diagm(Diag)
    L = DG - A
    
    # iterative gradient
    iter = 1
    for iter = 1:maxiter
        g = get_grad( X, Y, W, L, C, lambda ) 
        @printf "get gradient complete\n"
        # prox step
        #d = zeros(D)
        alpha = 1
        obj_old = obj(X, Y, W, A, C, lambda)
        @printf "obj complete\n"
        for inner_iter = 1:max_inner_iter
            threshold = alpha*lambda
            
            W_new = W - alpha*g 
            for jj = 1:D
                for k = 1:K
                    if W_new[jj, k] > threshold
                        W_new[jj, k] -= threshold
                    elseif W_new[jj, k] < -threshold
                        W_new[jj, k] += threshold
                    else
                        W_new[jj,k] = 0
                    end
                end
            end
            @printf "prox complete\n"
            obj_new = obj(X, Y, W_new, A, C, lambda)
            if obj_new < obj_old
                W = W_new
                println("stepsize: ", alpha)
                break 
            end
            alpha = alpha/2 
            if inner_iter == max_inner_iter
                error("linesearch failed") 
            end
        end
        @printf "iter %d finished!\n" iter
    end
    return W
end