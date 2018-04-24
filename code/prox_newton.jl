include("misc.jl")
# Hessian-vector multiplication

function proximal_newton(X, Y, A, C, lambda, tol=1e-4, eps = 1e-4, maxiter = 100)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    
    W = zeros(D, K)
    
    # get graph laplacian
    Diag = zeros(K)
    for i = 1:K
        Diag[i] = sum( A[:,i] ) 
    end
    DG = diagm(Diag)
    L = sparse(DG - A)
    
    iter = 1
    for iter = 1:maxiter
        # construct diagnoal H
        diagH = zeros(round(D*K))
        for k = 1:K
            #@show(Diag[k])
            diagH[ round((k-1)*D+1): round(k*D)] = Diag[k]
        end
        XW = zeros(N, K)
        for k = 1:K
            wk = W[:,k]
            Yk = full(Y[:,k])
            Yk = (Yk-0.5)*2
            score_k = X*wk
            XW[:,k] = score_k
            tmp = score_k.*Yk
            for i = 1:N
                if 1 - tmp[i] > 0
                    xi = XT[:,i]
                    idx, vals = findnz(xi)
                    for pp = 1:length(idx)
                        diagH[ (k-1)*D + idx[pp] ] += 2*C*vals[pp]^2 
                    end
                end
            end
        end
        
        g = get_grad( X, Y, W, L, C, lambda )
        
        #get active set
        rows = zeros(Int64, 0)
        cols = zeros(Int64, 0)
        for j = 1:D
            for k = 1:K
                if abs(g[j,k]) > lambda - eps
                    push!(rows, j)
                    push!(cols, k)
                end
            end
        end
        @printf "size of active set: %d\n" length(rows)
        #coordinate update
        #w = vec(W)
        obj_old = obj(X, Y, W, A, C, lambda)
        #permutation
        perm = randperm(length(rows))
        rows = rows[perm]
        cols = cols[perm]
        
        @printf "start coordinate descent!\n"
        for l = 1:length(rows)
            j = rows[l]
            k = cols[l]
            
            idx = round( (k-1)*D+j )
            h0 = diagH[ idx ]
    
            deno = g[j,k]
    
            # consider the k-th block
            H0w = 0
            Lk = L[:,k]
            idxs, vals = findnz(Lk)
            for pp = 1:length(idxs)
                kk = idxs[pp]
                H0w += 2*W[j,kk]*vals[pp] 
            end
    
            xj = X[:,j]
            idxs, vals = findnz(xj)
            for pp = 1:length(idxs)
                ii = idxs[pp]
                if 1 - Y[ii,k]*XW[ii,k] > 0
                    H0w += 2*C*vals[pp]*XW[ii,k] 
                end
            end
    
            H0w -= h0*W[j,k]
            deno += H0w
            if h0 <= 0
                error("h0 < 0 ") 
            end
            delta_jk = -deno/h0
            w_jk = W[j,k] + delta_jk
            if w_jk > lambda/h0
                w_jk -= lambda/h0 
            elseif w_jk < -lambda/h0
                w_jk += lambda/h0
            else
                w_jk = 0
            end
            
            diff = w_jk - W[j,k]
            
            obj1 = lambda*abs(W[j,k])
            obj2 = h0/2*diff^2 + diff*deno + lambda*abs(W[j,k]+diff)
            
            if obj1 < obj2
                @printf "check coordinate descent: %f, %f" obj1 obj2
                @printf "l: %d\n" l
                error("coordinate descent error") 
            end
            
            #@printf "check coordinate descent: %f, %f" obj1 obj2
            
            
            W[j,k] = w_jk
            
            #update XW
            XW[:,k] += (diff*xj)
            
        end
        
        obj_new = obj(X, Y, W, A, C, lambda)
        @printf "obj_old: %f, obj_new: %f \n" obj_old obj_new
        
        
    end
    return W;
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