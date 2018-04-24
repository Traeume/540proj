include("misc.jl")
# Hessian-vector multiplication

function obj_debug(X, Y, W, C)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    XW = X*W
    
    res = 0
    Y_full = 2*(full(Y)-0.5)
    for k = 1:K
        res += vecnorm(W[:,k], 2)^2
        res += C*sum(  (max.(0, 1 - Y_full[:,k].*XW[:,k] ) ).^2 )
    end
    return res
end

function grad_debug(X, Y, W, C)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    XW = X*W
    
    res = W
    for k = 1:K
        wk = W[:,k]
        score_k = X*wk
        Yk = 2*( full(Y[:,k]) - 0.5 )
        tmp = 1 - Yk.*score_k
        for j = 1:N
            if tmp[j] > 0
                xj = XT[:,j]
                idx, vals = findnz(xj)
                for pp = 1:length(idx)
                    res[ idx[pp] ,k] += (2*C*tmp[j]*(-Yk[j]))*vals[pp] 
                end
                #res[:,i] += (2*C*tmp[j]*(-Yi[j]))*xj
            end
        end
        
    end
    return res
    
end

function newton_debug(X, Y, C, maxiter = 100)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    W = zeros(D, K)
    
    for iter = 1:maxiter
        XW = X*W
        delta = zeros(D, K)
        g = grad_debug(X, Y, W, C)
        
        obj = obj_debug(X, Y, W, C)
        @printf "obj: %f\n" obj
        
        for k = 1:K
            score = XW[:,k]
            DD = zeros(N)
            Yk = 2*(full(Y[:,k]) -0.5 )
            for i = 1:N
                if 1 - Yk[i]*score[i] > 0
                    DD[i] = 1 
                end
            end
            Tmp = X
            H = eye(D) + XT*diagm(DD)*X
            d = H \ g
            delta[:,k] = d
        end
        
        W = W - delta
    end
    
end

function Ha(X, Y, W, L, XW, a, C, lambda)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    res = zeros(D*K)
    Ma = reshape(a, D, K)
    
    XMa = X*Ma
    
    rows = rowvals(L)
    vals = nonzeros(L)
    for k = 1:K
        tmp = nzrange(L, k)
        neighbours = rows[tmp]
        vv = vals[tmp]
        for pp = 1:length(neighbours)
            res[ ((k-1)*D+1):(k*D)] = 2*a[ ((neighbours[pp]-1)*D+1):(neighbours[pp]*D)]*vals[pp]
        end
        #res[(k-1)*D +  ]
    end
    
    for k = 1:K
        Xa = XMa[:,k]
        score_k = XW[:,k]
        Yk = full(Y[:,k])
        Yk = (Yk-0.5)*2
        tmp = 1 - Yk.*score_k
        for i = 1:N
            if tmp[i] > 0
                xi = XT[:,i]
                res[((k-1)*D+1):(k*D)] += 2*C*Xa[i]*xi 
            end
        end
        
    end
    
    return res
end
    
    
function proximal_newton(X, Y, A, C, lambda, tol=1e-4, eps = 1e-4, maxiter = 100)
    N, D = size(X)
    K = size(Y, 2)
    XT = X'
    Y_full = 2*(full(Y)-0.5)
    
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
        #if iter >= 2
        #    XW_2 = X*W
        #    vv = vecnorm(XW_2 - XW, 2)^2
        #    @printf "vv: %f\n" vv
        #end
        # construct diagnoal H
        diagH = zeros(round(D*K))
        for k = 1:K
            #@show(Diag[k])
            diagH[ round((k-1)*D+1): round(k*D)] = Diag[k]
        end
        
        XW = X*W
        XW2 = copy(XW)
        
        for k = 1:K
            wk = W[:,k]
            Yk = full(Y[:,k])
            Yk = (Yk-0.5)*2
            #score_k = X*wk
            #XW[:,k] = score_k
            score_k = XW[:,k]
            tmp = score_k.*Yk
            #@show(size(tmp))
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
                if abs(g[j,k]) > lambda - eps && W[j,k] == 0
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
        
        ### check 
        
        check_approx1 = lambda*vecnorm(W, 1) 
        check_d = zeros(D,K)
        ###
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
                    # time consuming?
                if 1 - Y_full[ii,k]*XW[ii,k] > 0
                    H0w += 2*C*vals[pp]*XW2[ii,k] 
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
            XW2[:,k] += (diff*xj)
            
            check_d[j,k] = diff
            
        end
        
        ### check approx
        
        check_approx2 = lambda*vecnorm(W, 1) + dot(vec(g), vec(check_d))
        Hd = Ha(X, Y, W, L, XW, vec(check_d), C, lambda)
        check_approx2 += 0.5*( dot(vec(check_d), Hd) )
        @printf "dot: %f\n" dot(vec(g), vec(check_d))
        @printf "hessian part: %f\n" 0.5*( dot(vec(check_d), Hd) )
        @printf "approx1: %f, approx2: %f\n" check_approx1 check_approx2
        ###
        
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