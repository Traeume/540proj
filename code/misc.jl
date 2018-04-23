function obj(X, Y, W, adj_mat, C, lambda)
    N, D = size(X)
    K = size(W, 2)
    #Diag = zeros(K)
    #vals = nonzeros(adj_mat)
    #for i = 1:K
    #    tmp = nzrange(adj_mat, i)
    #    Diag[i] = sum( vals[tmp] )
    #end
    #Dmat = sparse(diagm(Diag))
    res = 0
    #score_mat = X * W
    for i = 1:K
        #@show(i)
        wi = W[:,i]
        score = X*wi
        Yi = full(Y[:,i])
        Yi = (Yi-0.5)*2
        # L2-hinge loss
        res += C* sum((max.(0, 1 - score.*Yi)).^2 )
        # L1 regularization
        res += lambda*vecnorm(wi, 1)
    end
    # recursive regularization
    rows = rowvals(adj_mat)
    for i = 1:K
        tmp = nzrange(adj_mat, i)
        neighbours = rows[tmp]
        for j in neighbours
            res += vecnorm( W[:,i]-W[:,j], 2 )^2 
        end
    end
    #L = sparse(Dmat - adj_mat)
    #res += trace( W*L*W' )
    return res
end

function get_grad(X, Y, W, L, C, lambda)
    N, D = size(X)
    K = size(W, 2)
    XT = X';
    res = W*L
    for i = 1:K
        #@printf "i=%d\n" i
        wi = W[:,i]
        score = X*wi
        Yi = full(Y[:,i])
        Yi = (Yi-0.5)*2
        tmp = 1 - Yi.*score
        for j = 1:N
            if tmp[j] > 0
                xj = XT[:,j]
                idx, vals = findnz(xj)
                for pp in length(idx)
                    res[ idx[pp] ,i] += (2*C*tmp[j]*(-Yi[j]))*vals[pp] 
                end
                #res[:,i] += (2*C*tmp[j]*(-Yi[j]))*X[j,:]
            end
        end
    end
    return res
end

function linesearch(X, Y, W, d, g, adj_mat, C, lambda, alpha=1, maxiter=10, eta=0.01)
    #dTd = dot(d,d)
    D, K = size(W)
    d = alpha*d
    gTd = dot(g,d)
    d_reshape = reshape(d, D, K)
    obj_old = obj(X, Y, W, adj_mat, C, lambda)
    for i = 1:maxiter
        obj_new = obj(X, Y, W - alpha*d_reshape, adj_mat, C, lambda)
        if (obj_new - obj_old) <= eta*alpha*gTd
            break
        end
        alpha = alpha/2
        if i == maxiter
            error("line search failed") 
        end
    end
    return alpha
end